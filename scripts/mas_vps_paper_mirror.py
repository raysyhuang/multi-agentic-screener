#!/usr/bin/env python3
"""VPS paper-mirror launcher — versioned, fail-closed orchestration of the two-phase pipeline.

Usage:
  python scripts/mas_vps_paper_mirror.py --phase morning [--dry-run] [--out-root PATH]
  python scripts/mas_vps_paper_mirror.py --phase afternoon [--dry-run] [--out-root PATH]

Phase behavior:
  morning   : alembic upgrade head → worker --run-now → export dashboard-data.json
  afternoon : worker --check-now → export dashboard-data.json (NO alembic, NO briefs)

Output layout:
  OUT_ROOT / YYYY-MM-DD / {morning,afternoon} / run-meta.json
                                               / alembic.stdout.log (morning only)
                                               / worker.stdout.log
                                               / export.stdout.log
                                               / dashboard-data.json

Safety:
  Validates settings (trading_mode PAPER, execution_mode quant_only, isolated DB,
  Telegram blank, IBKR not importable) BEFORE running any command. Fail-closed.

This script is the in-repo version of the VPS-only ~hermes/.../mas_vps_paper_mirror.py
— it is NOT a byte-for-byte port. After merge, the VPS can copy or symlink this file.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso_now() -> str:
    """UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _validate_settings_fail_closed() -> dict[str, Any]:
    """Validate that the resolved settings are safe for a PAPER mirror run.

    Returns the resolved settings dict for audit logging.
    Exits non-zero if any fail-closed check fails.
    """
    # Import settings AFTER the environment is established (after argparse, after
    # any caller-set MAS_MIRROR_OUT_ROOT). Do not move this to module-level.
    from src.config import get_settings

    settings = get_settings()
    errors: list[str] = []

    # 1. Trading mode must be PAPER
    if settings.trading_mode != "PAPER":
        errors.append(f"trading_mode must be PAPER (got: {settings.trading_mode!r})")

    # 2. Execution mode must be quant_only (never agentic on the mirror)
    if settings.execution_mode != "quant_only":
        errors.append(f"execution_mode must be quant_only (got: {settings.execution_mode!r})")

    # 3. Database URL must be isolated (postgres, contains marker)
    db_url = settings.database_url
    if not db_url.startswith("postgres"):
        errors.append(f"database_url must be postgres (got: {db_url[:20]}...)")
    # Check for the isolated mirror DB marker (env-settable, default "mas_mirror_")
    db_marker = os.getenv("MAS_MIRROR_DB_MARKER", "mas_mirror_")
    if db_marker and db_marker not in db_url:
        errors.append(
            f"database_url must contain mirror marker {db_marker!r} (got: {db_url})"
        )

    # 4. Telegram credentials must be blank (afternoon lane is artifact-only, no briefs)
    if settings.telegram_bot_token:
        errors.append(
            f"telegram_bot_token must be empty (got non-empty: {settings.telegram_bot_token[:8]}...)"
        )
    if settings.telegram_chat_id:
        errors.append(
            f"telegram_chat_id must be empty (got: {settings.telegram_chat_id!r})"
        )

    # 5. IBKR module must not be importable (paper mirror never executes live broker)
    import importlib.util
    if importlib.util.find_spec("src.broker.ibkr") is not None:
        errors.append("src.broker.ibkr is importable (must not exist in mirror env)")

    if errors:
        print("FAIL-CLOSED: Settings validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    # Return the resolved settings as a dict for audit logging
    return {
        "trading_mode": settings.trading_mode,
        "execution_mode": settings.execution_mode,
        "database_url": db_url[:50] + "..." if len(db_url) > 50 else db_url,
        "telegram_bot_token": "(empty)" if not settings.telegram_bot_token else "(SET)",
        "telegram_chat_id": "(empty)" if not settings.telegram_chat_id else "(SET)",
    }


def _run_step(
    step_name: str, cmd: list[str], log_file: Path, cwd: Path | None = None
) -> dict[str, Any]:
    """Run a command step, stream output to log_file, return step metadata.

    Returns:
      {
        "step": step_name,
        "command": cmd,
        "started_at": iso8601,
        "completed_at": iso8601,
        "duration_s": float,
        "exit_code": int,
        "log_file": str,
      }
    """
    started = _iso_now()
    start_mono = __import__("time").monotonic()
    print(f"[{step_name}] Running: {' '.join(cmd)}")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=cwd,
        )

    duration = __import__("time").monotonic() - start_mono
    completed = _iso_now()
    exit_code = result.returncode

    print(
        f"[{step_name}] Completed in {duration:.2f}s with exit code {exit_code}"
    )

    return {
        "step": step_name,
        "command": cmd,
        "started_at": started,
        "completed_at": completed,
        "duration_s": round(duration, 3),
        "exit_code": exit_code,
        "log_file": str(log_file),
    }


def _resolve_plan(phase: str, out_root: Path, run_date: str, repo_root: Path) -> dict[str, Any]:
    """Resolve the run plan (steps + output paths) for the given phase.

    Does NOT run anything — purely planning.
    """
    phase_dir = out_root / run_date / phase
    steps = []

    if phase == "morning":
        steps.extend([
            {
                "name": "alembic",
                "command": ["alembic", "upgrade", "head"],
                "log_file": str(phase_dir / "alembic.stdout.log"),
                "cwd": str(repo_root),
            },
            {
                "name": "worker",
                "command": [sys.executable, "-m", "src.worker", "--run-now"],
                "log_file": str(phase_dir / "worker.stdout.log"),
                "cwd": str(repo_root),
            },
            {
                "name": "export",
                "command": [
                    sys.executable,
                    "scripts/export_dashboard_data.py",
                    "--out",
                    str(phase_dir / "dashboard-data.json"),
                    "--days",
                    "90",
                ],
                "log_file": str(phase_dir / "export.stdout.log"),
                "cwd": str(repo_root),
            },
        ])
    elif phase == "afternoon":
        steps.extend([
            {
                "name": "worker",
                "command": [sys.executable, "-m", "src.worker", "--check-now"],
                "log_file": str(phase_dir / "worker.stdout.log"),
                "cwd": str(repo_root),
            },
            {
                "name": "export",
                "command": [
                    sys.executable,
                    "scripts/export_dashboard_data.py",
                    "--out",
                    str(phase_dir / "dashboard-data.json"),
                    "--days",
                    "90",
                ],
                "log_file": str(phase_dir / "export.stdout.log"),
                "cwd": str(repo_root),
            },
        ])
    else:
        raise ValueError(f"Unknown phase: {phase!r}")

    return {
        "phase": phase,
        "run_date": run_date,
        "out_root": str(out_root),
        "phase_dir": str(phase_dir),
        "steps": steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="VPS paper-mirror launcher — morning or afternoon phase.",
        epilog=(
            "Morning: alembic + worker --run-now + export. "
            "Afternoon: worker --check-now + export (no alembic, no briefs)."
        ),
    )
    ap.add_argument(
        "--phase",
        required=True,
        choices=["morning", "afternoon"],
        help="Which phase to run: morning (full) or afternoon (check-only)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan and exit without running commands",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root directory (default: MAS_MIRROR_OUT_ROOT env or ./mas_mirror_out)",
    )
    args = ap.parse_args()

    # Resolve output root: --out-root flag > MAS_MIRROR_OUT_ROOT env > default
    out_root = (
        args.out_root
        or Path(os.getenv("MAS_MIRROR_OUT_ROOT", "./mas_mirror_out"))
    )
    out_root = out_root.resolve()

    # Run date: UTC date at invocation time (mirrors GHA's run-date logic)
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Repo root: parent of src/ (where alembic and worker must run from)
    repo_root = Path(__file__).resolve().parent.parent

    # Validate settings BEFORE planning (fail-closed, no wasted planning on unsafe env)
    print("Validating settings (fail-closed)...")
    validated_settings = _validate_settings_fail_closed()
    print("Settings OK:", json.dumps(validated_settings, indent=2))

    # Resolve the plan
    plan = _resolve_plan(args.phase, out_root, run_date, repo_root)
    print("\nResolved plan:")
    print(json.dumps(plan, indent=2))

    if args.dry_run:
        print("\n--dry-run: exiting without running commands.")
        sys.exit(0)

    # Execute the plan
    phase_dir = Path(plan["phase_dir"])
    phase_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "phase": args.phase,
        "run_date": run_date,
        "invoked_at": _iso_now(),
        "out_root": str(out_root),
        "repo_root": str(repo_root),
        "validated_settings": validated_settings,
        "steps": [],
    }

    failed_step = None
    for step_def in plan["steps"]:
        step_result = _run_step(
            step_name=step_def["name"],
            cmd=step_def["command"],
            log_file=Path(step_def["log_file"]),
            cwd=Path(step_def["cwd"]),
        )
        run_meta["steps"].append(step_result)
        if step_result["exit_code"] != 0:
            failed_step = step_result["step"]
            break

    run_meta["completed_at"] = _iso_now()
    run_meta["success"] = failed_step is None
    if failed_step:
        run_meta["failed_step"] = failed_step

    # Write run-meta.json
    meta_path = phase_dir / "run-meta.json"
    with meta_path.open("w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"\nWrote run metadata: {meta_path}")

    if not run_meta["success"]:
        print(f"\nFAILED at step '{failed_step}'", file=sys.stderr)
        sys.exit(1)

    print(f"\n{args.phase.upper()} phase completed successfully.")


if __name__ == "__main__":
    main()
