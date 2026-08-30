#!/usr/bin/env python3
"""Run the current GitHub MAS code as an isolated, paper-only VPS mirror."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

# ── Deployment configuration ─────────────────────────────────────────────
# Every path is supplied by the operator. There are NO DEFAULTS, deliberately:
# this file lives in a public repository, and a default would publish the
# layout of the host that runs it. Absent configuration is a fail-closed error,
# not a fallback.
#
#   MAS_MIRROR_REPO       checkout the mirror executes (required)
#   MAS_MIRROR_PYTHON     interpreter (optional; defaults to <repo>/.venv/bin/python)
#   MAS_MIRROR_OUT_ROOT   artifact root (required)
#   MAS_MIRROR_ENV_FILES  os.pathsep-separated env overlays, in precedence order (required)
#
# Use --print-config to verify resolution without running anything.


class ConfigError(RuntimeError):
    """Deployment configuration is missing or unusable."""


def _required_path(var: str) -> Path:
    raw = os.environ.get(var, "").strip()
    if not raw:
        raise ConfigError(
            f"{var} is not set. This launcher ships without host defaults so that "
            f"no deployment layout is published in the repository; set it in the "
            f"scheduler unit or wrapper that invokes this script."
        )
    return Path(raw).expanduser()


def resolve_config() -> tuple[Path, Path, Path, tuple[Path, ...]]:
    repo = _required_path("MAS_MIRROR_REPO")
    if not repo.is_dir():
        raise ConfigError(f"MAS_MIRROR_REPO does not exist or is not a directory: {repo}")

    raw_python = os.environ.get("MAS_MIRROR_PYTHON", "").strip()
    python = Path(raw_python).expanduser() if raw_python else repo / ".venv/bin/python"

    out_root = _required_path("MAS_MIRROR_OUT_ROOT")

    raw_envs = os.environ.get("MAS_MIRROR_ENV_FILES", "").strip()
    if not raw_envs:
        raise ConfigError("MAS_MIRROR_ENV_FILES is not set (os.pathsep-separated, in precedence order)")
    env_files = tuple(Path(p).expanduser() for p in raw_envs.split(os.pathsep) if p.strip())
    if not env_files:
        raise ConfigError("MAS_MIRROR_ENV_FILES resolved to no paths")

    return repo, python, out_root, env_files


def load_env_file(path: Path, env: dict[str, str]) -> None:
    if not path.is_file():
        raise ValueError(f"required environment overlay is missing: {path}")
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        if key:
            env[key] = value.strip().strip("'\"")


def build_mirror_env(base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    env.update({
        "TRADING_MODE": "PAPER",
        "EXECUTION_MODE": "quant_only",
        "PEAD_ENABLED": "true",
        "LOCAL_MIRROR_NO_TRADE": "1",
        "LOCAL_MIRROR_NO_TELEGRAM": "1",
        "PRIVATE_SCREENER_MIRROR": "1",
        "PYTHONUNBUFFERED": "1",
        # Explicit empties override a checkout .env under pydantic-settings;
        # removing these keys would let dotenv values re-enable Telegram.
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "",
    })
    return env


def validate_mirror_env(env: dict[str, str]) -> None:
    url = env.get("DATABASE_URL", "")
    if not url:
        raise ValueError("DATABASE_URL is required")
    parsed = urlsplit(url)
    if parsed.scheme not in {"postgres", "postgresql"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("DATABASE_URL must point to isolated local PostgreSQL")
    db_name = parsed.path.lstrip("/")
    if not db_name.startswith("mas_mirror_"):
        raise ValueError("DATABASE_URL must target an isolated mas_mirror_* database")
    if env.get("TRADING_MODE") != "PAPER" or env.get("EXECUTION_MODE") != "quant_only":
        raise ValueError("mirror must run PAPER + quant_only")
    if env.get("TELEGRAM_BOT_TOKEN") or env.get("TELEGRAM_CHAT_ID"):
        raise ValueError("Telegram credentials must be absent in the mirror")


def validate_resolved_settings(settings: dict[str, str]) -> None:
    """Fail closed on the configuration the upstream app actually resolved."""
    if settings.get("telegram_bot_token") or settings.get("telegram_chat_id"):
        raise ValueError("Telegram must resolve to empty credentials in the mirror")
    db_url = settings.get("database_url", "")
    parsed = urlsplit(db_url)
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or not parsed.path.lstrip("/").startswith("mas_mirror_")
    ):
        raise ValueError("resolved DATABASE_URL is not an isolated local mirror database")
    if settings.get("trading_mode") != "PAPER":
        raise ValueError("resolved trading mode must be PAPER")
    if settings.get("execution_mode") != "quant_only":
        raise ValueError("resolved execution mode must be quant_only")


def resolved_settings(env: dict[str, str], python: Path, repo: Path) -> dict[str, str]:
    code = (
        "import json; from src.config import get_settings; s=get_settings(); "
        "print(json.dumps({'telegram_bot_token':s.telegram_bot_token,'telegram_chat_id':s.telegram_chat_id,"
        "'database_url':s.database_url,'trading_mode':s.trading_mode,'execution_mode':s.execution_mode}))"
    )
    result = subprocess.run([str(python), "-c", code], cwd=repo, env=env, text=True, capture_output=True, timeout=90, check=False)
    if result.returncode:
        raise RuntimeError("could not resolve upstream MAS settings")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        # stdout carries database_url and telegram credentials. A JSONDecodeError
        # embeds a snippet of the document it failed on, and main() prints the
        # exception, so the raw message would leak them into the run log.
        raise RuntimeError(
            f"upstream MAS settings were not valid JSON ({exc.msg} at pos {exc.pos}); "
            f"stdout withheld ({len(result.stdout)} bytes)"
        ) from None


def short(value: str) -> str:
    return value[:12] if value else "unknown"


def format_summary(picks: list[dict], *, source_sha: str, artifact: str) -> str:
    names = [
        f"{row.get('ticker', '?')} {row.get('model', '?')} {float(row.get('confidence') or 0):.0f}"
        for row in picks[:6]
    ]
    rendered = ", ".join(names) if names else "no picks"
    return "\n".join([
        "MAS VPS PAPER ONLY — vps_paper_mirror, not official MAS",
        f"source={short(source_sha)} | picks={len(picks)}: {rendered}",
        f"artifact={artifact}",
    ])


def _redact_log(text: str) -> str:
    """Remove query/header credentials before persisting child logs."""
    credential_name = r"(?:key|token|secret|password|passwd|credential|signature|auth)"
    text = re.sub(
        rf"(?i)([?&](?:amp;)?[^?&=\s\"']*{credential_name}[^?&=\s\"']*=)[^&\s\"']+",
        r"\1%2A%2A%2A",
        text,
    )
    text = re.sub(
        rf"(?i)((?:%3f|%26)[^%\s\"']*{credential_name}[^%\s\"']*%3d).*?(?=%26|[\s\"']|$)",
        r"\1%2A%2A%2A",
        text,
    )
    return re.sub(
        r"(?i)([\"']?(?:authorization|proxy-authorization|x-api-key)[\"']?\s*[:=]\s*[\"']?"
        r"(?:(?:bearer|basic)\s+)?)[^\s,;}\]\"']+",
        r"\1***",
        text,
    )


def run(command: list[str], env: dict[str, str], out: Path, repo: Path) -> None:
    result = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True, timeout=1800, check=False)
    (out / f"{Path(command[-1]).name}.stdout.log").write_text(_redact_log(result.stdout), encoding="utf-8")
    (out / f"{Path(command[-1]).name}.stderr.log").write_text(_redact_log(result.stderr), encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")


def _sanitise_remote(url: str) -> str:
    """Strip any embedded credentials from a git remote.

    A regex over `//...@` is wrong: a password containing `@` truncates the
    host. `https://user:p@ss@github.com/a/b.git` becomes `https://ss@github…`,
    which both leaks and misreports. urlsplit handles it, and it is what this
    file already uses for DATABASE_URL.
    """
    if "://" not in url:
        return url  # scp form (git@host:path) carries no password
    parsed = urlsplit(url)
    if not parsed.hostname:
        return url
    netloc = parsed.hostname + (f":{parsed.port}" if parsed.port else "")
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def launcher_provenance() -> dict:
    """Where this file was invoked from, so the run record can distinguish a
    governed checkout from a copy sitting inside one.

    `launcher_sha256` alone proves only that the executed bytes match some
    committed blob — a `cp` produces an identical hash. That rules out a
    *drifted* copy, which is the likelier failure, but not provenance.

    `git status --porcelain` cannot carry that weight either: **ignored files
    produce no status output, so an ignored copy reads as clean.** That is not
    exotic — `.venv/` is ignored, and `.venv/bin/python` is exactly where this
    launcher's own MAS_MIRROR_PYTHON fallback points, so it is a directory
    guaranteed to exist inside the governed checkout on that host. A copy there
    would have reported the real HEAD, the real remote, clean, and a matching
    hash: four fields all saying governed, for a `cp`.

    So trackedness is established by asking HEAD directly:

        git rev-parse HEAD:./<name>   -> the blob recorded at HEAD, or an error
        git hash-object <path>        -> the blob of the bytes on disk

    Equal means the executed bytes ARE the committed blob at that HEAD, and
    trackedness comes free — an untracked or ignored path has no entry at HEAD
    to compare against.

    What is still not proven: that the scheduler chose THIS checkout rather
    than another equally valid one. That is not unanswerable, it is simply not
    answerable *from an artifact* — the choice lives in the scheduler unit that
    sets MAS_MIRROR_REPO, which is host config and not yet versioned.

    Absence is null, never an exception. A launcher outside a checkout is a
    fact worth recording, not a reason to fail a lane whose pipeline already
    succeeded.
    """
    path = Path(__file__).resolve()
    info: dict[str, object] = {
        "launcher_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "launcher_path": str(path),
        "launcher_git_head": None,
        "launcher_git_tracked": None,
        "launcher_git_clean": None,
        "launcher_git_remote": None,
    }

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(path.parent), *args],
                text=True, capture_output=True, timeout=30, check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            # check=False does NOT suppress TimeoutExpired. Letting it escape
            # would fail the lane at manifest time, after the pipeline had
            # already succeeded, discarding run-meta.json.
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    head = _git("rev-parse", "HEAD")
    if head is None:
        return info
    info["launcher_git_head"] = head

    blob_at_head = _git("rev-parse", f"HEAD:./{path.name}")
    info["launcher_git_tracked"] = blob_at_head is not None
    if blob_at_head is not None:
        blob_on_disk = _git("hash-object", str(path))
        info["launcher_git_clean"] = blob_on_disk is not None and blob_on_disk == blob_at_head
    else:
        info["launcher_git_clean"] = False

    remote = _git("remote", "get-url", "origin")
    if remote:
        info["launcher_git_remote"] = _sanitise_remote(remote)
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-config", action="store_true",
                        help="resolve and print deployment configuration, then exit")
    # Two lanes, separate artifact directories. The morning lane opens the book;
    # the afternoon lane marks it to market via --check-now, which is what stamps
    # entry fills and populates pnl_pct. run() names its log files after the last
    # argv token, so a second export into the same directory would overwrite the
    # morning dashboard and leave its run-meta dashboard_sha256 pointing at bytes
    # that no longer exist. Phase subdirectories keep each bundle's hash honest.
    parser.add_argument("--phase", choices=("morning", "afternoon"), default="morning")
    args = parser.parse_args()
    try:
        REPO, PYTHON, OUT_ROOT, ENV_FILES = resolve_config()
        if args.print_config:
            for k, v in (("repo", REPO), ("python", PYTHON), ("out_root", OUT_ROOT)):
                print(f"{k}={v}")
            print("env_files=" + os.pathsep.join(str(p) for p in ENV_FILES))
            return 0
        base = dict(os.environ)
        for path in ENV_FILES:
            load_env_file(path, base)
        env = build_mirror_env(base)
        validate_mirror_env(env)
        validate_resolved_settings(resolved_settings(env, PYTHON, REPO))
        if args.validate_only:
            print("MAS VPS paper mirror configuration valid")
            return 0
        if not PYTHON.is_file():
            raise RuntimeError(f"missing Python 3.12 project venv: {PYTHON}")
        # Partition on the EXCHANGE date, not UTC. A run between 00:00-04:00 UTC
        # is still the previous trading day in New York, and partitioning on UTC
        # would file it under tomorrow. #78 was this same defect in a smoke test.
        run_date = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        out = OUT_ROOT / run_date / args.phase
        out.mkdir(parents=True, exist_ok=True)
        if args.phase == "morning":
            run([str(PYTHON), "-m", "alembic", "upgrade", "head"], env, out, REPO)
            run([str(PYTHON), "-m", "src.worker", "--run-now"], env, out, REPO)
        else:
            # No alembic: the checkout only fast-forwards on the morning lane, so
            # there is nothing to migrate and it is a needless failure surface.
            run([str(PYTHON), "-m", "src.worker", "--check-now"], env, out, REPO)
        dashboard = out / "dashboard-data.json"
        run([str(PYTHON), "scripts/export_dashboard_data.py", "--out", str(dashboard), "--days", "90"], env, out, REPO)
        manifest = {
            # run_date is the partition key and is now the EXCHANGE date.
            # run_date_utc is retained (7 prior artifacts carry it) but now holds
            # the actual UTC date, so neither key is ever a mislabelled value.
            "run_date": run_date,
            "run_date_tz": "America/New_York",
            "run_date_utc": datetime.now(UTC).date().isoformat(),
            "phase": args.phase,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "source_tier": "vps_paper_mirror_of_github_main",
            # Content hash of THIS FILE. source_sha below is `git rev-parse
            # HEAD` in the pipeline checkout: it identifies the pipeline, not
            # the launcher.
            #
            # WHAT THIS PROVES, precisely: that the bytes executed match a
            # committed revision. Compare against the exact commit recorded in
            # source_sha:
            #     git show <source_sha>:scripts/mas_vps_paper_mirror.py | sha256sum
            # A mismatch means the run came from code that is not in the repo,
            # and the artifact is inadmissible.
            #
            # WHAT IT DOES NOT PROVE: that the scheduler invoked the repository
            # copy. A host copy with identical bytes passes identically. This
            # establishes code-content equivalence only; repointing the
            # scheduler is a separate step and is not evidenced here.
            **launcher_provenance(),
            "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "trading_mode": env["TRADING_MODE"],
            "execution_mode": env["EXECUTION_MODE"],
            "pead_enabled": env["PEAD_ENABLED"],
            "telegram_disabled": True,
            "dashboard_sha256": hashlib.sha256(dashboard.read_bytes()).hexdigest(),
            "dashboard": str(dashboard),
        }
        (out / "run-meta.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload = json.loads(dashboard.read_text(encoding="utf-8"))
        print(format_summary(
            payload.get("today_picks", []),
            source_sha=manifest["source_sha"],
            artifact=str(dashboard),
        ))
        return 0
    except Exception as exc:  # noqa: BLE001 - fail closed on ANY fault; a mirror that
        # half-runs is worse than one that refuses, and the message is the alert.
        print(f"⚠️ MAS VPS PAPER mirror failed closed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
