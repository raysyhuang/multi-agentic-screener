#!/usr/bin/env python3
"""Emit the single daily MAS GitHub-aligned brief for this Discord thread."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# ── Deployment configuration ─────────────────────────────────────────────
# Supplied by the operator; NO DEFAULTS. This file is in a public repository,
# and a default would publish the scheduler layout of the host that runs it.
# Absent configuration is a fail-closed error, never a fallback.
#
#   MAS_BRIEF_SCRIPTS  directory holding the health check and the launcher
#   MAS_BRIEF_STATE    path to the one-message-per-day state file
ET = ZoneInfo("America/New_York")


class ConfigError(RuntimeError):
    """Deployment configuration is missing or unusable."""


def _required_path(var: str) -> Path:
    raw = os.environ.get(var, "").strip()
    if not raw:
        raise ConfigError(
            f"{var} is not set. This script ships without host defaults so that no "
            f"deployment layout is published in the repository; set it in the "
            f"scheduler unit that invokes it."
        )
    return Path(raw).expanduser()


def resolve_config() -> tuple[Path, Path, Path]:
    scripts = _required_path("MAS_BRIEF_SCRIPTS")
    if not scripts.is_dir():
        raise ConfigError(f"MAS_BRIEF_SCRIPTS is not a directory: {scripts}")
    return scripts / "mas_github_pipeline_health.py", scripts / "mas_vps_paper_mirror.py", _required_path("MAS_BRIEF_STATE")


def combine_output(*, health: str, mirror: str) -> str:
    return f"[MAS] VPS Boston mirror run — GitHub-aligned\n{health.strip()}\n{mirror.strip()}"


def should_emit(today: str, delivered_date: str | None, eastern_hour: int, *, healthy: bool) -> bool:
    if delivered_date == today:
        return False
    return healthy or eastern_hour >= 10


def invoke(script: Path) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(script)], text=True, capture_output=True, timeout=1900, check=False
    )
    output = (result.stdout or result.stderr).strip()
    return result.returncode, output or f"⚠️ no output from {script.name}"


def _state(STATE: Path) -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def delivered_date(STATE: Path) -> str | None:
    return str(_state(STATE).get("date") or "") or None


def skip_streak(STATE: Path) -> int:
    """Consecutive days the mirror was skipped. Reset to 0 by any real run."""
    try:
        return int(_state(STATE).get("consecutive_skips") or 0)
    except (TypeError, ValueError):
        return 0


def mark_delivered(STATE: Path, today: str, *, skipped: bool = False, streak: int = 0) -> None:
    """Record the day, and whether the mirror actually RAN.

    The previous version wrote only {"date": ...}, so a skipped day and a
    successful day were indistinguishable in the state file and the streak was
    unrecoverable. Six consecutive skips (2026-08-14..19) therefore looked
    exactly like six successes to anyone not reading each day's message.
    """
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"date": today, "skipped": skipped, "consecutive_skips": streak}) + "\n",
        encoding="utf-8",
    )
    temporary.replace(STATE)


def main() -> int:
    try:
        HEALTH, MIRROR, STATE = resolve_config()
    except ConfigError as exc:
        # Same fail-closed convention as the launcher: a clear line the
        # scheduler can surface, and a non-zero exit so the job goes red.
        print(f"⚠️ MAS daily brief failed closed: {exc}")
        return 1
    now = datetime.now(ET)
    if now.weekday() >= 5 or not 7 <= now.hour <= 10:
        return 0
    today = now.date().isoformat()
    prior = delivered_date(STATE)
    if prior == today:
        return 0
    health_rc, health = invoke(HEALTH)
    healthy = health_rc == 0 and health.startswith("MAS GitHub pipeline healthy")
    if not healthy:
        if not should_emit(today, prior, now.hour, healthy=False):
            return 0
        streak = skip_streak(STATE) + 1
        reason = (
            "VPS paper mirror skipped: no actual MAS-GH morning worker run was confirmed by 10:00 ET."
            if health.startswith("MAS GitHub pipeline pending")
            else "VPS paper mirror skipped: GitHub authority/alignment is unhealthy."
        )
        if streak > 1:
            reason += f"\n⚠️ SKIPPED {streak} CONSECUTIVE DAYS — the morning lane is not running."
        print(combine_output(health=health, mirror=reason))
        mark_delivered(STATE, today, skipped=True, streak=streak)
        # Non-zero so the scheduler records last_status=error. Previously this
        # returned 0, so a skipped morning lane was recorded as a successful job
        # and the failure was visible only inside one routine daily message.
        # The afternoon wrapper already follows this convention.
        return 1
    mirror_rc, mirror = invoke(MIRROR)
    print(combine_output(health=health, mirror=mirror))
    mark_delivered(STATE, today, skipped=False, streak=0)
    return 0 if mirror_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
