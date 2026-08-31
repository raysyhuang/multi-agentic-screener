#!/usr/bin/env python3
"""Dispatch a governed GitHub morning fallback when Actions cron is late.

The host watchdog is silent when an actual/in-flight current-main run exists.
Inside Actions, the same module makes a late scheduled run and an already
successful fallback mutually exclusive. Workflow-level concurrency serializes
the two events so this completed-peer check cannot make both sides skip.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

GITHUB_REPO = "raysyhuang/multi-agentic-screener"
WORKFLOW_FILE = "scheduled-pipelines.yml"
FALLBACK_EVENT = "mas-morning-fallback"
FALLBACK_ACTOR = "raysyhuang"
ET = ZoneInfo("America/New_York")
AUTOMATIC_EVENTS = {"schedule", "repository_dispatch"}


def run_text(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, timeout=90, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")
    return result.stdout.strip()


def github_runs() -> list[dict]:
    return json.loads(run_text([
        "gh", "run", "list", "--repo", GITHUB_REPO,
        "--workflow", WORKFLOW_FILE, "--limit", "20", "--json",
        "databaseId,event,status,conclusion,headSha,createdAt,url",
    ]))


def job_steps(run_id: int) -> list[dict]:
    return json.loads(run_text([
        "gh", "run", "view", str(run_id), "--repo", GITHUB_REPO, "--json", "jobs",
    ])).get("jobs", [])


def worker_ran(jobs: list[dict]) -> bool:
    for job in jobs:
        if job.get("name") != "Run scheduled pipeline":
            continue
        for step in job.get("steps") or []:
            if step.get("name") == "Run morning pipeline" and step.get("conclusion") == "success":
                return True
    return False


def et_date(run: dict):
    raw = str(run["createdAt"])
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw).astimezone(ET).date()


def _matches_today(run: dict, head_sha: str, now: datetime) -> bool:
    return (
        run.get("event") in AUTOMATIC_EVENTS
        and run.get("headSha") == head_sha
        and et_date(run) == now.astimezone(ET).date()
    )


def needs_dispatch(
    runs: list[dict], jobs_by_id: dict[int, list[dict]], head_sha: str, now: datetime,
) -> bool:
    """False when current-main authority is already running or actually ran."""
    for candidate in runs:
        if not _matches_today(candidate, head_sha, now):
            continue
        if candidate.get("status") in {"queued", "in_progress", "waiting", "pending"}:
            return False
        if candidate.get("status") == "completed" and candidate.get("conclusion") == "success":
            run_id = int(candidate["databaseId"])
            if worker_ran(jobs_by_id.get(run_id, [])):
                return False
    return True


def workflow_should_run(
    event: str,
    runs: list[dict],
    jobs_by_id: dict[int, list[dict]],
    *,
    current_run_id: int,
    head_sha: str,
    now: datetime,
) -> bool:
    """Skip when any other automatic current-main worker already completed."""
    if event not in AUTOMATIC_EVENTS:
        return True
    for candidate in runs:
        if int(candidate.get("databaseId") or 0) == current_run_id:
            continue
        if not _matches_today(candidate, head_sha, now):
            continue
        if candidate.get("status") != "completed" or candidate.get("conclusion") != "success":
            continue
        run_id = int(candidate["databaseId"])
        if worker_ran(jobs_by_id.get(run_id, [])):
            return False
    return True


def in_fallback_window(now: datetime) -> bool:
    local = now.astimezone(ET)
    minute = local.hour * 60 + local.minute
    return local.weekday() < 5 and 6 * 60 + 27 <= minute < 9 * 60


def fallback_window_state(now: datetime) -> str:
    local = now.astimezone(ET)
    if local.weekday() >= 5:
        return "weekend"
    minute = local.hour * 60 + local.minute
    if minute < 6 * 60 + 27:
        return "before"
    if minute < 9 * 60:
        return "active"
    return "missed"


def _jobs_for_completed_successes(runs: list[dict], now: datetime, head_sha: str) -> dict[int, list[dict]]:
    return {
        int(run["databaseId"]): job_steps(int(run["databaseId"]))
        for run in runs
        if _matches_today(run, head_sha, now)
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
    }


def current_main_sha() -> str:
    return run_text(["gh", "api", f"repos/{GITHUB_REPO}/commits/main", "--jq", ".sha"])


def dispatch_fallback(*, expected_date: str, expected_sha: str) -> None:
    actor = run_text(["gh", "api", "user", "--jq", ".login"])
    if actor != FALLBACK_ACTOR:
        raise RuntimeError("authenticated GitHub actor is not the configured fallback actor")
    payload = json.dumps({
        "event_type": FALLBACK_EVENT,
        "client_payload": {"expected_date": expected_date, "expected_sha": expected_sha},
    })
    result = subprocess.run(
        ["gh", "api", "--method", "POST", f"repos/{GITHUB_REPO}/dispatches", "--input", "-"],
        input=payload,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"fallback dispatch failed ({result.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-should-run", action="store_true")
    args = parser.parse_args()

    now = datetime.now(UTC)
    if not args.workflow_should_run:
        window = fallback_window_state(now)
        if window in {"weekend", "before"}:
            return 0
        if window == "missed":
            raise RuntimeError("watchdog invocation missed the governed 06:27-09:00 ET window")
    head_sha = os.environ.get("GITHUB_SHA", "").strip() if args.workflow_should_run else current_main_sha()
    if not head_sha:
        raise RuntimeError("current GitHub main SHA is unavailable")
    runs = github_runs()
    jobs_by_id = _jobs_for_completed_successes(runs, now, head_sha)

    if args.workflow_should_run:
        event = os.environ.get("GITHUB_EVENT_NAME", "")
        current_id = int(os.environ.get("GITHUB_RUN_ID", "0"))
        print("true" if workflow_should_run(
            event, runs, jobs_by_id, current_run_id=current_id, head_sha=head_sha, now=now,
        ) else "false")
        return 0

    if not needs_dispatch(runs, jobs_by_id, head_sha, now):
        return 0

    expected_date = now.astimezone(ET).date().isoformat()
    dispatch_fallback(expected_date=expected_date, expected_sha=head_sha)
    print(
        "⚠️ MAS GitHub cron fallback dispatched"
        f" | ET date={expected_date} | main={head_sha[:12]}"
        " | reason=no actual or in-flight current-main morning worker by fallback gate"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - watchdog failures must alert, not vanish
        print(f"⚠️ MAS GitHub cron fallback failed closed: {type(exc).__name__}: {exc}")
        sys.exit(1)
