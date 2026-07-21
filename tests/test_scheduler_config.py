"""Tests for scheduler hardening in the active main entrypoint."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import src.main as main


def test_start_scheduler_uses_one_hour_misfire_grace(monkeypatch):
    """The production scheduler should tolerate a 1h restart window."""

    captured: dict[str, object] = {}

    class _FakeScheduler:
        def __init__(self, timezone):
            self.timezone = timezone
            self.jobs: list[dict] = []
            self.started = False
            captured["scheduler"] = self

        def add_listener(self, handler, event):
            captured["listener_event"] = event

        def add_job(self, func, trigger, **kwargs):
            self.jobs.append({"func": func, "trigger": trigger, **kwargs})

        def start(self):
            self.started = True

        def get_jobs(self):
            return [
                SimpleNamespace(
                    name=job["name"],
                    next_run_time="2026-03-13 09:30:00-04:00",
                )
                for job in self.jobs
            ]

    class _FakeCronTrigger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "apscheduler.schedulers.asyncio",
        SimpleNamespace(AsyncIOScheduler=_FakeScheduler),
    )
    monkeypatch.setitem(
        sys.modules,
        "apscheduler.triggers.cron",
        SimpleNamespace(CronTrigger=_FakeCronTrigger),
    )
    monkeypatch.setitem(
        sys.modules,
        "apscheduler.events",
        SimpleNamespace(EVENT_JOB_ERROR="job_error"),
    )
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(
            morning_run_hour=9,
            morning_run_minute=30,
            afternoon_check_hour=15,
            afternoon_check_minute=30,
        ),
    )

    scheduler = main.start_scheduler()

    assert scheduler.started is True
    # Morning pipeline + afternoon check (weekly meta-review removed with LLM stack)
    assert len(scheduler.jobs) == 2
    assert {job["misfire_grace_time"] for job in scheduler.jobs} == {3600}


# ---------------------------------------------------------------------------
# Workflow <-> worker one-off flag consistency (2026-07 regression)
# ---------------------------------------------------------------------------
# The lean strip removed --collect-now/--meta-now from worker.py but the
# GitHub Actions workflow kept scheduling them; an unknown flag then silently
# fell through into the blocking scheduler loop (a hung 6h CI job, only
# avoided by a lucky DST-guard skip). Lock both sides.

def test_workflow_flags_exist_in_worker():
    """Every --*-now flag referenced by scheduled-pipelines.yml must be a
    valid worker one-off flag."""
    import re
    from pathlib import Path

    workflow = Path(".github/workflows/scheduled-pipelines.yml").read_text()
    referenced = set(re.findall(r"--[a-z]+-now", workflow))
    valid = {"--run-now", "--check-now"}  # keep in sync with worker one_off_jobs
    assert referenced, "workflow references no worker flags — mapping removed?"
    assert referenced <= valid, f"workflow references removed flags: {referenced - valid}"


def test_worker_unknown_flag_exits_nonzero(monkeypatch):
    """An unrecognized worker flag must exit(2), never start the scheduler."""
    import asyncio
    from unittest.mock import AsyncMock

    import src.worker as worker

    monkeypatch.setattr(worker, "init_db", AsyncMock())
    monkeypatch.setattr(
        worker, "get_settings",
        lambda: SimpleNamespace(validate_keys_for_mode=lambda: None),
    )
    monkeypatch.setattr(sys, "argv", ["worker", "--collect-now"])

    import pytest
    with pytest.raises(SystemExit) as exc:
        asyncio.run(worker.start_worker())
    assert exc.value.code == 2
