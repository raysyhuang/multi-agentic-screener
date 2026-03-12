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
            cross_engine_enabled=True,
        ),
    )

    scheduler = main.start_scheduler()

    assert scheduler.started is True
    assert len(scheduler.jobs) == 4
    assert {job["misfire_grace_time"] for job in scheduler.jobs} == {3600}
