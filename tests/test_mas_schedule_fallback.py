"""Tests for the external GitHub-schedule fallback and workflow dedupe guard."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mas_schedule_fallback as fallback  # noqa: E402

SHA = "a" * 40
NOW = datetime(2026, 8, 31, 10, 32, tzinfo=UTC)  # Monday 06:32 EDT


def run(run_id: int, *, event="schedule", status="completed", conclusion: str | None = "success", sha=SHA):
    return {
        "databaseId": run_id,
        "event": event,
        "status": status,
        "conclusion": conclusion,
        "headSha": sha,
        "createdAt": "2026-08-31T10:17:00Z",
    }


def jobs(conclusion="success"):
    return [{
        "name": "Run scheduled pipeline",
        "steps": [{"name": "Run morning pipeline", "conclusion": conclusion}],
    }]


def test_watchdog_dispatches_when_no_current_run_exists():
    assert fallback.needs_dispatch([], {}, SHA, NOW) is True


def test_actual_scheduled_worker_prevents_fallback():
    assert fallback.needs_dispatch([run(1)], {1: jobs()}, SHA, NOW) is False


def test_dst_noop_does_not_prevent_fallback():
    assert fallback.needs_dispatch([run(1)], {1: jobs("skipped")}, SHA, NOW) is True


def test_inflight_current_sha_run_prevents_duplicate_dispatch():
    candidate = run(1, status="in_progress", conclusion=None)
    assert fallback.needs_dispatch([candidate], {}, SHA, NOW) is False


def test_old_sha_does_not_satisfy_current_authority():
    assert fallback.needs_dispatch([run(1, sha="b" * 40)], {1: jobs()}, SHA, NOW) is True


def test_successful_fallback_makes_late_schedule_skip():
    peer = run(1, event="repository_dispatch")
    assert fallback.workflow_should_run(
        "schedule", [peer], {1: jobs()}, current_run_id=2, head_sha=SHA, now=NOW,
    ) is False


def test_successful_schedule_makes_fallback_skip():
    peer = run(1, event="schedule")
    assert fallback.workflow_should_run(
        "repository_dispatch", [peer], {1: jobs()}, current_run_id=2, head_sha=SHA, now=NOW,
    ) is False


def test_duplicate_fallback_dispatch_is_suppressed_after_serial_peer_completes():
    peer = run(1, event="repository_dispatch")
    assert fallback.workflow_should_run(
        "repository_dispatch", [peer], {1: jobs()}, current_run_id=2, head_sha=SHA, now=NOW,
    ) is False


def test_failed_or_noop_peer_does_not_block_recovery():
    failed = run(1, event="repository_dispatch", conclusion="failure")
    noop = run(2, event="schedule")
    assert fallback.workflow_should_run(
        "schedule", [failed], {1: jobs(), 2: jobs("skipped")},
        current_run_id=3, head_sha=SHA, now=NOW,
    ) is True


def test_manual_dispatch_is_not_changed_by_automatic_dedupe():
    assert fallback.workflow_should_run(
        "workflow_dispatch", [], {}, current_run_id=1, head_sha=SHA, now=NOW,
    ) is True


def test_fallback_window_tracks_eastern_time_and_weekdays():
    assert fallback.in_fallback_window(NOW) is True
    assert fallback.fallback_window_state(NOW) == "active"
    assert fallback.in_fallback_window(datetime(2026, 8, 31, 11, 32, tzinfo=UTC)) is True
    assert fallback.in_fallback_window(datetime(2026, 8, 31, 13, 32, tzinfo=UTC)) is False
    assert fallback.fallback_window_state(
        datetime(2026, 8, 31, 13, 32, tzinfo=UTC)
    ) == "missed"
    assert fallback.in_fallback_window(datetime(2026, 8, 30, 10, 32, tzinfo=UTC)) is False
    assert fallback.fallback_window_state(
        datetime(2026, 8, 30, 10, 32, tzinfo=UTC)
    ) == "weekend"
