"""Behaviour tests for the daily brief — the gate that decides whether the
mirror's morning lane runs at all.

These exist because a defect here was invisible for six days: the state file
could not distinguish a skipped day from a successful one, and a skip exited
zero, so the scheduler recorded six consecutive failures as successes.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mas_daily_thread_brief as brief

ET = ZoneInfo("America/New_York")


@pytest.fixture
def env(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "mas_github_pipeline_health.py").write_text("")
    (scripts / "mas_vps_paper_mirror.py").write_text("")
    state = tmp_path / "state.json"
    monkeypatch.setenv("MAS_BRIEF_SCRIPTS", str(scripts))
    monkeypatch.setenv("MAS_BRIEF_STATE", str(state))
    return state


def _at(monkeypatch, when: datetime):
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return when
    monkeypatch.setattr(brief, "datetime", _Clock)


def _invocations(monkeypatch, *, health: str, health_rc: int = 0, mirror: str = "mirror ran"):
    calls = []

    def fake_invoke(script: Path):
        calls.append(script.name)
        if script.name.startswith("mas_github_pipeline_health"):
            return health_rc, health
        return 0, mirror

    monkeypatch.setattr(brief, "invoke", fake_invoke)
    return calls


# ── configuration ────────────────────────────────────────────────────────

def test_fails_closed_without_configuration(monkeypatch, capsys):
    monkeypatch.delenv("MAS_BRIEF_SCRIPTS", raising=False)
    monkeypatch.delenv("MAS_BRIEF_STATE", raising=False)
    assert brief.main() == 1
    assert "failed closed" in capsys.readouterr().out


def test_config_error_names_the_missing_variable(monkeypatch):
    monkeypatch.delenv("MAS_BRIEF_SCRIPTS", raising=False)
    with pytest.raises(brief.ConfigError, match="MAS_BRIEF_SCRIPTS"):
        brief.resolve_config()


# ── the window ───────────────────────────────────────────────────────────

def test_outside_the_window_is_a_silent_no_op(env, monkeypatch, capsys):
    _at(monkeypatch, datetime(2026, 8, 20, 14, 0, tzinfo=ET))  # 14:00 ET
    _invocations(monkeypatch, health="MAS GitHub pipeline healthy")
    assert brief.main() == 0
    assert capsys.readouterr().out == ""


def test_weekend_is_a_silent_no_op(env, monkeypatch, capsys):
    _at(monkeypatch, datetime(2026, 8, 22, 8, 0, tzinfo=ET))  # Saturday
    _invocations(monkeypatch, health="MAS GitHub pipeline healthy")
    assert brief.main() == 0
    assert capsys.readouterr().out == ""


# ── pending early in the window is NORMAL, not a failure ─────────────────

def test_pending_at_07_is_silent_and_exits_zero(env, monkeypatch, capsys):
    """A health check still pending at 07:00 ET has not failed — the GitHub
    morning worker simply has not run yet. It must not go red, must not
    consume the day's delivery, and must leave the streak untouched."""
    _at(monkeypatch, datetime(2026, 8, 20, 7, 0, tzinfo=ET))
    _invocations(monkeypatch, health="MAS GitHub pipeline pending | awaiting")
    assert brief.main() == 0
    assert capsys.readouterr().out == ""
    assert not env.exists(), "an early pending must not mark the day delivered"


# ── giving up at 10:00 ET IS a failure ───────────────────────────────────

def test_unhealthy_at_10_exits_nonzero_and_records_the_skip(env, monkeypatch, capsys):
    _at(monkeypatch, datetime(2026, 8, 20, 10, 0, tzinfo=ET))
    calls = _invocations(monkeypatch, health="MAS GitHub pipeline pending | awaiting")
    rc = brief.main()
    assert rc == 1, "a skipped morning lane must make the scheduler job go red"
    assert "mas_vps_paper_mirror.py" not in calls, "the mirror must not run when unhealthy"
    state = json.loads(env.read_text())
    assert state["skipped"] is True
    assert state["consecutive_skips"] == 1
    assert "skipped" in capsys.readouterr().out


def test_streak_escalates_across_consecutive_skips(env, monkeypatch, capsys):
    for day, expected in ((17, 1), (18, 2), (19, 3), (20, 4)):
        _at(monkeypatch, datetime(2026, 8, day, 10, 0, tzinfo=ET))
        _invocations(monkeypatch, health="MAS GitHub pipeline pending")
        assert brief.main() == 1
        assert json.loads(env.read_text())["consecutive_skips"] == expected
    out = capsys.readouterr().out
    assert "4 CONSECUTIVE DAYS" in out, "a repeated outage must escalate, not repeat itself"


def test_first_skip_does_not_claim_a_streak(env, monkeypatch, capsys):
    _at(monkeypatch, datetime(2026, 8, 20, 10, 0, tzinfo=ET))
    _invocations(monkeypatch, health="MAS GitHub pipeline pending")
    brief.main()
    assert "CONSECUTIVE DAYS" not in capsys.readouterr().out


# ── a real run clears the streak ─────────────────────────────────────────

def test_successful_mirror_run_resets_the_streak(env, monkeypatch):
    _at(monkeypatch, datetime(2026, 8, 19, 10, 0, tzinfo=ET))
    _invocations(monkeypatch, health="MAS GitHub pipeline pending")
    brief.main()
    assert json.loads(env.read_text())["consecutive_skips"] == 1

    _at(monkeypatch, datetime(2026, 8, 20, 8, 0, tzinfo=ET))
    calls = _invocations(monkeypatch, health="MAS GitHub pipeline healthy | ok")
    assert brief.main() == 0
    assert "mas_vps_paper_mirror.py" in calls, "a healthy gate must actually run the mirror"
    state = json.loads(env.read_text())
    assert state["skipped"] is False
    assert state["consecutive_skips"] == 0


def test_one_message_per_day_contract_holds(env, monkeypatch, capsys):
    _at(monkeypatch, datetime(2026, 8, 20, 8, 0, tzinfo=ET))
    _invocations(monkeypatch, health="MAS GitHub pipeline healthy")
    brief.main()
    capsys.readouterr()
    brief.main()  # second invocation the same day
    assert capsys.readouterr().out == "", "the day's brief must be emitted exactly once"


# ── legacy and malformed state ───────────────────────────────────────────

def test_legacy_state_without_the_new_keys_is_readable(env, monkeypatch):
    env.write_text(json.dumps({"date": "2026-08-19"}))
    assert brief.delivered_date(env) == "2026-08-19"
    assert brief.skip_streak(env) == 0, "a legacy file must read as no streak, not crash"


def test_malformed_state_does_not_crash(env, monkeypatch):
    env.write_text("{not json")
    assert brief.delivered_date(env) is None
    assert brief.skip_streak(env) == 0


def test_non_integer_streak_is_tolerated(env):
    env.write_text(json.dumps({"date": "2026-08-19", "consecutive_skips": "many"}))
    assert brief.skip_streak(env) == 0
