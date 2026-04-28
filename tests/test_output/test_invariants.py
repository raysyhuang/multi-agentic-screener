"""Tests for src/output/invariants.py — alert-only phantom-exit detectors."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Avoid pandas_ta dependency at import time
if "pandas_ta" not in sys.modules:
    sys.modules["pandas_ta"] = MagicMock()

from src.output.invariants import (
    Alert,
    detect_phantom_pnl_fingerprint,
    detect_same_day_sniper_close,
    detect_zero_real_exits,
    format_alert_message,
    run_invariants,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _outcome(
    *,
    outcome_id: int = 1,
    ticker: str = "TEST",
    entry_date: date = date(2026, 4, 22),
    exit_date: date | None = date(2026, 4, 22),
    exit_reason: str | None = "trail_stop",
    pnl_pct: float | None = -0.20,
    still_open: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=outcome_id,
        ticker=ticker,
        entry_date=entry_date,
        exit_date=exit_date,
        exit_reason=exit_reason,
        pnl_pct=pnl_pct,
        still_open=still_open,
    )


def _signal(*, signal_id: int = 1, signal_model: str = "sniper") -> SimpleNamespace:
    return SimpleNamespace(id=signal_id, signal_model=signal_model)


# ---------------------------------------------------------------------------
# P1: same-day sniper close
# ---------------------------------------------------------------------------

def test_p1_fires_for_same_day_sniper_trail_stop():
    o = _outcome(entry_date=date(2026, 4, 22), exit_date=date(2026, 4, 22),
                 exit_reason="trail_stop", pnl_pct=-0.20)
    a = detect_same_day_sniper_close(o, _signal(signal_model="sniper"))
    assert a is not None
    assert a.pattern_id == "P1_SAME_DAY_SNIPER_CLOSE"
    assert a.severity == "HIGH"
    assert a.ticker == "TEST"


def test_p1_fires_for_same_day_sniper_time_stop():
    o = _outcome(entry_date=date(2026, 4, 22), exit_date=date(2026, 4, 22),
                 exit_reason="time_stop", pnl_pct=-9.56)
    a = detect_same_day_sniper_close(o, _signal(signal_model="sniper"))
    assert a is not None
    assert a.pattern_id == "P1_SAME_DAY_SNIPER_CLOSE"


def test_p1_silent_for_mr_same_day_close():
    o = _outcome(entry_date=date(2026, 4, 22), exit_date=date(2026, 4, 22))
    assert detect_same_day_sniper_close(o, _signal(signal_model="mean_reversion")) is None


def test_p1_silent_for_sniper_multi_day_close():
    o = _outcome(entry_date=date(2026, 4, 22), exit_date=date(2026, 4, 24))
    assert detect_same_day_sniper_close(o, _signal()) is None


def test_p1_silent_for_open_outcome():
    o = _outcome(still_open=True, exit_date=None, exit_reason=None)
    assert detect_same_day_sniper_close(o, _signal()) is None


def test_p1_silent_for_real_stop_or_target():
    o = _outcome(entry_date=date(2026, 4, 22), exit_date=date(2026, 4, 22),
                 exit_reason="stop", pnl_pct=-3.2)
    assert detect_same_day_sniper_close(o, _signal()) is None
    o.exit_reason = "target"
    assert detect_same_day_sniper_close(o, _signal()) is None


# ---------------------------------------------------------------------------
# P2: phantom pnl fingerprint
# ---------------------------------------------------------------------------

def test_p2_fires_at_minus_0_20_pct():
    o = _outcome(exit_reason="trail_stop", pnl_pct=-0.20)
    a = detect_phantom_pnl_fingerprint(o, _signal())
    assert a is not None
    assert a.pattern_id == "P2_PHANTOM_PNL_FINGERPRINT"


def test_p2_fires_within_tolerance_band():
    for pnl in (-0.15, -0.18, -0.22, -0.25, -0.29):
        o = _outcome(exit_reason="trail_stop", pnl_pct=pnl)
        assert detect_phantom_pnl_fingerprint(o, _signal()) is not None, pnl


def test_p2_silent_outside_tolerance():
    for pnl in (-0.05, -0.30, -0.50, -1.00, +0.5):
        o = _outcome(exit_reason="trail_stop", pnl_pct=pnl)
        assert detect_phantom_pnl_fingerprint(o, _signal()) is None, pnl


def test_p2_silent_for_non_trail_stop():
    o = _outcome(exit_reason="stop", pnl_pct=-0.20)
    assert detect_phantom_pnl_fingerprint(o, _signal()) is None


def test_p2_silent_for_mr():
    o = _outcome(exit_reason="trail_stop", pnl_pct=-0.20)
    assert detect_phantom_pnl_fingerprint(o, _signal(signal_model="mean_reversion")) is None


# ---------------------------------------------------------------------------
# P3: zero real exits cluster
# ---------------------------------------------------------------------------

def _mock_session_with_rows(rows: list) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.all.return_value = rows
    session.execute.return_value = result
    return session


@pytest.mark.asyncio
async def test_p3_fires_when_no_target_or_real_stop_in_sample():
    rows = [
        (_outcome(exit_reason="trail_stop", pnl_pct=-0.20), _signal())
        for _ in range(12)
    ]
    session = _mock_session_with_rows(rows)
    a = await detect_zero_real_exits(session, lookback_days=14, min_sample=10)
    assert a is not None
    assert a.pattern_id == "P3_NO_REAL_EXITS_CLUSTER"
    assert a.severity == "MEDIUM"
    assert a.evidence["sample_size"] == 12


@pytest.mark.asyncio
async def test_p3_silent_when_target_exits_present():
    rows = [(_outcome(exit_reason="trail_stop"), _signal()) for _ in range(11)]
    rows.append((_outcome(exit_reason="target", pnl_pct=+5.0), _signal()))
    session = _mock_session_with_rows(rows)
    assert await detect_zero_real_exits(session, min_sample=10) is None


@pytest.mark.asyncio
async def test_p3_silent_when_real_stop_present():
    rows = [(_outcome(exit_reason="trail_stop"), _signal()) for _ in range(11)]
    rows.append((_outcome(exit_reason="stop", pnl_pct=-3.0), _signal()))
    session = _mock_session_with_rows(rows)
    assert await detect_zero_real_exits(session, min_sample=10) is None


@pytest.mark.asyncio
async def test_p3_silent_when_sample_too_small():
    rows = [(_outcome(exit_reason="trail_stop"), _signal()) for _ in range(5)]
    session = _mock_session_with_rows(rows)
    assert await detect_zero_real_exits(session, min_sample=10) is None


@pytest.mark.asyncio
async def test_p3_treats_time_stop_as_non_real_exit():
    # time_stop is not target nor real stop — should still count as suspicious
    rows = [(_outcome(exit_reason="time_stop", pnl_pct=-2.0), _signal()) for _ in range(10)]
    session = _mock_session_with_rows(rows)
    a = await detect_zero_real_exits(session, min_sample=10)
    assert a is not None


# ---------------------------------------------------------------------------
# Composite runner
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_invariants_aggregates_per_outcome_and_cluster():
    o1 = _outcome(outcome_id=1, ticker="QXO", entry_date=date(2026, 4, 22),
                  exit_date=date(2026, 4, 22), exit_reason="time_stop", pnl_pct=-9.56)
    o2 = _outcome(outcome_id=2, ticker="NXT", entry_date=date(2026, 4, 27),
                  exit_date=date(2026, 4, 27), exit_reason="trail_stop", pnl_pct=-0.20)
    sig = _signal()
    rows = [(_outcome(exit_reason="trail_stop"), _signal()) for _ in range(10)]
    session = _mock_session_with_rows(rows)
    alerts = await run_invariants(session, [(o1, sig), (o2, sig)])
    pattern_ids = [a.pattern_id for a in alerts]
    # o1: P1 (same-day sniper time_stop)
    # o2: P1 (same-day sniper trail_stop) + P2 (phantom -0.20%)
    # cluster: P3
    assert "P1_SAME_DAY_SNIPER_CLOSE" in pattern_ids
    assert "P2_PHANTOM_PNL_FINGERPRINT" in pattern_ids
    assert "P3_NO_REAL_EXITS_CLUSTER" in pattern_ids
    assert pattern_ids.count("P1_SAME_DAY_SNIPER_CLOSE") == 2  # both o1 and o2


@pytest.mark.asyncio
async def test_run_invariants_silent_on_clean_state():
    rows = [(_outcome(exit_reason="target", pnl_pct=+3.0), _signal()) for _ in range(10)]
    session = _mock_session_with_rows(rows)
    alerts = await run_invariants(session, [])
    assert alerts == []


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------

def test_format_alert_message_groups_by_severity():
    alerts = [
        Alert("P1_SAME_DAY_SNIPER_CLOSE", "HIGH", "QXO", "msg1", {}),
        Alert("P3_NO_REAL_EXITS_CLUSTER", "MEDIUM", None, "msg2", {}),
    ]
    msg = format_alert_message(alerts)
    assert "Sniper Invariant Alarm" in msg
    assert "<b>HIGH</b>" in msg
    assert "<b>MEDIUM</b>" in msg
    assert "QXO" in msg
    assert "P1_SAME_DAY_SNIPER_CLOSE" in msg


def test_format_alert_message_empty_returns_empty_string():
    assert format_alert_message([]) == ""
