"""Regression tests for _evaluate_position — score-tiered stops bug fix.

The Signal DB model uses `confidence` (not `score`) for the 0-100 signal
quality rating. A previous bug used `signal.score` which raised
AttributeError on every evaluation, silently breaking the afternoon check.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# Ensure src.output.health can be imported even without pandas_ta
# by stubbing the missing module before any transitive import pulls it in.
if "pandas_ta" not in sys.modules:
    sys.modules["pandas_ta"] = MagicMock()

import src.output.performance as perf


def _make_signal(
    *,
    confidence: float = 85.0,
    stop_loss: float = 98.0,
    target_1: float = 105.0,
    entry_price: float = 100.0,
    holding_period_days: int = 3,
    signal_model: str = "mean_reversion",
    features: dict | None = None,
) -> SimpleNamespace:
    """Build a fake Signal with confidence (not score) — matches DB model."""
    sig = SimpleNamespace(
        id=1,
        confidence=confidence,
        stop_loss=stop_loss,
        target_1=target_1,
        entry_price=entry_price,
        holding_period_days=holding_period_days,
        signal_model=signal_model,
        features=features or {},
    )
    # Ensure `score` does NOT exist — mirrors the real Signal model
    assert not hasattr(sig, "score")
    return sig


def _make_outcome(
    *,
    signal_id: int = 1,
    ticker: str = "TEST",
    entry_date: date = date(2026, 3, 10),
    entry_price: float = 100.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        signal_id=signal_id,
        ticker=ticker,
        entry_date=entry_date,
        entry_price=entry_price,
        still_open=True,
        partial_exit_price=None,
        partial_exit_date=None,
    )


def _make_ohlcv(entry_date: date, days: int = 10, close: float = 101.0) -> pd.DataFrame:
    """Generate a simple OHLCV DataFrame starting from entry_date."""
    rows = []
    d = entry_date
    for i in range(days):
        d = d + timedelta(days=1)
        # Skip weekends
        while d.weekday() >= 5:
            d = d + timedelta(days=1)
        rows.append({
            "date": d,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        })
    return pd.DataFrame(rows)


def _mock_session_and_monkeypatch(monkeypatch, signal):
    """Set up a mock DB session that returns the given signal and patch it in."""
    from contextlib import asynccontextmanager

    # result.scalar_one_or_none() is a sync call on the awaited result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = signal

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result

    @asynccontextmanager
    async def _fake_session():
        yield mock_session

    monkeypatch.setattr(perf, "get_session", _fake_session)


@pytest.mark.asyncio
async def test_evaluate_position_uses_confidence_not_score(monkeypatch):
    """_evaluate_position must not reference signal.score (regression for the
    March 2026 afternoon check outage)."""
    signal = _make_signal(confidence=90.0)
    outcome = _make_outcome()
    df = _make_ohlcv(outcome.entry_date, days=10, close=101.0)

    _mock_session_and_monkeypatch(monkeypatch, signal)

    aggregator = AsyncMock()
    aggregator.get_ohlcv = AsyncMock(return_value=df)

    fake_settings = SimpleNamespace(
        score_tiered_stops_enabled=True,
        trail_activate_pct=0.5,
        trail_distance_pct=0.3,
        partial_tp_enabled=False,
        partial_tp_atr_multiple=1.0,
        partial_tp_fraction=0.5,
        breakeven_after_partial=True,
        sniper_time_stop_days=1,
    )
    monkeypatch.setattr("src.config.get_settings", lambda: fake_settings)
    monkeypatch.setattr(
        "src.utils.trading_calendar.trading_days_between",
        lambda start, end: 5,
    )

    # This must NOT raise AttributeError: 'Signal' has no attribute 'score'
    update_data, result_df = await perf._evaluate_position(outcome, aggregator)

    assert update_data is not None
    assert "pnl_pct" in update_data


@pytest.mark.asyncio
async def test_evaluate_position_expiry_closes_after_hold_period(monkeypatch):
    """Positions held past holding_period_days should close as 'expiry'."""
    signal = _make_signal(confidence=80.0, holding_period_days=3)
    outcome = _make_outcome()
    df = _make_ohlcv(outcome.entry_date, days=10, close=101.0)

    _mock_session_and_monkeypatch(monkeypatch, signal)

    aggregator = AsyncMock()
    aggregator.get_ohlcv = AsyncMock(return_value=df)

    fake_settings = SimpleNamespace(
        score_tiered_stops_enabled=False,
        trail_activate_pct=100.0,  # disable trailing so expiry can fire
        trail_distance_pct=0.3,
        partial_tp_enabled=False,
        partial_tp_atr_multiple=1.0,
        partial_tp_fraction=0.5,
        breakeven_after_partial=True,
        sniper_time_stop_days=1,
    )
    monkeypatch.setattr("src.config.get_settings", lambda: fake_settings)
    monkeypatch.setattr(
        "src.utils.trading_calendar.trading_days_between",
        lambda start, end: 5,
    )

    update_data, _ = await perf._evaluate_position(outcome, aggregator)

    assert update_data is not None
    assert update_data.get("exit_reason") == "expiry"
    assert update_data.get("still_open") is False


@pytest.mark.asyncio
async def test_score_tiered_stops_tiers(monkeypatch):
    """Verify score-tiered stop multipliers for high/mid/low confidence."""
    entry_price = 100.0
    stop_loss = 98.5  # 0.75×ATR → ATR=2.0

    cases = [
        (90.0, entry_price - 1.25 * 2.0),   # high: 1.25×ATR
        (75.0, entry_price - 0.85 * 2.0),   # mid: 0.85×ATR
        (60.0, entry_price - 0.50 * 2.0),   # low: 0.50×ATR
    ]

    for confidence, expected_stop in cases:
        signal = _make_signal(
            confidence=confidence,
            stop_loss=stop_loss,
            entry_price=entry_price,
        )
        outcome = _make_outcome(entry_price=entry_price)
        df = _make_ohlcv(outcome.entry_date, days=5, close=entry_price + 0.1)

        _mock_session_and_monkeypatch(monkeypatch, signal)

        aggregator = AsyncMock()
        aggregator.get_ohlcv = AsyncMock(return_value=df)

        fake_settings = SimpleNamespace(
            score_tiered_stops_enabled=True,
            trail_activate_pct=100.0,  # disable trailing
            trail_distance_pct=0.3,
            partial_tp_enabled=False,
            partial_tp_atr_multiple=1.0,
            partial_tp_fraction=0.5,
            breakeven_after_partial=True,
            sniper_time_stop_days=1,
        )
        monkeypatch.setattr("src.config.get_settings", lambda: fake_settings)
        monkeypatch.setattr(
            "src.utils.trading_calendar.trading_days_between",
            lambda start, end: 1,
        )

        update_data, _ = await perf._evaluate_position(outcome, aggregator)
        assert update_data is not None, f"confidence={confidence} returned None"
