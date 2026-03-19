"""Tests for _evaluate_position — bar-by-bar execution realism.

Covers: T+1 open entry, gap rejection, intraday stop/target, trailing stop
ratcheting, expiry, and score-tiered stops using signal.confidence.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

# Ensure src.output.health can be imported even without pandas_ta
if "pandas_ta" not in sys.modules:
    sys.modules["pandas_ta"] = MagicMock()

import src.output.performance as perf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    *,
    confidence: float = 85.0,
    stop_loss: float = 98.0,
    target_1: float = 105.0,
    entry_price: float = 100.0,
    holding_period_days: int = 3,
    signal_model: str = "mean_reversion",
    features: dict | None = None,
    max_entry_price: float | None = None,
) -> SimpleNamespace:
    sig = SimpleNamespace(
        id=1,
        confidence=confidence,
        stop_loss=stop_loss,
        target_1=target_1,
        entry_price=entry_price,
        holding_period_days=holding_period_days,
        signal_model=signal_model,
        features=features or {},
        max_entry_price=max_entry_price,
    )
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
        skip_reason=None,
    )


def _make_ohlcv_bars(bars: list[dict], start_date: date = date(2026, 3, 10)) -> pd.DataFrame:
    """Build OHLCV from explicit bar dicts. Each dict has open/high/low/close."""
    rows = []
    d = start_date
    for bar in bars:
        while d.weekday() >= 5:
            d = d + timedelta(days=1)
        rows.append({"date": d, **bar})
        d = d + timedelta(days=1)
    return pd.DataFrame(rows)


def _flat_bars(n: int, price: float = 100.5) -> list[dict]:
    """N bars of flat trading at `price`."""
    return [{"open": price, "high": price + 0.5, "low": price - 0.5, "close": price}] * n


def _default_settings(**overrides) -> SimpleNamespace:
    defaults = dict(
        slippage_pct=0.001,
        score_tiered_stops_enabled=False,
        trail_activate_pct=100.0,  # disabled by default
        trail_distance_pct=0.3,
        partial_tp_enabled=False,
        partial_tp_atr_multiple=1.0,
        partial_tp_fraction=0.5,
        breakeven_after_partial=True,
        sniper_time_stop_days=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _patch_deps(monkeypatch, signal, df, settings=None):
    """Patch DB session, aggregator, and settings for _evaluate_position."""
    from contextlib import asynccontextmanager

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = signal
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result

    @asynccontextmanager
    async def _fake_session():
        yield mock_session

    monkeypatch.setattr(perf, "get_session", _fake_session)
    monkeypatch.setattr("src.config.get_settings", lambda: settings or _default_settings())

    aggregator = AsyncMock()
    aggregator.get_ohlcv = AsyncMock(return_value=df)
    return aggregator


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_entry_at_t1_open_with_slippage(monkeypatch):
    """Entry price should be T+1 open * (1 + slippage), not prior close."""
    signal = _make_signal(entry_price=100.0, target_1=110.0, stop_loss=95.0)
    outcome = _make_outcome(entry_price=100.0)
    # First bar (entry day): open=101.0
    bars = [{"open": 101.0, "high": 102.0, "low": 100.5, "close": 101.5}]
    bars += _flat_bars(1, price=101.5)  # still open after 2 bars
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    settings = _default_settings(slippage_pct=0.001)
    aggregator = _patch_deps(monkeypatch, signal, df, settings)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    # entry_price should be 101.0 * 1.001 = 101.101
    assert update is not None
    assert abs(update["entry_price"] - 101.0 * 1.001) < 0.01


@pytest.mark.asyncio
async def test_gap_above_limit_skips_trade(monkeypatch):
    """If T+1 open exceeds max_entry_price, trade should be skipped."""
    signal = _make_signal(entry_price=100.0, max_entry_price=101.0)
    outcome = _make_outcome(entry_price=100.0)
    # Open gaps above limit: 102.0 > 101.0
    bars = [{"open": 102.0, "high": 103.0, "low": 101.5, "close": 102.5}]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    aggregator = _patch_deps(monkeypatch, signal, df)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    assert update["skip_reason"] == "gap_above_limit"
    assert update["still_open"] is False


@pytest.mark.asyncio
async def test_gap_within_limit_fills(monkeypatch):
    """If T+1 open is within max_entry_price, trade should fill normally."""
    signal = _make_signal(entry_price=100.0, max_entry_price=101.0, target_1=110.0, stop_loss=95.0)
    outcome = _make_outcome(entry_price=100.0)
    bars = [{"open": 100.5, "high": 101.0, "low": 100.0, "close": 100.5}]
    bars += _flat_bars(1, price=100.5)
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    aggregator = _patch_deps(monkeypatch, signal, df)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    assert "skip_reason" not in update or update.get("skip_reason") is None
    assert abs(update["entry_price"] - 100.5 * 1.001) < 0.01


@pytest.mark.asyncio
async def test_intraday_stop_triggers(monkeypatch):
    """Low breaching stop should exit even if close recovers above stop."""
    signal = _make_signal(entry_price=100.0, stop_loss=98.0, target_1=110.0)
    outcome = _make_outcome()
    bars = [
        # Entry bar: normal
        {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5},
        # Day 2: low dips to 97.5 (below stop 98.0) but close recovers to 99.0
        {"open": 99.5, "high": 100.0, "low": 97.5, "close": 99.0},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    aggregator = _patch_deps(monkeypatch, signal, df)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    assert update["exit_reason"] == "stop"
    assert update["still_open"] is False
    # Exit at stop price (98.0) minus slippage, not at close (99.0)
    assert update["exit_price"] < 99.0


@pytest.mark.asyncio
async def test_intraday_target_triggers(monkeypatch):
    """High reaching target should exit even if close fades below target."""
    signal = _make_signal(entry_price=100.0, stop_loss=95.0, target_1=103.0)
    outcome = _make_outcome()
    bars = [
        # Entry bar
        {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5},
        # Day 2: high touches 103.5 (above target) but close fades to 101.0
        {"open": 101.0, "high": 103.5, "low": 100.5, "close": 101.0},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    aggregator = _patch_deps(monkeypatch, signal, df)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    assert update["exit_reason"] == "target"
    assert update["still_open"] is False
    # Exit at target price minus slippage, not at close
    assert abs(update["exit_price"] - 103.0 * 0.999) < 0.05


@pytest.mark.asyncio
async def test_gap_through_stop_exits_at_open(monkeypatch):
    """Open below stop should exit at open, not at stop price."""
    signal = _make_signal(entry_price=100.0, stop_loss=98.0, target_1=110.0)
    outcome = _make_outcome()
    bars = [
        {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5},
        # Day 2: gap down through stop — open at 96.0
        {"open": 96.0, "high": 97.0, "low": 95.5, "close": 96.5},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    aggregator = _patch_deps(monkeypatch, signal, df)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update["exit_reason"] == "stop"
    # Exit at open (96.0) minus slippage, worse than stop (98.0)
    assert update["exit_price"] < 97.0


@pytest.mark.asyncio
async def test_gap_through_target_exits_at_open(monkeypatch):
    """Open above target should exit at open, not target price."""
    signal = _make_signal(entry_price=100.0, stop_loss=95.0, target_1=103.0)
    outcome = _make_outcome()
    bars = [
        {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5},
        # Day 2: gap up through target — open at 105.0
        {"open": 105.0, "high": 106.0, "low": 104.5, "close": 105.5},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    aggregator = _patch_deps(monkeypatch, signal, df)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update["exit_reason"] == "target"
    # Exit at open (105.0) minus slippage, better than target (103.0)
    assert update["exit_price"] > 104.0


@pytest.mark.asyncio
async def test_same_bar_stop_and_target_resolves_stop(monkeypatch):
    """When both stop and target are reachable in the same bar, stop wins."""
    signal = _make_signal(entry_price=100.0, stop_loss=98.0, target_1=103.0)
    outcome = _make_outcome()
    bars = [
        {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5},
        # Day 2: huge range — low hits stop AND high hits target
        {"open": 99.0, "high": 104.0, "low": 97.0, "close": 100.0},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    aggregator = _patch_deps(monkeypatch, signal, df)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    # Conservative: stop checked before target
    assert update["exit_reason"] == "stop"


@pytest.mark.asyncio
async def test_trailing_stop_ratchets_across_days(monkeypatch):
    """Trailing stop should ratchet up incrementally, not use end-of-period max."""
    signal = _make_signal(entry_price=100.0, stop_loss=95.0, target_1=115.0)
    outcome = _make_outcome()
    bars = [
        # Day 1 (entry): flat
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.2},
        # Day 2: rally to 102 — triggers trail at 0.5% gain, watermark=102
        {"open": 100.5, "high": 102.0, "low": 100.3, "close": 101.5},
        # Day 3: slight pullback — trail stop = 102 * (1 - 0.003) = 101.694
        # Low = 101.5, still above trail
        {"open": 101.5, "high": 101.8, "low": 101.5, "close": 101.6},
        # Day 4: drops to 101.0 which is below trail stop (101.694)
        {"open": 101.6, "high": 101.7, "low": 101.0, "close": 101.2},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    settings = _default_settings(trail_activate_pct=0.5, trail_distance_pct=0.3)
    aggregator = _patch_deps(monkeypatch, signal, df, settings)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    assert update["exit_reason"] == "trail_stop"
    assert update["still_open"] is False


@pytest.mark.asyncio
async def test_expiry_at_close_after_hold_period(monkeypatch):
    """Position held past hold period exits at close of that bar."""
    signal = _make_signal(
        entry_price=100.0, stop_loss=90.0, target_1=115.0, holding_period_days=3
    )
    outcome = _make_outcome()
    # 3 flat bars — no stop/target hit
    bars = _flat_bars(3, price=101.0)
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    aggregator = _patch_deps(monkeypatch, signal, df)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    assert update["exit_reason"] == "expiry"
    assert update["still_open"] is False


@pytest.mark.asyncio
async def test_confidence_used_for_score_tiered_stops(monkeypatch):
    """Score-tiered stops must use signal.confidence, not signal.score."""
    signal = _make_signal(confidence=90.0, entry_price=100.0, stop_loss=98.5, target_1=110.0)
    outcome = _make_outcome()
    bars = _flat_bars(2, price=100.5)
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    settings = _default_settings(score_tiered_stops_enabled=True)
    aggregator = _patch_deps(monkeypatch, signal, df, settings)

    # Must NOT raise AttributeError: 'Signal' has no attribute 'score'
    update, _ = await perf._evaluate_position(outcome, aggregator)
    assert update is not None
    assert "entry_price" in update
