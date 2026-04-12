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
    daily_prices: dict | None = None,
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
        daily_prices=daily_prices,
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
async def test_newly_armed_trail_does_not_exit_same_bar_via_open(monkeypatch):
    """Regression for the phantom `trail_stop` bug: on the bar where the
    trail first arms, the newly-computed trail level must not clip the bar's
    own open via a same-bar gap-through.

    Before the fix, the sim did (update HWM from bar_high) → (arm trail) →
    (compute trail_stop above bar_open) → (exit because bar_open <= trail_stop).
    That produced exits at bar_open * (1 - slippage) ≈ entry * 0.998 (-0.2%)
    even though the bar was a winner (e.g. EQT 2026-04-06: open 59.55, high
    60.91, close 60.40 recorded as a -0.20% trail_stop loss).

    Narrow-fix invariant: a newly-armed trail only enforces from the NEXT
    bar. Ratcheting behavior for already-active trails is unchanged.
    """
    signal = _make_signal(entry_price=100.0, stop_loss=95.0, target_1=115.0, holding_period_days=3)
    outcome = _make_outcome()
    # Entry bar: opens 100, rallies to 102.18 (gain 2.18% — above 0.5% arm),
    # low 99.47 (above base stop 95), close 100.40. This is a winning day.
    # Plus a calm following bar so if an exit happens at all it must not be
    # the phantom -0.20% fingerprint.
    bars = [
        {"open": 100.0, "high": 102.18, "low": 99.47, "close": 100.40},
        {"open": 100.40, "high": 100.60, "low": 100.20, "close": 100.50},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    settings = _default_settings(trail_activate_pct=0.5, trail_distance_pct=0.3)
    aggregator = _patch_deps(monkeypatch, signal, df, settings)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    # Must NOT produce the phantom -0.20% fingerprint (entry_fill * 0.998).
    if "exit_price" in update:
        entry_fill = update.get("entry_price", outcome.entry_price)
        phantom_price = entry_fill * 0.998
        assert abs(update["exit_price"] - phantom_price) > 0.01, (
            f"Phantom trail_stop regression: exit at {update['exit_price']} "
            f"matches phantom signature {phantom_price}"
        )


@pytest.mark.asyncio
async def test_newly_armed_trail_does_not_exit_same_bar_via_low(monkeypatch):
    """On the activation bar, bar_low dipping below the newly-computed trail
    level must not trigger an exit — daily OHLC can't prove the low came
    after the high. The stop should instead fire on a later bar (where the
    trail is in force from the prior bar's HWM)."""
    signal = _make_signal(entry_price=100.0, stop_loss=90.0, target_1=115.0, holding_period_days=5)
    outcome = _make_outcome()
    bars = [
        # Day 1 (entry): high 102 arms trail; low 100.3 is below the same-bar
        # trail (102 * 0.997 = 101.694). Under narrow fix, must NOT exit day 1.
        {"open": 100.0, "high": 102.0, "low": 100.3, "close": 101.5},
        # Day 2: flat. bar_low 101.50 is below yesterday's armed trail (101.694),
        # so the trail fires now (ratchet path — already-active behavior).
        {"open": 101.55, "high": 101.80, "low": 101.50, "close": 101.60},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    settings = _default_settings(trail_activate_pct=0.5, trail_distance_pct=0.3)
    aggregator = _patch_deps(monkeypatch, signal, df, settings)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    assert update["exit_reason"] == "trail_stop"
    # Exit on day 2 (not day 1). Exit price should be near yesterday's
    # ratcheted trail (101.694 * 0.999 ≈ 101.59), not day 1's low.
    assert update["exit_price"] > 101.0, (
        f"Expected day-2 trail exit near 101.59; got {update['exit_price']}"
    )


@pytest.mark.asyncio
async def test_trail_already_active_ratchets_and_exits_same_bar(monkeypatch):
    """Codex-requested regression: when a trail is already active from a
    prior bar and the current bar prints a new high that ratchets the trail
    tighter, an intraday low below the newly ratcheted level must still
    exit on the SAME bar. The narrow fix defers only newly-armed trails —
    ratcheting semantics for already-active trails are preserved."""
    signal = _make_signal(entry_price=100.0, stop_loss=90.0, target_1=130.0, holding_period_days=5)
    outcome = _make_outcome()
    bars = [
        # Day 1: rally to 102 arms trail (gain ~1.9% on fill 100.1). Newly
        # armed — will NOT enforce on day 1 (narrow fix).
        {"open": 100.0, "high": 102.0, "low": 100.0, "close": 101.5},
        # Day 2: new high 106 ratchets trail to 106*0.997=105.682.
        # bar_open 105.7 is above 105.682 (no gap-through).
        # bar_low 105.0 is below 105.682 → intraday trail stop fires SAME bar.
        {"open": 105.7, "high": 106.0, "low": 105.0, "close": 105.5},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    settings = _default_settings(trail_activate_pct=0.5, trail_distance_pct=0.3)
    aggregator = _patch_deps(monkeypatch, signal, df, settings)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    assert update["exit_reason"] == "trail_stop"
    # Must exit at the newly ratcheted level (~105.68 * 0.999 ≈ 105.58),
    # NOT carry forward to day 3+ with stale day-1 trail (~101.69).
    assert 105.4 < update["exit_price"] < 106.0, (
        f"Expected exit via same-bar ratcheted trail ~105.58; got {update['exit_price']}"
    )


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


@pytest.mark.asyncio
async def test_fill_stamping_is_idempotent(monkeypatch):
    """Re-evaluation should NOT re-derive entry_price if already stamped."""
    signal = _make_signal(entry_price=100.0, target_1=110.0, stop_loss=90.0)
    # Simulate already-filled outcome: daily_prices is set (from prior evaluation)
    outcome = _make_outcome(
        entry_price=100.5,  # previously stamped at T+1 open
        daily_prices={"2026-03-10": {"open": 100.5, "high": 101.0, "low": 100.0, "close": 100.5}},
    )
    bars = _flat_bars(2, price=101.0)
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    settings = _default_settings(slippage_pct=0.05)  # extreme slippage to detect re-stamping
    aggregator = _patch_deps(monkeypatch, signal, df, settings)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    # entry_price should NOT appear in update (already stamped)
    assert "entry_price" not in update


@pytest.mark.asyncio
async def test_fill_stamping_when_open_equals_planned(monkeypatch):
    """Fill should still stamp correctly even if T+1 open == signal close."""
    signal = _make_signal(entry_price=100.0, target_1=110.0, stop_loss=90.0)
    # Fresh outcome: daily_prices is None (never evaluated)
    outcome = _make_outcome(entry_price=100.0)
    # T+1 open is exactly the signal close — would break price-comparison heuristic
    bars = [{"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5}]
    bars += _flat_bars(1, price=100.5)
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    settings = _default_settings(slippage_pct=0.0)  # zero slippage — fill == planned
    aggregator = _patch_deps(monkeypatch, signal, df, settings)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    # Should still stamp entry_price (first evaluation, daily_prices was None)
    assert "entry_price" in update
    # With zero slippage and open=100.0, fill should be 100.0
    assert update["entry_price"] == 100.0


@pytest.mark.asyncio
async def test_leg1_fill_does_not_raise_breakeven_same_bar(monkeypatch):
    """Regression: when leg1 partial TP fills on this bar, the newly-created
    breakeven floor at entry_price must NOT participate in same-bar stop
    checks. Daily OHLC cannot prove the partial_target high came before the
    intraday low, so a phantom breakeven stop on the same bar would be
    path-dependent. The floor starts enforcing from the NEXT bar."""
    # atr reverse-engineered from stop distance: (100 - 95)/0.75 = 6.667
    # entry_fill = 100 * 1.001 = 100.1
    # partial_target = 100.1 + 1.0 * 6.667 = 106.767
    signal = _make_signal(entry_price=100.0, stop_loss=95.0, target_1=115.0, holding_period_days=5)
    outcome = _make_outcome()
    bars = [
        # Day 1: bar_high 107 fills leg1; bar_low 99.8 is below entry_fill
        # (100.1) but above base_stop (95). Under the fix this must NOT
        # produce a same-bar breakeven exit.
        {"open": 100.0, "high": 107.0, "low": 99.8, "close": 100.5},
        # Day 2: flat around breakeven; position keeps running (or expires).
        {"open": 100.5, "high": 100.7, "low": 100.3, "close": 100.5},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    settings = _default_settings(
        partial_tp_enabled=True,
        partial_tp_atr_multiple=1.0,
        partial_tp_fraction=0.5,
        breakeven_after_partial=True,
    )
    aggregator = _patch_deps(monkeypatch, signal, df, settings)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    # leg1 partial TP should have filled on day 1
    assert update.get("partial_exit_price") is not None
    assert update.get("partial_exit_date") == outcome.entry_date
    # Must NOT have exited on the same bar as the leg1 fill via a phantom
    # breakeven stop. Either the position is still open or it exited later.
    if update.get("exit_date") is not None:
        assert update["exit_date"] != outcome.entry_date, (
            f"Phantom same-bar breakeven exit: reason={update.get('exit_reason')}, "
            f"price={update.get('exit_price')}"
        )


@pytest.mark.asyncio
async def test_leg1_filled_prior_bar_enforces_breakeven_next_bar(monkeypatch):
    """On the bar AFTER leg1 fills, the breakeven floor must enforce normally.
    Defer applies only to the fill bar itself — subsequent bars behave as
    before."""
    signal = _make_signal(entry_price=100.0, stop_loss=95.0, target_1=115.0, holding_period_days=5)
    outcome = _make_outcome()
    bars = [
        # Day 1: bar_high 107 fills leg1 cleanly; low 100.5 stays above entry.
        # No same-bar ambiguity, fine under both old and new code.
        {"open": 100.0, "high": 107.0, "low": 100.5, "close": 106.5},
        # Day 2: low 99.5 dips below entry_fill (100.1). Breakeven floor from
        # day 1's leg1 fill must enforce now and stop out the position.
        {"open": 101.0, "high": 101.5, "low": 99.5, "close": 100.0},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    settings = _default_settings(
        partial_tp_enabled=True,
        partial_tp_atr_multiple=1.0,
        partial_tp_fraction=0.5,
        breakeven_after_partial=True,
    )
    aggregator = _patch_deps(monkeypatch, signal, df, settings)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    assert update.get("partial_exit_price") is not None
    # Day 2 must exit via the breakeven floor (labeled "stop" — not a trail stop).
    assert update.get("exit_reason") == "stop"
    # Exit price should be near entry_fill * (1 - slippage) ≈ 99.999
    assert 99.5 < update["exit_price"] < 100.5, (
        f"Expected breakeven-level exit near 100.0; got {update['exit_price']}"
    )


@pytest.mark.asyncio
async def test_leg1_pre_filled_from_persistence_enforces_breakeven(monkeypatch):
    """If leg1 was filled in a PRIOR evaluation pass (outcome.partial_exit_price
    is already set when _evaluate_position is called), the breakeven floor
    must enforce from the very first loop iteration of this call. This covers
    the persisted/resumable state path."""
    signal = _make_signal(entry_price=100.0, stop_loss=95.0, target_1=115.0, holding_period_days=5)
    # Outcome already has leg1 filled AND daily_prices set — mimics a re-evaluation
    # after a prior afternoon check stamped leg1.
    outcome = _make_outcome(
        entry_price=100.1,
        daily_prices={
            "2026-03-10": {"open": 100.0, "high": 107.0, "low": 100.5, "close": 106.5}
        },
    )
    outcome.partial_exit_price = 106.77
    outcome.partial_exit_date = date(2026, 3, 10)
    bars = [
        # Day 1 (already seen): reproduces the prior fill day
        {"open": 100.0, "high": 107.0, "low": 100.5, "close": 106.5},
        # Day 2: low dips below entry_fill; breakeven floor must enforce
        {"open": 101.0, "high": 101.5, "low": 99.4, "close": 99.8},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    settings = _default_settings(
        partial_tp_enabled=True,
        partial_tp_atr_multiple=1.0,
        partial_tp_fraction=0.5,
        breakeven_after_partial=True,
    )
    aggregator = _patch_deps(monkeypatch, signal, df, settings)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    # leg1 was pre-filled from persistence — position should now exit at breakeven
    assert update.get("exit_reason") == "stop"
    assert 99.5 < update["exit_price"] < 100.5, (
        f"Expected breakeven exit near 100.0; got {update['exit_price']}"
    )


@pytest.mark.asyncio
async def test_same_pass_leg1_leg2_exit(monkeypatch):
    """If leg1 fills and leg2 exits in the same evaluation, weighted PnL should be correct."""
    signal = _make_signal(entry_price=100.0, stop_loss=95.0, target_1=110.0)
    outcome = _make_outcome()
    # ATR reverse-engineered from stop: (100 - 95) / 0.75 = 6.67
    # partial_target = entry + 1.0 * 6.67 ≈ 106.67
    bars = [
        # Entry bar: normal
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.2},
        # Day 2: rallies to 108 (fills leg1 at ~106.67), then closes at 107
        {"open": 101.0, "high": 108.0, "low": 100.5, "close": 107.0},
        # Day 3: crashes through stop at 95
        {"open": 96.0, "high": 97.0, "low": 94.0, "close": 94.5},
    ]
    df = _make_ohlcv_bars(bars, start_date=outcome.entry_date)

    settings = _default_settings(
        partial_tp_enabled=True,
        partial_tp_atr_multiple=1.0,
        partial_tp_fraction=0.5,
        breakeven_after_partial=True,
    )
    aggregator = _patch_deps(monkeypatch, signal, df, settings)

    update, _ = await perf._evaluate_position(outcome, aggregator)

    assert update is not None
    assert update["still_open"] is False
    # Leg1 should have filled (partial_exit_price set)
    assert "partial_exit_price" in update
    # Weighted PnL should blend leg1 and leg2, not just leg2
    assert update.get("leg2_exit_reason") is not None
