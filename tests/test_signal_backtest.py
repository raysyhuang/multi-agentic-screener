"""Tests for the standalone signal model backtester."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.research.signal_backtest import (
    ModelResult,
    classify_regime,
    format_model_report,
    run_model_backtest,
    scan_breakout,
    scan_mean_reversion,
    simulate_trade,
)


def _make_ohlcv(days: int = 120, start_price: float = 100.0, trend: float = 0.001) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a mild uptrend."""
    np.random.seed(42)
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(days)]
    closes = [start_price]
    for _ in range(days - 1):
        ret = trend + np.random.normal(0, 0.015)
        closes.append(closes[-1] * (1 + ret))

    closes = np.array(closes)
    highs = closes * (1 + np.random.uniform(0.002, 0.02, days))
    lows = closes * (1 - np.random.uniform(0.002, 0.02, days))
    opens = closes * (1 + np.random.uniform(-0.01, 0.01, days))
    volumes = np.random.randint(500_000, 5_000_000, days)

    return pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


class TestSimulateTrade:
    def test_target_hit(self):
        df = _make_ohlcv()
        # Place target very close so it's guaranteed to hit
        sig_date = df["date"].iloc[50]
        close = float(df["close"].iloc[50])
        result = simulate_trade(df, sig_date, stop_loss=close * 0.8, target=close * 1.001, max_hold=5)
        assert result is not None
        assert result["exit_reason"] == "target"
        assert result["pnl_pct"] > 0

    def test_stop_hit(self):
        df = _make_ohlcv()
        sig_date = df["date"].iloc[50]
        close = float(df["close"].iloc[50])
        # Stop very close below entry — likely to be hit on any dip
        result = simulate_trade(df, sig_date, stop_loss=close * 0.999, target=close * 2.0, max_hold=5)
        assert result is not None
        assert result["exit_reason"] == "stop"

    def test_expiry(self):
        df = _make_ohlcv()
        sig_date = df["date"].iloc[50]
        close = float(df["close"].iloc[50])
        # Very wide stop and target — will expire
        result = simulate_trade(df, sig_date, stop_loss=close * 0.5, target=close * 2.0, max_hold=3)
        assert result is not None
        assert result["exit_reason"] == "expiry"

    def test_no_future_data(self):
        df = _make_ohlcv(days=5)
        sig_date = df["date"].iloc[-1]
        result = simulate_trade(df, sig_date, stop_loss=50, target=200, max_hold=5)
        assert result is None

    def test_mfe_mae_tracked(self):
        df = _make_ohlcv()
        sig_date = df["date"].iloc[50]
        close = float(df["close"].iloc[50])
        result = simulate_trade(df, sig_date, stop_loss=close * 0.5, target=close * 2.0, max_hold=10)
        assert result is not None
        assert "mfe_pct" in result
        assert "mae_pct" in result

    def test_trailing_stop_activates(self):
        """Trailing stop should lock in gains when activated."""
        df = _make_ohlcv(days=120, trend=0.005)  # uptrend
        sig_date = df["date"].iloc[50]
        close = float(df["close"].iloc[50])
        # Wide fixed stop, impossible target, but trailing should activate
        result_no_trail = simulate_trade(
            df, sig_date, stop_loss=close * 0.8, target=close * 2.0, max_hold=10,
        )
        result_with_trail = simulate_trade(
            df, sig_date, stop_loss=close * 0.8, target=close * 2.0, max_hold=10,
            trail_activate_pct=0.5, trail_distance_pct=0.5,
        )
        assert result_no_trail is not None
        assert result_with_trail is not None
        # With trailing, should either exit via trail_stop or still expire
        assert result_with_trail["exit_reason"] in ("trail_stop", "expiry", "target")

    def test_leg1_fill_does_not_raise_breakeven_same_bar(self):
        """Regression for the leg1 same-bar breakeven phantom: when leg1
        partial TP fills on a bar, the newly-created breakeven floor at
        entry_price must NOT participate in that same bar's stop check.
        Daily OHLC cannot prove the partial_target high came before the
        intraday low, so a same-bar breakeven exit would be path-dependent.

        Mirrors the equivalent regression in test_evaluate_position.py for
        the production _evaluate_position path. Kept here to lock down the
        research simulator directly against future drift.
        """
        # entry_delay_days=0 so signal_date IS the entry bar. The loop then
        # starts walking from day 1 (T+1). slippage=0 for clean arithmetic.
        #
        # entry_price = 100 (day 0 open, no slippage).
        # partial_target = 100 + 1.0 * 5.0 = 105.
        # base stop = 95.
        df = pd.DataFrame({
            "date": [
                date(2024, 1, 2),  # day 0 — entry bar (not in exit loop)
                date(2024, 1, 3),  # day 1 — fills leg1 AND dips below entry
                date(2024, 1, 4),  # day 2 — flat above entry
                date(2024, 1, 5),  # day 3 — flat
                date(2024, 1, 6),  # day 4 — flat
                date(2024, 1, 7),  # day 5 — expiry close
            ],
            "open":  [100.0, 100.2, 100.5, 100.4, 100.3, 100.4],
            # Day 1 high 106 fills leg1 (>= 105); day 1 low 99.5 is below entry
            # (100) but above base stop (95). Under the fix this must NOT
            # produce a same-bar breakeven exit — the trade should survive
            # day 1 and exit later (expiry, in this scenario).
            "high":  [100.5, 106.0, 100.8, 100.7, 100.6, 100.5],
            "low":   [ 99.5,  99.5, 100.2, 100.2, 100.1, 100.2],
            "close": [100.3, 100.3, 100.5, 100.4, 100.3, 100.4],
            "volume": [1_000_000] * 6,
        })

        result = simulate_trade(
            df,
            signal_date=date(2024, 1, 2),
            stop_loss=95.0,
            target=150.0,
            max_hold=5,
            slippage=0.0,
            entry_delay_days=0,
            partial_tp_atr_mult=1.0,
            atr_value=5.0,
        )

        assert result is not None
        # Leg1 should have filled on day 1
        assert result.get("leg1_pnl") is not None
        assert abs(result["leg1_pnl"] - 5.0) < 0.01  # (105 - 100)/100 * 100 = 5%
        # Must NOT have same-bar exit on day 1 via phantom breakeven stop.
        # Under the fix, the position survives day 1 and reaches expiry.
        assert result["exit_reason"] == "expiry", (
            f"Expected expiry after leg1 fill; got {result['exit_reason']} "
            f"on {result['exit_date']}"
        )
        assert result["exit_date"] == date(2024, 1, 7)

    def test_leg1_filled_prior_bar_enforces_breakeven_next_bar(self):
        """After leg1 fills on a prior bar, the breakeven floor must enforce
        normally on subsequent bars — the deferral rule applies only to the
        bar leg1 fills on."""
        df = pd.DataFrame({
            "date": [
                date(2024, 1, 2),  # day 0 — entry
                date(2024, 1, 3),  # day 1 — fills leg1 cleanly (low stays above entry)
                date(2024, 1, 4),  # day 2 — low dips below entry; breakeven enforces
                date(2024, 1, 5),
                date(2024, 1, 6),
            ],
            "open":  [100.0, 100.5, 101.0, 100.0, 100.0],
            "high":  [100.5, 106.0, 101.5, 100.5, 100.5],
            "low":   [ 99.5, 100.3,  99.5, 100.0, 100.0],
            "close": [100.3, 105.5, 100.0, 100.2, 100.2],
            "volume": [1_000_000] * 5,
        })

        result = simulate_trade(
            df,
            signal_date=date(2024, 1, 2),
            stop_loss=95.0,
            target=150.0,
            max_hold=5,
            slippage=0.0,
            entry_delay_days=0,
            partial_tp_atr_mult=1.0,
            atr_value=5.0,
        )

        assert result is not None
        assert result.get("leg1_pnl") is not None
        # Day 2 must exit via breakeven — labeled "stop" (trail not armed).
        assert result["exit_reason"] == "stop"
        assert result["exit_date"] == date(2024, 1, 4)
        # Exit at ~entry_price (100.0 - slippage=0), leg2_pnl ≈ 0.
        # Weighted pnl = 0.5 * 5.0 + 0.5 * 0 = 2.5%.
        assert abs(result["pnl_pct"] - 2.5) < 0.1

    def test_trailing_stop_disabled_when_zero(self):
        """Setting trail params to 0 should behave identically to no trailing."""
        df = _make_ohlcv()
        sig_date = df["date"].iloc[50]
        close = float(df["close"].iloc[50])
        r1 = simulate_trade(df, sig_date, stop_loss=close * 0.9, target=close * 1.5, max_hold=5)
        r2 = simulate_trade(df, sig_date, stop_loss=close * 0.9, target=close * 1.5, max_hold=5,
                            trail_activate_pct=0.0, trail_distance_pct=0.0)
        assert r1 == r2


class TestClassifyRegime:
    def test_bull(self):
        df = _make_ohlcv(days=100, trend=0.005)  # strong uptrend
        assert classify_regime(df) == "bull"

    def test_bear(self):
        df = _make_ohlcv(days=100, trend=-0.005)  # strong downtrend
        assert classify_regime(df) == "bear"

    def test_short_data(self):
        df = _make_ohlcv(days=10)
        assert classify_regime(df) == "unknown"


class TestScanBreakout:
    def test_scans_without_error(self):
        df = _make_ohlcv(days=120, trend=0.003)
        signals = scan_breakout("TEST", df)
        # May or may not find signals depending on synthetic data
        assert isinstance(signals, list)

    def test_respects_min_score(self):
        df = _make_ohlcv(days=120)
        high_bar = scan_breakout("TEST", df, min_score=90)
        low_bar = scan_breakout("TEST", df, min_score=30)
        assert len(high_bar) <= len(low_bar)

    def test_short_data_returns_empty(self):
        df = _make_ohlcv(days=20)
        assert scan_breakout("TEST", df) == []


class TestScanMeanReversion:
    def test_scans_without_error(self):
        df = _make_ohlcv(days=120)
        signals = scan_mean_reversion("TEST", df)
        assert isinstance(signals, list)

    def test_rsi_threshold(self):
        df = _make_ohlcv(days=120)
        strict = scan_mean_reversion("TEST", df, rsi2_threshold=5)
        relaxed = scan_mean_reversion("TEST", df, rsi2_threshold=25)
        assert len(strict) <= len(relaxed)


class TestRunModelBacktest:
    def test_breakout_runs(self):
        data = {"TEST": _make_ohlcv(days=120, trend=0.003)}
        result = run_model_backtest("breakout", data)
        assert isinstance(result, ModelResult)
        assert result.model == "breakout"
        assert result.metrics.total_trades >= 0

    def test_mean_reversion_runs(self):
        data = {"TEST": _make_ohlcv(days=120)}
        result = run_model_backtest("mean_reversion", data)
        assert isinstance(result, ModelResult)
        assert result.model == "mean_reversion"

    def test_unknown_model_raises(self):
        data = {"TEST": _make_ohlcv()}
        with pytest.raises(ValueError, match="Unknown model"):
            run_model_backtest("invalid_model", data)

    def test_empty_data(self):
        result = run_model_backtest("breakout", {})
        assert result.metrics.total_trades == 0

    def test_short_data_no_crash(self):
        data = {"TEST": _make_ohlcv(days=20)}
        result = run_model_backtest("breakout", data)
        assert result.metrics.total_trades == 0


class TestFormatReport:
    def test_formats_without_error(self):
        from src.backtest.metrics import _empty_metrics
        result = ModelResult(
            model="breakout", params={}, metrics=_empty_metrics(),
            trades=[], by_regime={}, by_exit_reason={},
            avg_mfe_pct=0, avg_mae_pct=0, dsr=0,
        )
        text = format_model_report(result)
        assert "BREAKOUT" in text
        assert "Win Rate" in text
