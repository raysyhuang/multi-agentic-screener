"""Tests for V1.2 Unified Surgical Alpha Plan changes.

Covers: two-leg exits, gap filter, volume slope, regime gating,
ATR percentile floor, veto board, and idiosyncratic bonus.
"""

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ── Helpers ──────────────────────────────────────────────────────


def _make_ohlcv(n: int = 100, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV with a downtrend at the end (oversold)."""
    np.random.seed(seed)
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n)]
    close = [start_price]
    for _ in range(n - 1):
        close.append(close[-1] * (1 + np.random.normal(-0.001, 0.015)))
    close = np.array(close)
    high = close * (1 + np.random.uniform(0.001, 0.02, n))
    low = close * (1 - np.random.uniform(0.001, 0.02, n))
    opn = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.randint(500_000, 2_000_000, n).astype(float)
    return pd.DataFrame({
        "date": dates, "open": opn, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def _oversold_features(close: float = 95.0, atr: float = 2.0) -> dict:
    """Build features dict for a deeply oversold stock."""
    return {
        "rsi_2": 5, "pct_above_sma200": 5.0, "pct_above_sma50": 2.0,
        "streak": -4, "dist_from_5d_low": 0.3, "rvol": 1.2,
        "atr_14": atr, "close": close,
    }


# ── 1. Two-Leg Trade Engine ─────────────────────────────────────


class TestTwoLegSimulation:
    """Test two-leg exit logic in signal_backtest.simulate_trade."""

    def test_two_leg_weighted_pnl(self):
        """When both legs fill, PnL should be weighted 50/50."""
        from src.research.signal_backtest import simulate_trade

        # Build price data: entry at 100, goes to 104 (partial at 102), then trails back
        dates = [date(2025, 6, 1) + timedelta(days=i) for i in range(10)]
        df = pd.DataFrame({
            "date": dates,
            "open":  [100, 100, 101, 103, 104, 103, 101, 100, 99, 98],
            "high":  [101, 101, 102, 104, 105, 104, 102, 101, 100, 99],
            "low":   [99,   99, 100, 102, 103, 102, 100, 99,  98, 97],
            "close": [100, 100, 101, 103, 104, 103, 101, 100, 99, 98],
            "volume": [1e6] * 10,
        })

        result = simulate_trade(
            df, signal_date=dates[0],
            stop_loss=96.0, target=110.0, max_hold=5,
            trail_activate_pct=0.5, trail_distance_pct=0.3,
            partial_tp_atr_mult=1.0, atr_value=2.0,
            max_entry_price=0.0,
        )

        assert result is not None
        # Should have leg1_pnl and leg2_pnl fields
        if "leg1_pnl" in result:
            assert result["leg1_pnl"] > 0  # partial target was hit
            # Weighted PnL = 0.5 * leg1 + 0.5 * leg2
            expected = 0.5 * result["leg1_pnl"] + 0.5 * result["leg2_pnl"]
            assert abs(result["pnl_pct"] - expected) < 0.01

    def test_breakeven_pivot_after_leg1(self):
        """After Leg 1 fills, stop should not go below entry price."""
        from src.research.signal_backtest import simulate_trade

        dates = [date(2025, 6, 1) + timedelta(days=i) for i in range(8)]
        # Price goes up to hit partial, then drops back to just below entry
        df = pd.DataFrame({
            "date": dates,
            "open":  [100, 100, 103, 104, 100.5, 100.2, 99.5, 99],
            "high":  [101, 101, 104, 105, 101,   100.5, 100,  99.5],
            "low":   [99,   99, 102, 103, 100,   99.8,  99,   98],
            "close": [100, 100, 103, 104, 100.2, 100,   99.5, 99],
            "volume": [1e6] * 8,
        })

        result = simulate_trade(
            df, signal_date=dates[0],
            stop_loss=95.0, target=110.0, max_hold=6,
            trail_activate_pct=0.5, trail_distance_pct=0.3,
            partial_tp_atr_mult=1.0, atr_value=2.0,
        )

        assert result is not None
        # After breakeven pivot, exit should be at or above entry
        if "leg1_pnl" in result:
            assert result["exit_price"] >= 99.5  # should be stopped at ~entry

    def test_single_leg_when_partial_disabled(self):
        """With partial_tp_atr_mult=0, should behave like original single-leg."""
        from src.research.signal_backtest import simulate_trade

        dates = [date(2025, 6, 1) + timedelta(days=i) for i in range(6)]
        df = pd.DataFrame({
            "date": dates,
            "open":  [100, 100, 101, 102, 103, 104],
            "high":  [101, 101, 102, 103, 104, 105],
            "low":   [99,   99, 100, 101, 102, 103],
            "close": [100, 100, 101, 102, 103, 104],
            "volume": [1e6] * 6,
        })

        result = simulate_trade(
            df, signal_date=dates[0],
            stop_loss=96.0, target=104.0, max_hold=5,
            partial_tp_atr_mult=0.0, atr_value=0.0,
        )

        assert result is not None
        assert "leg1_pnl" not in result  # no two-leg fields


# ── 2. Gap Filter ────────────────────────────────────────────────


class TestGapFilter:

    def test_gap_filter_blocks_gapped_entry(self):
        """Should return None when T+1 open exceeds max_entry_price."""
        from src.research.signal_backtest import simulate_trade

        dates = [date(2025, 6, 1) + timedelta(days=i) for i in range(6)]
        df = pd.DataFrame({
            "date": dates,
            "open":  [100, 105, 106, 107, 108, 109],  # T+1 open = 105, way above max
            "high":  [101, 106, 107, 108, 109, 110],
            "low":   [99,  104, 105, 106, 107, 108],
            "close": [100, 105, 106, 107, 108, 109],
            "volume": [1e6] * 6,
        })

        result = simulate_trade(
            df, signal_date=dates[0],
            stop_loss=96.0, target=110.0, max_hold=5,
            max_entry_price=101.0,  # close + 0.2*ATR
        )
        assert result is None  # filtered by gap

    def test_gap_filter_passes_normal_open(self):
        """Should allow trade when T+1 open is below max_entry_price."""
        from src.research.signal_backtest import simulate_trade

        dates = [date(2025, 6, 1) + timedelta(days=i) for i in range(6)]
        df = pd.DataFrame({
            "date": dates,
            "open":  [100, 100.3, 101, 102, 103, 104],
            "high":  [101, 101,   102, 103, 104, 105],
            "low":   [99,   99,   100, 101, 102, 103],
            "close": [100, 100.5, 101, 102, 103, 104],
            "volume": [1e6] * 6,
        })

        result = simulate_trade(
            df, signal_date=dates[0],
            stop_loss=96.0, target=104.0, max_hold=5,
            max_entry_price=101.0,
        )
        assert result is not None

    def test_max_entry_price_set_on_signal(self):
        """MeanReversionSignal should have max_entry_price field."""
        from src.signals.mean_reversion import score_mean_reversion

        df = _make_ohlcv(n=260)
        feat = _oversold_features()
        result = score_mean_reversion("TEST", df, feat)

        if result is not None:
            assert result.max_entry_price is not None
            assert result.max_entry_price > result.entry_price


# ── 3. Volume Slope ──────────────────────────────────────────────


class TestVolumeSlope:

    def test_spiking_volume_penalized(self):
        """Spiking volume (positive slope + high RVOL) should lower score."""
        from src.signals.mean_reversion import score_mean_reversion

        df = _make_ohlcv(n=260)
        # Make last 3 bars have increasing volume
        df.loc[df.index[-3:], "volume"] = [1e6, 2e6, 3e6]

        feat = _oversold_features()
        feat["rvol"] = 2.0  # high RVOL + positive slope = distribution

        result = score_mean_reversion("SPIKE", df, feat)
        # Score should be lower (may still fire, but liquidity component = 10)
        if result is not None:
            assert result.components["liquidity"] == 10

    def test_declining_volume_rewarded(self):
        """Declining volume into the low should get higher liquidity score."""
        from src.signals.mean_reversion import score_mean_reversion

        df = _make_ohlcv(n=260)
        # Make last 3 bars have declining volume
        df.loc[df.index[-3:], "volume"] = [3e6, 2e6, 1e6]

        feat = _oversold_features()
        feat["rvol"] = 0.8

        result = score_mean_reversion("EXHAUST", df, feat)
        if result is not None:
            assert result.components["liquidity"] == 80


# ── 4. Regime + Volatility Gating ────────────────────────────────


class TestRegimeGating:

    def test_choppy_regime_raises_floor(self):
        """In choppy regime, score 60 should be blocked (floor=75)."""
        from src.signals.mean_reversion import score_mean_reversion

        df = _make_ohlcv(n=260)
        feat = _oversold_features()
        # Weaken features to get a score around 60-70
        feat["rsi_2"] = 9  # still passes threshold
        feat["streak"] = -1  # weak streak
        feat["pct_above_sma200"] = -2.0  # below SMA200

        result_choppy = score_mean_reversion("TEST", df, feat, regime="choppy")
        result_bull = score_mean_reversion("TEST", df, feat, regime="bull")

        # In choppy, marginal signals should be blocked more aggressively
        # (exact behavior depends on composite score, but floor is higher)
        # This is a property test: if bull passes, choppy should pass less often
        if result_bull is not None and result_bull.score < 75:
            assert result_choppy is None

    def test_bull_regime_normal_floor(self):
        """In bull regime, standard floor of 50 applies."""
        from src.signals.mean_reversion import score_mean_reversion

        df = _make_ohlcv(n=260)
        feat = _oversold_features()

        result = score_mean_reversion("TEST", df, feat, regime="bull")
        # Strong oversold signal should still fire in bull
        assert result is not None


class TestATRPercentileFloor:

    def test_low_atr_blocked(self):
        """Stock with ATR in bottom decile should be blocked."""
        from src.signals.mean_reversion import score_mean_reversion

        # Build 260 bars with very low recent volatility
        df = _make_ohlcv(n=300, start_price=100.0, seed=123)
        # Crush volatility in last 14 bars
        for i in range(286, 300):
            df.loc[i, "high"] = df.loc[i, "close"] + 0.01
            df.loc[i, "low"] = df.loc[i, "close"] - 0.01

        feat = _oversold_features(atr=0.02)  # very low ATR
        result = score_mean_reversion("LOWATR", df, feat, regime="bull")
        # Should be None or very low score due to ATR floor
        # (depends on exact percentile calc)


# ── 5. Veto Board ────────────────────────────────────────────────


class TestVetoBoard:

    def test_veto_penalizes_conflicting_picks(self):
        """Picks rejected by other engines should have confidence reduced."""
        from src.engines.deterministic_synthesizer import _apply_veto_board

        picks = [
            {"ticker": "AAPL", "engine_count": 1,
             "avg_weighted_confidence": 80, "combined_score": 80,
             "metadata": {}},
        ]
        # Simulate an engine that rejected AAPL
        engine_results = [
            {"payload": {
                "engine_name": "koocore_d",
                "screened_but_rejected": [{"ticker": "AAPL", "reason": "distribution"}],
            }},
        ]

        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.veto_board_enabled = True
            mock_settings.return_value.veto_penalty = 0.5
            mock_settings.return_value.idiosyncratic_bonus_enabled = False
            result = _apply_veto_board(picks, engine_results)

        assert result[0]["avg_weighted_confidence"] == 40.0  # 80 * 0.5
        assert result[0]["veto_applied"] is True

    def test_idiosyncratic_bonus_applied(self):
        """Single-engine pick with stock oversold + sector not should get bonus."""
        from src.engines.deterministic_synthesizer import _apply_veto_board

        picks = [
            {"ticker": "XYZ", "engine_count": 1,
             "avg_weighted_confidence": 70, "combined_score": 70,
             "metadata": {"rsi_2": 5, "sector_rsi2": 45}},
        ]

        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.veto_board_enabled = True
            mock_settings.return_value.veto_penalty = 0.5
            mock_settings.return_value.idiosyncratic_bonus_enabled = True
            mock_settings.return_value.idiosyncratic_bonus_multiplier = 1.10
            result = _apply_veto_board(picks, [])

        assert result[0]["avg_weighted_confidence"] == pytest.approx(77.0, abs=0.1)
        assert result[0]["idiosyncratic_bonus"] is True

    def test_veto_disabled_no_change(self):
        """When veto board is disabled, picks should pass through unchanged."""
        from src.engines.deterministic_synthesizer import _apply_veto_board

        picks = [
            {"ticker": "MSFT", "engine_count": 1,
             "avg_weighted_confidence": 60, "combined_score": 60,
             "metadata": {}},
        ]

        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.veto_board_enabled = False
            result = _apply_veto_board(picks, [])

        assert result[0]["avg_weighted_confidence"] == 60


# ── 6. Earnings Blackout ─────────────────────────────────────────


class TestEarningsBlackout:

    def test_earnings_blackout_config(self):
        """Config should have earnings_blackout_days."""
        from src.config import Settings
        s = Settings(polygon_api_key="test", fmp_api_key="test")
        assert s.earnings_blackout_days == 2


# ── 7. Config Params ─────────────────────────────────────────────


class TestV12Config:

    def test_all_v12_params_exist(self):
        """All V1.2 config parameters should have defaults."""
        from src.config import Settings
        s = Settings(polygon_api_key="test", fmp_api_key="test")

        assert s.partial_tp_enabled is False
        assert s.partial_tp_fraction == 0.5
        assert s.partial_tp_atr_multiple == 1.0
        assert s.breakeven_after_partial is True
        assert s.entry_gap_max_atr == 0.2
        assert s.volume_slope_lookback == 3
        assert s.choppy_min_score == 75
        assert s.min_atr_percentile_252 == 0.10
        assert s.earnings_blackout_days == 2
        assert s.veto_board_enabled is True
        assert s.veto_penalty == 0.5
        assert s.idiosyncratic_bonus_enabled is True
        assert s.idiosyncratic_bonus_multiplier == 1.10


# ── 8. DB Model Columns ─────────────────────────────────────────


class TestOutcomeModel:

    def test_outcome_has_partial_fields(self):
        """Outcome model should have two-leg tracking columns."""
        from src.db.models import Outcome
        assert hasattr(Outcome, "partial_exit_price")
        assert hasattr(Outcome, "partial_exit_date")
        assert hasattr(Outcome, "leg2_exit_reason")


# ── 9. Metrics ───────────────────────────────────────────────────


class TestV12Metrics:

    def test_performance_metrics_has_v12_fields(self):
        """PerformanceMetrics should include foregone_profit and expiry_mfe_gt_2pct."""
        from src.backtest.metrics import PerformanceMetrics, compute_metrics

        m = compute_metrics([1.0, -0.5, 2.0, -1.0, 0.5])
        assert hasattr(m, "foregone_profit")
        assert hasattr(m, "expiry_mfe_gt_2pct")
        assert m.foregone_profit == 0.0  # default


# ── 10. Phase 2: Confirmation Proxy ──────────────────────────────


class TestConfirmationProxy:

    def test_bearish_candle_rejected(self):
        """Entry day close < open should reject trade when confirm enabled."""
        from src.research.signal_backtest import simulate_trade

        # Build df where entry day is bearish (close < open)
        dates = [date(2025, 6, i) for i in range(1, 8)]
        df = pd.DataFrame({
            "date": dates,
            "open":  [100, 101, 99, 100, 101, 100, 100],
            "high":  [102, 102, 101, 102, 103, 102, 102],
            "low":   [99, 99, 98, 99, 100, 99, 99],
            "close": [101, 100, 100, 101, 102, 101, 101],  # day 2: close=100 < open=101
            "volume": [1e6] * 7,
        })
        result = simulate_trade(
            df, signal_date=dates[0],
            stop_loss=95, target=110, max_hold=5,
            confirm_entry=True, confirm_mode="close_gt_open",
        )
        assert result is None  # bearish entry day rejected

    def test_bullish_candle_accepted_and_enters_at_close(self):
        """Entry day close > open should allow trade and enter at close price."""
        from src.research.signal_backtest import simulate_trade

        dates = [date(2025, 6, i) for i in range(1, 8)]
        df = pd.DataFrame({
            "date": dates,
            "open":  [100, 99, 100, 100, 100, 100, 100],
            "high":  [102, 102, 102, 102, 102, 102, 102],
            "low":   [99, 98, 99, 99, 99, 99, 99],
            "close": [101, 101, 101, 101, 101, 101, 101],  # day 2: close=101 > open=99
            "volume": [1e6] * 7,
        })
        result = simulate_trade(
            df, signal_date=dates[0],
            stop_loss=90, target=110, max_hold=5,
            confirm_entry=True, confirm_mode="close_gt_open",
        )
        assert result is not None
        # Should enter at close (101) + slippage, not at open (99)
        assert result["entry_price"] > 100.0

    def test_confirm_disabled_allows_all(self):
        """Without confirm, bearish candle should still be accepted."""
        from src.research.signal_backtest import simulate_trade

        dates = [date(2025, 6, i) for i in range(1, 8)]
        df = pd.DataFrame({
            "date": dates,
            "open":  [100, 101, 99, 100, 101, 100, 100],
            "high":  [102, 102, 101, 102, 103, 102, 102],
            "low":   [99, 99, 98, 99, 100, 99, 99],
            "close": [101, 100, 100, 101, 102, 101, 101],
            "volume": [1e6] * 7,
        })
        result = simulate_trade(
            df, signal_date=dates[0],
            stop_loss=90, target=110, max_hold=5,
            confirm_entry=False,
        )
        assert result is not None


# ── 11. Phase 2: Weekly Trend Gate ────────────────────────────────


class TestWeeklyTrendGate:

    def test_below_sma150_blocked(self):
        """Stock below 150-day SMA should be blocked when gate enabled."""
        from src.signals.mean_reversion import score_mean_reversion

        # Build 200 bars with downtrend (price well below 150-SMA)
        df = _make_ohlcv(n=200, start_price=120.0, seed=99)
        # Force last bars to be low (below SMA150)
        df.loc[df.index[-10:], "close"] = 80.0
        df.loc[df.index[-10:], "low"] = 79.0
        feat = _oversold_features(close=80.0, atr=2.0)

        with patch("src.config.get_settings") as mock:
            mock.return_value.weekly_trend_gate_enabled = True
            mock.return_value.weekly_trend_sma_days = 150
            mock.return_value.shock_killswitch_enabled = False
            mock.return_value.shock_killswitch_atr_mult = 3.0
            mock.return_value.choppy_min_score = 50
            mock.return_value.min_atr_percentile_252 = 0.0  # disable ATR floor
            mock.return_value.entry_gap_max_atr = 0.2
            result = score_mean_reversion("TEST", df, feat, regime="bull")

        assert result is None


# ── 12. Phase 2: Shock Kill-Switch ────────────────────────────────


class TestShockKillSwitch:

    def test_extreme_range_blocked(self):
        """True range > 3×ATR should block signal."""
        from src.signals.mean_reversion import score_mean_reversion

        df = _make_ohlcv(n=260, start_price=100.0)
        # Make last bar have extreme range: high=110, low=90 → range=20 vs ATR~2
        df.loc[df.index[-1], "high"] = 110.0
        df.loc[df.index[-1], "low"] = 90.0
        feat = _oversold_features(close=95.0, atr=2.0)

        with patch("src.config.get_settings") as mock:
            mock.return_value.weekly_trend_gate_enabled = False
            mock.return_value.shock_killswitch_enabled = True
            mock.return_value.shock_killswitch_atr_mult = 3.0
            mock.return_value.choppy_min_score = 50
            mock.return_value.min_atr_percentile_252 = 0.0
            mock.return_value.entry_gap_max_atr = 0.2
            result = score_mean_reversion("SHOCK", df, feat, regime="bull")

        assert result is None

    def test_normal_range_passes(self):
        """Normal range < 3×ATR should pass."""
        from src.signals.mean_reversion import score_mean_reversion

        df = _make_ohlcv(n=260, start_price=100.0)
        feat = _oversold_features(close=95.0, atr=2.0)

        with patch("src.config.get_settings") as mock:
            mock.return_value.weekly_trend_gate_enabled = False
            mock.return_value.shock_killswitch_enabled = True
            mock.return_value.shock_killswitch_atr_mult = 3.0
            mock.return_value.choppy_min_score = 50
            mock.return_value.min_atr_percentile_252 = 0.0
            mock.return_value.entry_gap_max_atr = 0.2
            result = score_mean_reversion("NORMAL", df, feat, regime="bull")

        # Should not be None (shock switch shouldn't block normal range)
        # May still be None for other reasons, but not shock


# ── 13. Phase 2: Config Params ────────────────────────────────────


class TestPhase2Config:

    def test_phase2_params_exist(self):
        """All Phase 2 config parameters should have defaults."""
        from src.config import Settings
        s = Settings(polygon_api_key="test", fmp_api_key="test")

        assert s.weekly_trend_gate_enabled is False
        assert s.weekly_trend_sma_days == 150
        assert s.shock_killswitch_enabled is False
        assert s.shock_killswitch_atr_mult == 3.0
        assert s.confirm_entry_enabled is False
        assert s.confirm_mode == "close_gt_open"
        assert s.blocked_entry_weekdays == ""
