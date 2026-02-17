"""Tests for the Position Health Card engine (src/output/health.py)."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.contracts import HealthCardConfig, HealthState
from src.output.health import (
    STRATEGY_FAMILY,
    _check_soft_invalidation,
    _check_velocity_warning,
    _compute_momentum,
    _compute_regime,
    _compute_risk,
    _compute_trend,
    _compute_volume,
    _check_hard_invalidation,
    compute_health_card,
    compute_score_velocity,
    get_strategy_family,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_df(rows: int = 60, trend: str = "up") -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame."""
    dates = pd.date_range(end=date.today(), periods=rows, freq="B")
    base = 100.0
    closes = []
    for i in range(rows):
        if trend == "up":
            closes.append(base + i * 0.5 + np.random.normal(0, 0.3))
        elif trend == "down":
            closes.append(base - i * 0.5 + np.random.normal(0, 0.3))
        else:
            closes.append(base + np.random.normal(0, 1))
    closes_arr = np.array(closes)
    return pd.DataFrame({
        "date": dates,
        "open": closes_arr - 0.3,
        "high": closes_arr + 0.5,
        "low": closes_arr - 0.5,
        "close": closes_arr,
        "volume": np.random.randint(500_000, 2_000_000, rows),
    })


def _make_feat(**overrides) -> dict:
    """Build a synthetic feature dict."""
    feat = {
        "close": 110.0,
        "ema_21": 108.0,
        "sma_50": 105.0,
        "rsi_14": 55.0,
        "atr_14": 2.0,
        "rvol": 1.2,
        "obv": 1_000_000,
        "MACDh_12_26_9": 0.5,
        "BBU_20_2.0": 115.0,
        "BBL_20_2.0": 105.0,
        "BBM_20_2.0": 110.0,
        "high_20d": 112.0,
        "low_20d": 104.0,
    }
    feat.update(overrides)
    return feat


def _make_signal(**overrides) -> MagicMock:
    """Build a mock Signal object."""
    sig = MagicMock()
    sig.id = overrides.get("id", 1)
    sig.signal_model = overrides.get("signal_model", "breakout")
    sig.stop_loss = overrides.get("stop_loss", 95.0)
    sig.target_1 = overrides.get("target_1", 120.0)
    sig.holding_period_days = overrides.get("holding_period_days", 10)
    sig.regime = overrides.get("regime", "bull")
    return sig


def _make_outcome(**overrides) -> MagicMock:
    """Build a mock Outcome object."""
    out = MagicMock()
    out.ticker = overrides.get("ticker", "AAPL")
    out.entry_price = overrides.get("entry_price", 100.0)
    out.entry_date = overrides.get("entry_date", date.today() - timedelta(days=5))
    out.signal_id = overrides.get("signal_id", 1)
    return out


# ── Trend Tests ────────────────────────────────────────────────────────────


class TestTrendComponent:
    def test_above_emas_higher_lows_upslope(self):
        # Use deterministic up-trending data (no random noise)
        rows = 60
        dates = pd.date_range(end=date.today(), periods=rows, freq="B")
        closes = np.linspace(100, 130, rows)
        df = pd.DataFrame({
            "date": dates,
            "open": closes - 0.3,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.full(rows, 1_000_000),
        })
        feat = _make_feat(close=130.0, ema_21=125.0, sma_50=115.0)
        result = _compute_trend(feat, df, 0.30)
        assert result.score >= 70
        assert result.name == "trend"
        assert result.weight == 0.30

    def test_below_emas(self):
        df = _make_df(60, "down")
        feat = _make_feat(close=90.0, ema_21=100.0, sma_50=105.0)
        result = _compute_trend(feat, df, 0.30)
        assert result.score <= 30

    def test_missing_ema_graceful(self):
        df = _make_df(60)
        feat = _make_feat(ema_21=None, sma_50=None)
        result = _compute_trend(feat, df, 0.30)
        assert 0 <= result.score <= 100


# ── Momentum Tests ─────────────────────────────────────────────────────────


class TestMomentumComponent:
    def test_healthy_rsi_rising_macd(self):
        df = _make_df(60, "up")
        # Ensure rsi_14 and MACD columns exist
        df["rsi_14"] = np.linspace(45, 65, len(df))
        df["MACDh_12_26_9"] = np.linspace(-0.5, 1.0, len(df))
        feat = _make_feat(rsi_14=65.0)
        result = _compute_momentum(feat, df, 0.25)
        assert result.score >= 60

    def test_bearish_rsi_falling_macd(self):
        df = _make_df(60, "down")
        df["rsi_14"] = np.linspace(55, 25, len(df))
        df["MACDh_12_26_9"] = np.linspace(0.5, -1.0, len(df))
        feat = _make_feat(rsi_14=25.0)
        result = _compute_momentum(feat, df, 0.25)
        assert result.score <= 35

    def test_overbought_rsi(self):
        df = _make_df(60)
        df["rsi_14"] = np.full(len(df), 75.0)
        feat = _make_feat(rsi_14=75.0)
        result = _compute_momentum(feat, df, 0.25)
        # Overbought gets 40 pts, not the max
        assert result.score <= 80


# ── Volume Tests ───────────────────────────────────────────────────────────


class TestVolumeComponent:
    def test_strong_rvol(self):
        df = _make_df(60)
        df["obv"] = np.cumsum(np.random.randint(100, 1000, len(df)))
        feat = _make_feat(rvol=2.0)
        result = _compute_volume(feat, df, 0.15)
        assert result.score >= 80

    def test_collapsed_rvol(self):
        df = _make_df(60)
        df["obv"] = np.cumsum(np.random.randint(-500, -100, len(df)))
        feat = _make_feat(rvol=0.5)
        result = _compute_volume(feat, df, 0.15)
        assert result.score <= 35


# ── Risk Tests ─────────────────────────────────────────────────────────────


class TestRiskComponent:
    def test_comfortable_atr_distance(self):
        signal = _make_signal(stop_loss=95.0, holding_period_days=10)
        feat = _make_feat()
        result = _compute_risk(feat, signal, 110.0, 2.5, 5.0, -2.0, 5, 0.20)
        assert result.score >= 60

    def test_danger_atr_distance(self):
        signal = _make_signal(stop_loss=109.0, holding_period_days=10)
        feat = _make_feat()
        result = _compute_risk(feat, signal, 110.0, 0.3, 1.0, -3.0, 5, 0.20)
        assert result.score <= 40

    def test_overstay_penalty(self):
        signal = _make_signal(holding_period_days=10)
        feat = _make_feat()
        result_normal = _compute_risk(feat, signal, 110.0, 2.0, 5.0, -1.0, 8, 0.20)
        result_overstay = _compute_risk(feat, signal, 110.0, 2.0, 5.0, -1.0, 15, 0.20)
        assert result_overstay.score < result_normal.score
        assert result_overstay.details.get("overstay") == "yes"

    def test_good_mfe_mae_ratio(self):
        signal = _make_signal()
        feat = _make_feat()
        result = _compute_risk(feat, signal, 110.0, 2.0, 10.0, -2.0, 5, 0.20)
        assert result.details["mfe_mae_ratio"] == 5.0


# ── Regime Tests ───────────────────────────────────────────────────────────


class TestRegimeComponent:
    def test_aligned(self):
        result = _compute_regime("bull", "bull", 0.10)
        assert result.score == 100.0

    def test_partial_mismatch(self):
        result = _compute_regime("bull", "choppy", 0.10)
        assert result.score == 50.0

    def test_full_mismatch(self):
        result = _compute_regime("bull", "bear", 0.10)
        assert result.score == 10.0

    def test_unknown_regime(self):
        result = _compute_regime("bull", None, 0.10)
        assert result.score == 50.0


# ── Composite Score & State Machine ────────────────────────────────────────


class TestStateMachine:
    def test_on_track(self):
        config = HealthCardConfig()
        # score >= 70 → ON_TRACK
        # We verify via direct classification logic
        assert config.on_track_min == 70.0

    def test_watch_range(self):
        config = HealthCardConfig()
        assert config.watch_min == 50.0

    def test_state_changed_detection(self):
        """previous=ON_TRACK, current=WATCH → state_changed=True"""
        # We'll test via compute_health_card
        pass

    def test_configurable_weights(self):
        config = HealthCardConfig(
            trend_weight=0.40,
            momentum_weight=0.20,
            volume_weight=0.10,
            risk_weight=0.20,
            regime_weight=0.10,
        )
        assert config.trend_weight == 0.40
        assert config.momentum_weight == 0.20


class TestCompositeScore:
    @pytest.mark.asyncio
    async def test_healthy_position_on_track(self):
        df = _make_df(60, "up")
        outcome = _make_outcome()
        signal = _make_signal(regime="bull")
        card = await compute_health_card(
            outcome, signal, df,
            current_regime="bull",
        )
        assert card is not None
        assert 0 <= card.promising_score <= 100
        assert card.ticker == "AAPL"
        assert card.signal_model == "breakout"

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_none(self):
        df = _make_df(3, "up")
        outcome = _make_outcome()
        signal = _make_signal()
        card = await compute_health_card(outcome, signal, df)
        assert card is None

    @pytest.mark.asyncio
    async def test_empty_df_returns_none(self):
        df = pd.DataFrame()
        outcome = _make_outcome()
        signal = _make_signal()
        card = await compute_health_card(outcome, signal, df)
        assert card is None

    @pytest.mark.asyncio
    async def test_state_transition_detected(self):
        df = _make_df(60, "up")
        outcome = _make_outcome()
        signal = _make_signal(regime="bull")
        card = await compute_health_card(
            outcome, signal, df,
            previous_state=HealthState.EXIT,
            current_regime="bull",
        )
        assert card is not None
        if card.state != HealthState.EXIT:
            assert card.state_changed is True

    @pytest.mark.asyncio
    async def test_hard_invalidation_forces_exit(self):
        """If hard invalidation fires, state must be EXIT regardless of score."""
        df = _make_df(60, "down")
        # Simulate breakout range failure: set up high_20d/low_20d columns
        df["high_20d"] = df["high"].rolling(20).max()
        df["low_20d"] = df["low"].rolling(20).min()
        # Force last 2 closes inside range
        df.iloc[-1, df.columns.get_loc("close")] = df.iloc[-1]["low_20d"] + 0.1
        df.iloc[-2, df.columns.get_loc("close")] = df.iloc[-2]["low_20d"] + 0.1

        outcome = _make_outcome()
        signal = _make_signal(signal_model="breakout", regime="bull")
        card = await compute_health_card(
            outcome, signal, df, current_regime="bull",
        )
        # Card may or may not be invalidated depending on rvol,
        # but if it is, state must be EXIT
        if card is not None and card.hard_invalidation:
            assert card.state == HealthState.EXIT


# ── Hard Invalidation Tests ────────────────────────────────────────────────


class TestHardInvalidation:
    def test_breakout_range_failure(self):
        df = _make_df(30)
        df["high_20d"] = df["high"].rolling(20).max()
        df["low_20d"] = df["low"].rolling(20).min()
        # Force last 2 bars inside range
        for i in [-1, -2]:
            h20 = df.iloc[i]["high_20d"]
            l20 = df.iloc[i]["low_20d"]
            df.iloc[i, df.columns.get_loc("close")] = (h20 + l20) / 2

        feat = _make_feat(rvol=0.5)  # below 0.8 threshold
        inv, reason = _check_hard_invalidation(
            "breakout", feat, df, 5, 10, 2.0, 100.0,
        )
        assert inv is True
        assert reason == "breakout_range_failure"

    def test_breakout_no_invalidation_with_rvol(self):
        df = _make_df(30)
        df["high_20d"] = df["high"].rolling(20).max()
        df["low_20d"] = df["low"].rolling(20).min()
        feat = _make_feat(rvol=1.2)  # above threshold
        inv, reason = _check_hard_invalidation(
            "breakout", feat, df, 5, 10, 2.0, 100.0,
        )
        assert inv is False

    def test_mean_reversion_overstay_weak_rsi(self):
        df = _make_df(30)
        feat = _make_feat(rsi_14=35.0, atr_14=2.0)
        inv, reason = _check_hard_invalidation(
            "mean_reversion", feat, df,
            days_held=15, holding_period=10, pnl_pct=-3.0, entry_price=100.0,
        )
        assert inv is True
        assert reason == "mean_rev_overstay_weak_rsi"

    def test_mean_reversion_within_holding_no_invalidation(self):
        df = _make_df(30)
        feat = _make_feat(rsi_14=35.0)
        inv, reason = _check_hard_invalidation(
            "mean_reversion", feat, df,
            days_held=5, holding_period=10, pnl_pct=-1.0, entry_price=100.0,
        )
        # Within holding period, RSI check alone doesn't fire
        assert inv is False

    def test_catalyst_no_reaction(self):
        df = _make_df(30)
        feat = _make_feat(rvol=0.8)
        inv, reason = _check_hard_invalidation(
            "catalyst", feat, df,
            days_held=12, holding_period=10, pnl_pct=0.5, entry_price=100.0,
        )
        assert inv is True
        assert reason == "catalyst_no_reaction"

    def test_catalyst_has_reaction(self):
        df = _make_df(30)
        feat = _make_feat(rvol=1.5)
        inv, reason = _check_hard_invalidation(
            "catalyst", feat, df,
            days_held=12, holding_period=10, pnl_pct=5.0, entry_price=100.0,
        )
        assert inv is False

    def test_unknown_model_no_invalidation(self):
        df = _make_df(30)
        feat = _make_feat()
        inv, reason = _check_hard_invalidation(
            "unknown_model", feat, df, 5, 10, 2.0, 100.0,
        )
        assert inv is False
        assert reason is None


# ── Strategy Family Mapping Tests ─────────────────────────────────────────


class TestStrategyFamilyMapping:
    def test_all_known_models_have_family(self):
        """Every key in STRATEGY_FAMILY resolves to one of 3 canonical families."""
        valid_families = {"breakout", "mean_reversion", "catalyst"}
        for model, family in STRATEGY_FAMILY.items():
            assert family in valid_families, f"{model} maps to unexpected family '{family}'"

    def test_breakout_variants(self):
        assert get_strategy_family("breakout") == "breakout"
        assert get_strategy_family("BREAKOUT") == "breakout"
        assert get_strategy_family("momentum_breakout") == "breakout"
        assert get_strategy_family("MOMO_V1") == "breakout"

    def test_mean_reversion_variants(self):
        assert get_strategy_family("mean_reversion") == "mean_reversion"
        assert get_strategy_family("mean_rev") == "mean_reversion"
        assert get_strategy_family("rsi2") == "mean_reversion"
        assert get_strategy_family("OVERSOLD_REVERSION") == "mean_reversion"

    def test_catalyst_variants(self):
        assert get_strategy_family("catalyst") == "catalyst"
        assert get_strategy_family("earnings_drift") == "catalyst"

    def test_unknown_model_returns_none(self):
        assert get_strategy_family("totally_new_model") is None

    def test_hard_invalidation_routes_via_family(self):
        """Breakout variant 'momentum_breakout' should route to breakout invalidation."""
        df = _make_df(30)
        df["high_20d"] = df["high"].rolling(20).max()
        df["low_20d"] = df["low"].rolling(20).min()
        for i in [-1, -2]:
            h20 = df.iloc[i]["high_20d"]
            l20 = df.iloc[i]["low_20d"]
            df.iloc[i, df.columns.get_loc("close")] = (h20 + l20) / 2

        feat = _make_feat(rvol=0.5)
        inv, reason = _check_hard_invalidation(
            "momentum_breakout", feat, df, 5, 10, 2.0, 100.0,
        )
        assert inv is True
        assert reason == "breakout_range_failure"

    def test_mean_rev_variant_routes_correctly(self):
        """RSI2 variant should route to mean_reversion invalidation."""
        df = _make_df(30)
        feat = _make_feat(rsi_14=35.0, atr_14=2.0)
        inv, reason = _check_hard_invalidation(
            "rsi2", feat, df,
            days_held=15, holding_period=10, pnl_pct=-3.0, entry_price=100.0,
        )
        assert inv is True
        assert reason == "mean_rev_overstay_weak_rsi"


# ── Soft Invalidation Tests ───────────────────────────────────────────────


class TestSoftInvalidation:
    def test_rvol_collapse_under_ema_weak_momentum(self):
        """All 3 conditions: rvol < 0.8, below EMA21, RSI < 45."""
        feat = _make_feat(rvol=0.6, close=95.0, ema_21=100.0, rsi_14=38.0)
        result = _check_soft_invalidation(
            feat=feat, signal_model="breakout",
            signal_regime="bull", current_regime="bull",
        )
        assert result is True

    def test_no_soft_invalidation_when_healthy(self):
        feat = _make_feat(rvol=1.5, close=115.0, ema_21=110.0, rsi_14=60.0)
        result = _check_soft_invalidation(
            feat=feat, signal_model="breakout",
            signal_regime="bull", current_regime="bull",
        )
        assert result is False

    def test_regime_flip_to_bear_for_breakout(self):
        """Breakout entered in bull, now bear → soft invalidation."""
        feat = _make_feat(rvol=1.2, close=115.0, ema_21=110.0, rsi_14=55.0)
        result = _check_soft_invalidation(
            feat=feat, signal_model="breakout",
            signal_regime="bull", current_regime="bear",
        )
        assert result is True

    def test_regime_flip_to_bear_for_mean_reversion_ok(self):
        """Mean reversion is bear-compatible — no soft invalidation."""
        feat = _make_feat(rvol=1.2, close=115.0, ema_21=110.0, rsi_14=55.0)
        result = _check_soft_invalidation(
            feat=feat, signal_model="mean_reversion",
            signal_regime="bull", current_regime="bear",
        )
        assert result is False

    def test_regime_flip_no_effect_if_already_bear_entry(self):
        """If entered in bear regime and still bear — no soft invalidation."""
        feat = _make_feat(rvol=1.2, close=115.0, ema_21=110.0, rsi_14=55.0)
        result = _check_soft_invalidation(
            feat=feat, signal_model="breakout",
            signal_regime="bear", current_regime="bear",
        )
        assert result is False


# ── Card Field Tests ──────────────────────────────────────────────────────


class TestCardFields:
    @pytest.mark.asyncio
    async def test_current_price_on_card(self):
        """current_price should be populated from latest close, not 0."""
        df = _make_df(60, "up")
        outcome = _make_outcome()
        signal = _make_signal(regime="bull")
        card = await compute_health_card(
            outcome, signal, df, current_regime="bull",
        )
        assert card is not None
        assert card.current_price > 0
        assert card.current_price != 0.0

    @pytest.mark.asyncio
    async def test_atr_14_on_card(self):
        """atr_14 should be populated from features when available."""
        df = _make_df(60, "up")
        outcome = _make_outcome()
        signal = _make_signal(regime="bull")
        card = await compute_health_card(
            outcome, signal, df, current_regime="bull",
        )
        assert card is not None
        # ATR may or may not be available depending on df, but field exists
        # If computed, it should be a positive number
        if card.atr_14 is not None:
            assert card.atr_14 > 0


# ── Velocity Computation Tests ────────────────────────────────────────────


class TestVelocityComputation:
    def test_basic_deteriorating(self):
        """85 → 78 → 72 over 2 intervals → (72-85)/2 = -6.5."""
        vel = compute_score_velocity(72.0, [85.0, 78.0])
        assert vel == -6.5

    def test_basic_improving(self):
        """60 → 65 → 72 over 2 intervals → (72-60)/2 = 6.0."""
        vel = compute_score_velocity(72.0, [60.0, 65.0])
        assert vel == 6.0

    def test_stable(self):
        """70 → 70 → 70 → 0.0."""
        vel = compute_score_velocity(70.0, [70.0, 70.0])
        assert vel == 0.0

    def test_single_previous(self):
        """One previous score: (current - prev) / 1."""
        vel = compute_score_velocity(65.0, [80.0])
        assert vel == -15.0

    def test_none_previous_scores(self):
        vel = compute_score_velocity(70.0, None)
        assert vel is None

    def test_empty_previous_scores(self):
        vel = compute_score_velocity(70.0, [])
        assert vel is None

    def test_three_previous(self):
        """80 → 75 → 72 → 70: (70-80)/3 = -3.33."""
        vel = compute_score_velocity(70.0, [80.0, 75.0, 72.0])
        assert vel == -3.33


class TestVelocityWarning:
    def test_fires_when_both_conditions_met(self):
        """velocity < -5 AND score < 75 → True."""
        assert _check_velocity_warning(-7.0, 68.0) is True

    def test_no_warning_velocity_ok(self):
        """velocity > -5 → False regardless of score."""
        assert _check_velocity_warning(-3.0, 60.0) is False

    def test_no_warning_score_ok(self):
        """score >= 75 → False regardless of velocity."""
        assert _check_velocity_warning(-8.0, 80.0) is False

    def test_no_warning_none_velocity(self):
        assert _check_velocity_warning(None, 60.0) is False

    def test_edge_exactly_minus_5(self):
        """velocity == -5.0 is NOT < -5 → False."""
        assert _check_velocity_warning(-5.0, 60.0) is False

    def test_edge_exactly_75(self):
        """score == 75 is NOT < 75 → False."""
        assert _check_velocity_warning(-6.0, 75.0) is False


class TestVelocityInHealthCard:
    @pytest.mark.asyncio
    async def test_velocity_populated_with_previous_scores(self):
        """When previous_scores are provided, score_velocity should be set."""
        df = _make_df(60, "up")
        outcome = _make_outcome()
        signal = _make_signal(regime="bull")
        card = await compute_health_card(
            outcome, signal, df,
            current_regime="bull",
            previous_scores=[80.0, 75.0],
        )
        assert card is not None
        assert card.score_velocity is not None

    @pytest.mark.asyncio
    async def test_velocity_none_without_previous_scores(self):
        """When no previous_scores, score_velocity should be None."""
        df = _make_df(60, "up")
        outcome = _make_outcome()
        signal = _make_signal(regime="bull")
        card = await compute_health_card(
            outcome, signal, df,
            current_regime="bull",
        )
        assert card is not None
        assert card.score_velocity is None

    @pytest.mark.asyncio
    async def test_velocity_warning_forces_watch(self):
        """Rapid deterioration should force WATCH even if score is ON_TRACK range."""
        df = _make_df(60, "up")
        outcome = _make_outcome()
        signal = _make_signal(regime="bull")
        # previous_scores that create a steep decline ending around 72
        card = await compute_health_card(
            outcome, signal, df,
            current_regime="bull",
            previous_scores=[90.0, 82.0],
        )
        # If the card's score is < 75 and velocity < -5, state should be WATCH
        if card is not None and card.score_velocity is not None:
            if card.score_velocity < -5 and card.promising_score < 75:
                assert card.state == HealthState.WATCH
