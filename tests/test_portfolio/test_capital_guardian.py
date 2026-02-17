"""Tests for the Capital Guardian — portfolio-level risk defense."""

from __future__ import annotations

import pytest

from src.portfolio.capital_guardian import (
    PortfolioRiskState,
    GuardianVerdict,
    compute_guardian_verdict,
    apply_guardian_to_portfolio,
    format_guardian_summary,
)


def _make_state(**overrides) -> PortfolioRiskState:
    """Create a PortfolioRiskState with sensible defaults."""
    defaults = dict(
        equity_curve_pct=[1.0, 2.0, 3.0],
        peak_equity_pct=3.0,
        current_equity_pct=3.0,
        current_drawdown_pct=0.0,
        recent_results=[5.0, -2.0, 3.0, 1.0, -1.0],
        consecutive_losses=0,
        consecutive_wins=0,
        open_position_count=1,
        total_open_risk_pct=3.0,
        recent_win_rate=0.60,
        recent_avg_return=1.2,
        total_closed_trades=20,
    )
    defaults.update(overrides)
    return PortfolioRiskState(**defaults)


def _make_portfolio(n: int = 3) -> list[dict]:
    """Create a simple portfolio of n positions."""
    return [
        {
            "ticker": f"TICK{i}",
            "weight_pct": 15.0,
            "source": "convergent",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "target_price": 110.0,
            "holding_period_days": 7,
        }
        for i in range(n)
    ]


class TestDrawdownCircuitBreaker:
    def test_no_drawdown_full_sizing(self):
        state = _make_state(current_drawdown_pct=0.0)
        verdict = compute_guardian_verdict(state, regime="bull")
        assert verdict.sizing_multiplier == 1.0
        assert not verdict.halt

    def test_moderate_drawdown_reduces_sizing(self):
        state = _make_state(current_drawdown_pct=-10.0)  # 50% of 20% threshold
        verdict = compute_guardian_verdict(state, regime="bull")
        assert 0.5 < verdict.sizing_multiplier < 1.0
        assert not verdict.halt

    def test_max_drawdown_halts(self):
        state = _make_state(current_drawdown_pct=-21.0)
        verdict = compute_guardian_verdict(state, regime="bull")
        assert verdict.halt
        assert verdict.sizing_multiplier == 0.0
        assert "circuit breaker" in verdict.halt_reason.lower()

    def test_at_exact_threshold_halts(self):
        state = _make_state(current_drawdown_pct=-20.0)
        verdict = compute_guardian_verdict(state, regime="bull")
        assert verdict.halt


class TestStreakReduction:
    def test_no_streak_full_sizing(self):
        state = _make_state(consecutive_losses=0)
        verdict = compute_guardian_verdict(state, regime="bull")
        assert verdict.streak_factor == 1.0

    def test_below_threshold_no_reduction(self):
        state = _make_state(consecutive_losses=2)
        verdict = compute_guardian_verdict(state, regime="bull")
        assert verdict.streak_factor == 1.0

    def test_at_threshold_starts_reduction(self):
        state = _make_state(consecutive_losses=3)
        verdict = compute_guardian_verdict(state, regime="bull")
        # 3 losses = threshold, 0 excess, factor should still be 1.0
        assert verdict.streak_factor == 1.0

    def test_above_threshold_reduces(self):
        state = _make_state(consecutive_losses=4)
        verdict = compute_guardian_verdict(state, regime="bull")
        assert verdict.streak_factor == 0.75  # 1 excess × 0.25 reduction

    def test_halt_threshold_halts(self):
        state = _make_state(consecutive_losses=6)
        verdict = compute_guardian_verdict(state, regime="bull")
        assert verdict.halt
        assert "loss streak" in verdict.halt_reason.lower()

    def test_winning_streak_mild_bonus(self):
        state = _make_state(consecutive_wins=5, consecutive_losses=0)
        verdict = compute_guardian_verdict(state, regime="bull")
        assert verdict.streak_factor > 1.0
        assert verdict.streak_factor <= 1.25  # capped


class TestRegimeScaling:
    def test_bull_full_sizing(self):
        state = _make_state()
        verdict = compute_guardian_verdict(state, regime="bull")
        assert verdict.regime_factor == 1.0

    def test_bear_reduced(self):
        state = _make_state()
        verdict = compute_guardian_verdict(state, regime="bear")
        assert verdict.regime_factor == 0.5

    def test_choppy_reduced(self):
        state = _make_state()
        verdict = compute_guardian_verdict(state, regime="choppy")
        assert verdict.regime_factor == 0.75

    def test_unknown_regime_conservative(self):
        state = _make_state()
        verdict = compute_guardian_verdict(state, regime="unknown_regime")
        assert verdict.regime_factor == 0.75


class TestPortfolioHeat:
    def test_low_heat_no_reduction(self):
        state = _make_state(total_open_risk_pct=3.0)
        verdict = compute_guardian_verdict(state, regime="bull")
        assert verdict.heat_factor == 1.0

    def test_high_heat_blocks(self):
        state = _make_state(total_open_risk_pct=10.0)  # at max
        verdict = compute_guardian_verdict(state, regime="bull")
        assert verdict.heat_factor == 0.0

    def test_approaching_heat_cap_reduces(self):
        state = _make_state(total_open_risk_pct=8.0)  # 80% of 10% cap
        verdict = compute_guardian_verdict(state, regime="bull")
        assert 0.0 < verdict.heat_factor < 1.0


class TestCombinedFactors:
    def test_bear_with_drawdown(self):
        state = _make_state(current_drawdown_pct=-10.0)
        verdict = compute_guardian_verdict(state, regime="bear")
        # Both drawdown and regime should reduce
        assert verdict.sizing_multiplier < 0.5
        assert verdict.drawdown_factor < 1.0
        assert verdict.regime_factor == 0.5

    def test_all_factors_multiply(self):
        state = _make_state(
            current_drawdown_pct=-5.0,
            consecutive_losses=4,
            total_open_risk_pct=8.0,
        )
        verdict = compute_guardian_verdict(state, regime="choppy")
        # All four factors should be < 1.0
        assert verdict.drawdown_factor < 1.0
        assert verdict.streak_factor < 1.0
        assert verdict.regime_factor < 1.0
        assert verdict.heat_factor < 1.0
        # Combined should be quite small
        assert verdict.sizing_multiplier < 0.35


class TestApplyGuardian:
    def test_halt_returns_empty(self):
        portfolio = _make_portfolio(3)
        verdict = GuardianVerdict(halt=True, halt_reason="test halt")
        result = apply_guardian_to_portfolio(portfolio, verdict)
        assert result == []

    def test_full_sizing_preserves_weights(self):
        portfolio = _make_portfolio(3)
        verdict = GuardianVerdict(sizing_multiplier=1.0)
        result = apply_guardian_to_portfolio(portfolio, verdict)
        assert len(result) == 3
        for p in result:
            assert p["weight_pct"] == 15.0

    def test_half_sizing_halves_weights(self):
        portfolio = _make_portfolio(3)
        verdict = GuardianVerdict(sizing_multiplier=0.5)
        result = apply_guardian_to_portfolio(portfolio, verdict)
        assert len(result) == 3
        for p in result:
            assert p["weight_pct"] == 7.5
            assert p.get("guardian_adjusted") is True
            assert p["original_weight_pct"] == 15.0

    def test_very_small_multiplier_filters_positions(self):
        portfolio = _make_portfolio(3)
        # With weight 15% and multiplier 0.05, new weight = 0.75% < 1.0% minimum
        verdict = GuardianVerdict(sizing_multiplier=0.05)
        result = apply_guardian_to_portfolio(portfolio, verdict)
        assert len(result) == 0

    def test_per_trade_risk_cap(self):
        portfolio = [{
            "ticker": "HIGH_RISK",
            "weight_pct": 30.0,
            "entry_price": 100.0,
            "stop_loss": 80.0,  # 20% risk per unit — needs capping
            "target_price": 130.0,
        }]
        verdict = GuardianVerdict(sizing_multiplier=1.0)
        # With 2% risk cap and 20% risk per unit: max_weight = (2/20)*100 = 10%
        result = apply_guardian_to_portfolio(portfolio, verdict)
        assert len(result) == 1
        assert result[0]["weight_pct"] == 10.0


class TestFormatSummary:
    def test_halt_message(self):
        verdict = GuardianVerdict(halt=True, halt_reason="drawdown exceeded")
        summary = format_guardian_summary(verdict)
        assert "HALT" in summary
        assert "drawdown exceeded" in summary

    def test_normal_message(self):
        state = _make_state(
            current_drawdown_pct=-5.0,
            consecutive_losses=2,
            open_position_count=2,
            total_open_risk_pct=4.0,
        )
        verdict = GuardianVerdict(
            sizing_multiplier=0.8,
            drawdown_factor=0.85,
            streak_factor=1.0,
            regime_factor=1.0,
            heat_factor=1.0,
            risk_state=state,
        )
        summary = format_guardian_summary(verdict)
        assert "80%" in summary
        assert "Drawdown" in summary
        assert "2 positions" in summary
