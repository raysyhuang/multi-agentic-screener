"""Tests for the deterministic cross-engine synthesizer."""

from __future__ import annotations

from unittest.mock import MagicMock


from src.engines.deterministic_synthesizer import (
    deterministic_regime_weight_adjust,
    synthesize_deterministic,
)


def _make_stats(weight: float = 0.5, hit_rate: float = 0.3, per_strategy: dict | None = None):
    stats = MagicMock()
    stats.weight = weight
    stats.hit_rate = hit_rate
    stats.per_strategy = per_strategy or {}
    return stats


def _make_pick(ticker: str, engine: str = "eng_a", score: float = 50.0, engines: int = 1):
    return {
        "ticker": ticker,
        "engine_name": engine,
        "combined_score": score,
        "avg_weighted_confidence": score * 0.8,
        "engine_count": engines,
        "engines": [engine] if engines == 1 else ["eng_a", "eng_b"],
        "strategies": ["mean_reversion"],
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "target_price": 105.0,
        "holding_period_days": 3,
        "thesis": "Test thesis",
        "convergence_type": "ticker" if engines >= 2 else "none",
    }


class TestDeterministicRegimeWeightAdjust:
    def test_bear_boosts_reversion(self):
        stats = {"eng": _make_stats(weight=1.0, per_strategy={"mean_reversion": {}})}
        adj = deterministic_regime_weight_adjust(stats, "bear")
        assert "eng" in adj
        assert stats["eng"].weight > 1.0

    def test_bear_penalizes_momentum(self):
        stats = {"eng": _make_stats(weight=1.0, per_strategy={"momentum_breakout": {}})}
        adj = deterministic_regime_weight_adjust(stats, "bear")
        assert "eng" in adj
        assert stats["eng"].weight < 1.0

    def test_bull_boosts_momentum(self):
        stats = {"eng": _make_stats(weight=1.0, per_strategy={"momentum_breakout": {}})}
        adj = deterministic_regime_weight_adjust(stats, "bull")
        assert "eng" in adj
        assert stats["eng"].weight > 1.0

    def test_unknown_regime_no_change(self):
        stats = {"eng": _make_stats(weight=1.0, per_strategy={"mean_reversion": {}})}
        adj = deterministic_regime_weight_adjust(stats, "unknown")
        assert len(adj) == 0

    def test_weight_clamped(self):
        stats = {"eng": _make_stats(weight=0.05, per_strategy={"mean_reversion": {}})}
        deterministic_regime_weight_adjust(stats, "bear")
        assert stats["eng"].weight >= 0.1


class TestSynthesizeDeterministic:
    def test_empty_picks(self):
        result = synthesize_deterministic([], "bear", 2)
        assert result.portfolio == []
        assert result.convergent_picks == []
        assert "No tradeable" in result.executive_summary

    def test_unique_picks_become_portfolio(self):
        picks = [_make_pick("AAPL", score=80), _make_pick("MSFT", score=70)]
        result = synthesize_deterministic(picks, "bear", 2)
        assert len(result.portfolio) == 2
        assert result.portfolio[0].ticker == "AAPL"
        assert result.portfolio[1].ticker == "MSFT"
        assert all(p.source == "unique" for p in result.portfolio)

    def test_convergent_picks_prioritized(self):
        picks = [
            _make_pick("CONV", engines=2, score=60),
            _make_pick("UNIQ", engines=1, score=90),
        ]
        result = synthesize_deterministic(picks, "bear", 2)
        assert result.portfolio[0].ticker == "CONV"
        assert result.portfolio[0].source == "convergent"
        assert len(result.convergent_picks) == 1

    def test_max_portfolio_positions(self):
        picks = [_make_pick(f"T{i}", score=100 - i) for i in range(10)]
        result = synthesize_deterministic(picks, "bull", 3)
        assert len(result.portfolio) <= 5

    def test_regime_in_output(self):
        picks = [_make_pick("AAPL")]
        result = synthesize_deterministic(picks, "bear", 2)
        assert result.regime_consensus == "bear"

    def test_equal_weights(self):
        picks = [_make_pick("AAPL"), _make_pick("MSFT")]
        result = synthesize_deterministic(picks, "bull", 2)
        assert result.portfolio[0].weight_pct == result.portfolio[1].weight_pct

    def test_price_fields_populated(self):
        picks = [_make_pick("AAPL")]
        result = synthesize_deterministic(picks, "bear", 1)
        pos = result.portfolio[0]
        assert pos.entry_price == 100.0
        assert pos.stop_loss == 98.0
        assert pos.target_price == 105.0
