"""Tests for parameterized credibility and ranker functions."""

from src.engines.credibility import (
    compute_convergence_multiplier,
    compute_weighted_picks,
    EngineStats,
)


class TestConvergenceMultiplierOverrides:
    def test_no_overrides_uses_settings(self):
        """Without overrides, should return settings defaults."""
        result = compute_convergence_multiplier(2)
        # Default is 1.3 from Settings
        assert result == 1.3

    def test_override_2_engine(self):
        overrides = {"convergence_2_engine_multiplier": 2.0}
        assert compute_convergence_multiplier(2, overrides=overrides) == 2.0

    def test_override_3_engine(self):
        overrides = {"convergence_3_engine_multiplier": 1.5}
        assert compute_convergence_multiplier(3, overrides=overrides) == 1.5

    def test_override_1_engine(self):
        overrides = {"convergence_1_engine_multiplier": 0.7}
        assert compute_convergence_multiplier(1, overrides=overrides) == 0.7

    def test_override_4_engine(self):
        overrides = {"convergence_4_engine_multiplier": 3.0}
        assert compute_convergence_multiplier(4, overrides=overrides) == 3.0

    def test_partial_override_falls_back(self):
        """Override only 2-engine, rest should fall back to settings."""
        overrides = {"convergence_2_engine_multiplier": 2.5}
        # 3-engine should use default (1.0)
        assert compute_convergence_multiplier(3, overrides=overrides) == 1.0
        # 2-engine should use override
        assert compute_convergence_multiplier(2, overrides=overrides) == 2.5


class TestWeightedPicksOverrides:
    def _make_picks(self):
        return [
            {
                "ticker": "AAPL",
                "engine_name": "koocore_d",
                "confidence": 75.0,
                "strategy": "breakout",
                "sector": "Technology",
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "target_price": 160.0,
                "holding_period_days": 5,
                "metadata": {},
            },
            {
                "ticker": "AAPL",
                "engine_name": "gemini_stst",
                "confidence": 70.0,
                "strategy": "momentum",
                "sector": "Technology",
                "entry_price": 150.0,
                "stop_loss": 146.0,
                "target_price": 158.0,
                "holding_period_days": 5,
                "metadata": {},
            },
        ]

    def _make_stats(self):
        return {
            "koocore_d": EngineStats(engine_name="koocore_d", weight=1.2),
            "gemini_stst": EngineStats(engine_name="gemini_stst", weight=0.9),
        }

    def test_no_overrides(self):
        picks = compute_weighted_picks(self._make_picks(), self._make_stats())
        assert len(picks) == 1  # AAPL grouped
        assert picks[0]["ticker"] == "AAPL"

    def test_with_convergence_override(self):
        """Higher convergence multiplier should increase combined_score."""
        baseline = compute_weighted_picks(self._make_picks(), self._make_stats())
        boosted = compute_weighted_picks(
            self._make_picks(), self._make_stats(),
            config_overrides={"convergence_2_engine_multiplier": 2.0},
        )
        assert boosted[0]["combined_score"] > baseline[0]["combined_score"]

    def test_lower_convergence_override(self):
        """Lower convergence multiplier should decrease combined_score."""
        baseline = compute_weighted_picks(self._make_picks(), self._make_stats())
        weakened = compute_weighted_picks(
            self._make_picks(), self._make_stats(),
            config_overrides={"convergence_2_engine_multiplier": 0.5},
        )
        assert weakened[0]["combined_score"] < baseline[0]["combined_score"]
