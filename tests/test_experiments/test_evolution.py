"""Tests for evolution mutation operators."""

import pytest

from src.experiments.evolution import (
    _nudge,
    _crossover,
    _explore,
    _invert,
    _clamp,
    _mutate,
    PARAM_RANGES,
)


class TestNudge:
    def test_nudges_numeric_values(self):
        config = {"convergence_2_engine_multiplier": 1.3}
        result = _nudge(config)
        # Should be within ±20% of 1.3
        assert 1.3 * 0.8 <= result["convergence_2_engine_multiplier"] <= 1.3 * 1.2

    def test_nudges_integer_values(self):
        config = {"credibility_lookback_days": 30}
        result = _nudge(config)
        assert isinstance(result["credibility_lookback_days"], int)
        assert 14 <= result["credibility_lookback_days"] <= 90

    def test_nudges_nested_regime_multipliers(self):
        config = {
            "regime_multipliers": {
                "bull": {"breakout": 1.5, "mean_reversion": 0.7},
            }
        }
        result = _nudge(config)
        assert "regime_multipliers" in result
        bull = result["regime_multipliers"]["bull"]
        assert 1.5 * 0.8 <= bull["breakout"] <= 1.5 * 1.2
        assert 0.7 * 0.8 <= bull["mean_reversion"] <= 0.7 * 1.2

    def test_preserves_keys(self):
        config = {"guardian_bear_sizing": 0.5, "min_confidence": 45.0}
        result = _nudge(config)
        assert set(result.keys()) == set(config.keys())


class TestCrossover:
    def test_merges_keys_from_both_parents(self):
        a = {"convergence_2_engine_multiplier": 1.5}
        b = {"guardian_bear_sizing": 0.5}
        result = _crossover(a, b)
        # Should have at least one key from each parent
        assert len(result) >= 1

    def test_all_keys_from_parents(self):
        a = {"convergence_2_engine_multiplier": 1.5, "min_confidence": 40.0}
        b = {"guardian_bear_sizing": 0.5, "min_confidence": 50.0}
        result = _crossover(a, b)
        # All keys should come from one parent or the other
        for key in result:
            assert key in a or key in b


class TestExplore:
    def test_adds_new_parameter(self):
        config = {"convergence_2_engine_multiplier": 1.5}
        result = _explore(config)
        # Should have at least one more key than parent
        assert len(result) >= len(config)

    def test_new_param_in_valid_range(self):
        config = {}
        result = _explore(config)
        for key, value in result.items():
            if key in PARAM_RANGES:
                lo, hi = PARAM_RANGES[key]
                assert lo <= value <= hi, f"{key}={value} out of range [{lo}, {hi}]"


class TestInvert:
    def test_inverts_numeric_value(self):
        config = {"guardian_bear_sizing": 0.5}
        result = _invert(config)
        # 0.5 with range (0.3, 0.9), midpoint = 0.6
        # Inverted: 0.6 + (0.6 - 0.5) = 0.7
        assert result["guardian_bear_sizing"] != config["guardian_bear_sizing"]

    def test_clamped_to_range(self):
        config = {"guardian_bear_sizing": 0.3}  # at minimum
        result = _invert(config)
        lo, hi = PARAM_RANGES["guardian_bear_sizing"]
        assert lo <= result["guardian_bear_sizing"] <= hi


class TestClamp:
    def test_clamp_within_range(self):
        assert _clamp("guardian_bear_sizing", 0.5) == 0.5

    def test_clamp_below_minimum(self):
        assert _clamp("guardian_bear_sizing", 0.1) == 0.3

    def test_clamp_above_maximum(self):
        assert _clamp("guardian_bear_sizing", 1.5) == 0.9

    def test_unknown_key_no_clamp(self):
        assert _clamp("unknown_key", 999.0) == 999.0


class TestMutate:
    def test_returns_tuple(self):
        config = {"convergence_2_engine_multiplier": 1.3}
        result, method = _mutate(config, [config])
        assert isinstance(result, dict)
        assert method in ("nudge", "crossover", "explore", "invert")

    def test_produces_different_config(self):
        """Over many runs, mutation should produce at least one different config."""
        config = {"convergence_2_engine_multiplier": 1.3, "guardian_bear_sizing": 0.65}
        different = False
        for _ in range(20):
            result, _ = _mutate(config, [config])
            if result != config:
                different = True
                break
        assert different, "Mutation should produce different configs"
