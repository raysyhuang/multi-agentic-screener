"""Tests for shadow track runner — override application logic."""

import pytest

from src.engines.credibility import EngineStats
from src.experiments.runner import _run_single_track, _apply_guardian_sizing


class FakeShadowTrack:
    """Minimal stub matching ShadowTrack interface."""
    def __init__(self, name, config):
        self.id = 1
        self.name = name
        self.config = config
        self.status = "active"


class TestRunSingleTrack:
    def _make_picks(self):
        return [
            {
                "ticker": "MSFT",
                "engine_name": "koocore_d",
                "confidence": 80.0,
                "strategy": "breakout",
                "sector": "Technology",
                "entry_price": 400.0,
                "stop_loss": 390.0,
                "target_price": 420.0,
                "holding_period_days": 5,
                "metadata": {},
            },
            {
                "ticker": "MSFT",
                "engine_name": "gemini_stst",
                "confidence": 70.0,
                "strategy": "momentum",
                "sector": "Technology",
                "entry_price": 400.0,
                "stop_loss": 392.0,
                "target_price": 415.0,
                "holding_period_days": 5,
                "metadata": {},
            },
        ]

    def _make_stats(self):
        return {
            "koocore_d": EngineStats(engine_name="koocore_d", weight=1.0),
            "gemini_stst": EngineStats(engine_name="gemini_stst", weight=1.0),
        }

    def test_baseline_no_overrides(self):
        track = FakeShadowTrack("baseline", {})
        from src.config import get_settings
        settings = get_settings()

        picks = _run_single_track(
            track, self._make_picks(), self._make_stats(),
            {"regime": "bull"}, settings,
        )
        assert len(picks) == 1  # MSFT grouped
        assert picks[0]["ticker"] == "MSFT"

    def test_high_convergence_override(self):
        track = FakeShadowTrack("high_conv", {"convergence_2_engine_multiplier": 2.0})
        from src.config import get_settings
        settings = get_settings()

        baseline_track = FakeShadowTrack("base", {})
        baseline = _run_single_track(
            baseline_track, self._make_picks(), self._make_stats(),
            {"regime": "bull"}, settings,
        )
        boosted = _run_single_track(
            track, self._make_picks(), self._make_stats(),
            {"regime": "bull"}, settings,
        )
        assert boosted[0]["combined_score"] > baseline[0]["combined_score"]

    def test_min_confidence_filter(self):
        track = FakeShadowTrack("high_conf", {"min_confidence": 90.0})
        from src.config import get_settings
        settings = get_settings()

        picks = _run_single_track(
            track, self._make_picks(), self._make_stats(),
            {"regime": "bull"}, settings,
        )
        # All picks have confidence ~75 avg, so min_confidence=90 should filter them
        assert len(picks) == 0


class TestGuardianSizing:
    def test_bull_regime_full_sizing(self):
        picks = [
            {"combined_score": 60.0},
            {"combined_score": 50.0},
        ]
        from src.config import get_settings
        settings = get_settings()
        result = _apply_guardian_sizing(
            picks, {}, {"regime": "bull"}, settings,
        )
        # Bull regime_factor = 1.0
        assert result[0].get("weight_pct", 0) > 0

    def test_bear_regime_reduced_sizing(self):
        picks = [{"combined_score": 60.0}]
        from src.config import get_settings
        settings = get_settings()

        bull_result = _apply_guardian_sizing(
            [{"combined_score": 60.0}], {}, {"regime": "bull"}, settings,
        )
        bear_result = _apply_guardian_sizing(
            [{"combined_score": 60.0}],
            {"guardian_bear_sizing": 0.5},
            {"regime": "bear"},
            settings,
        )
        assert bear_result[0]["weight_pct"] < bull_result[0]["weight_pct"]

    def test_heat_cap(self):
        picks = [
            {"combined_score": 80.0},
            {"combined_score": 70.0},
            {"combined_score": 60.0},
        ]
        from src.config import get_settings
        settings = get_settings()
        result = _apply_guardian_sizing(
            picks,
            {"guardian_max_portfolio_heat_pct": 5.0},
            {"regime": "bull"},
            settings,
        )
        total_weight = sum(p.get("weight_pct", 0) for p in result)
        assert total_weight <= 5.1  # allow small rounding
