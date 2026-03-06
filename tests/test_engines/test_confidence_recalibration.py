"""Tests for confidence recalibration in credibility system."""

from src.config import get_settings
from src.engines.credibility import EngineStats, PaperTradeStats, compute_weighted_picks


def _make_pick(ticker: str, engine: str, strategy: str = "mean_reversion", confidence: float = 60.0) -> dict:
    return {
        "ticker": ticker,
        "engine_name": engine,
        "strategy": strategy,
        "confidence": confidence,
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "target_price": 110.0,
        "holding_period_days": 3,
        "thesis": "test",
        "risk_factors": [],
        "sector": "Tech",
    }


def _make_engine_stats(
    name: str,
    hit_rate: float = 0.5,
    avg_confidence: float = 60.0,
    resolved_picks: int = 20,
    weight: float = 1.0,
    per_strategy: dict | None = None,
) -> EngineStats:
    stats = EngineStats(engine_name=name)
    stats.hit_rate = hit_rate
    stats.avg_confidence = avg_confidence
    stats.resolved_picks = resolved_picks
    stats.weight = weight
    stats.has_enough_data = True
    if per_strategy:
        stats.per_strategy = per_strategy
    return stats


def test_overconfident_engine_gets_scaled_down():
    """An engine claiming 60% confidence but hitting 14% should be scaled down."""
    settings = get_settings()
    old_enabled = settings.credibility_recalibration_enabled
    old_min = settings.credibility_recalibration_min_picks
    old_floor_enabled = settings.credibility_strategy_floor_enabled
    try:
        settings.credibility_recalibration_enabled = True
        settings.credibility_recalibration_min_picks = 5
        settings.credibility_strategy_floor_enabled = False

        picks = [_make_pick("AAPL", "overconfident_engine", confidence=60.0)]
        engine_stats = {
            "overconfident_engine": _make_engine_stats(
                "overconfident_engine",
                hit_rate=0.14,          # 14% actual
                avg_confidence=60.0,    # claims 60%
                resolved_picks=20,
            ),
        }

        result = compute_weighted_picks(picks, engine_stats)
        assert len(result) == 1
        # scale = 0.14 / 0.60 = 0.233, clamped to 0.3
        # adjusted confidence = 60 * 0.3 = 18.0
        assert result[0]["avg_weighted_confidence"] <= 20.0
    finally:
        settings.credibility_recalibration_enabled = old_enabled
        settings.credibility_recalibration_min_picks = old_min
        settings.credibility_strategy_floor_enabled = old_floor_enabled


def test_well_calibrated_engine_unchanged():
    """An engine with confidence matching reality should be mostly unchanged."""
    settings = get_settings()
    old_enabled = settings.credibility_recalibration_enabled
    old_min = settings.credibility_recalibration_min_picks
    old_floor_enabled = settings.credibility_strategy_floor_enabled
    try:
        settings.credibility_recalibration_enabled = True
        settings.credibility_recalibration_min_picks = 5
        settings.credibility_strategy_floor_enabled = False

        picks = [_make_pick("AAPL", "good_engine", confidence=55.0)]
        engine_stats = {
            "good_engine": _make_engine_stats(
                "good_engine",
                hit_rate=0.55,          # 55% actual
                avg_confidence=55.0,    # claims 55%
                resolved_picks=20,
            ),
        }

        result = compute_weighted_picks(picks, engine_stats)
        assert len(result) == 1
        # scale = 0.55 / 0.55 = 1.0 → no change
        conf = result[0]["avg_weighted_confidence"]
        assert 53.0 <= conf <= 57.0  # approximately unchanged
    finally:
        settings.credibility_recalibration_enabled = old_enabled
        settings.credibility_recalibration_min_picks = old_min
        settings.credibility_strategy_floor_enabled = old_floor_enabled


def test_recalibration_skips_engines_below_min_picks():
    """Engines with too few resolved picks should not be recalibrated."""
    settings = get_settings()
    old_enabled = settings.credibility_recalibration_enabled
    old_min = settings.credibility_recalibration_min_picks
    old_floor_enabled = settings.credibility_strategy_floor_enabled
    try:
        settings.credibility_recalibration_enabled = True
        settings.credibility_recalibration_min_picks = 15
        settings.credibility_strategy_floor_enabled = False

        picks = [_make_pick("AAPL", "new_engine", confidence=70.0)]
        engine_stats = {
            "new_engine": _make_engine_stats(
                "new_engine",
                hit_rate=0.10,          # terrible but only 5 picks
                avg_confidence=70.0,
                resolved_picks=5,       # below min_picks=15
            ),
        }

        result = compute_weighted_picks(picks, engine_stats)
        assert len(result) == 1
        # Should NOT be recalibrated — confidence stays at 70
        conf = result[0]["avg_weighted_confidence"]
        assert conf >= 65.0
    finally:
        settings.credibility_recalibration_enabled = old_enabled
        settings.credibility_recalibration_min_picks = old_min
        settings.credibility_strategy_floor_enabled = old_floor_enabled


def test_recalibration_disabled_passes_through():
    """When disabled, confidence is not adjusted."""
    settings = get_settings()
    old_enabled = settings.credibility_recalibration_enabled
    old_floor_enabled = settings.credibility_strategy_floor_enabled
    try:
        settings.credibility_recalibration_enabled = False
        settings.credibility_strategy_floor_enabled = False

        picks = [_make_pick("AAPL", "bad_engine", confidence=80.0)]
        engine_stats = {
            "bad_engine": _make_engine_stats(
                "bad_engine",
                hit_rate=0.05,
                avg_confidence=80.0,
                resolved_picks=50,
            ),
        }

        result = compute_weighted_picks(picks, engine_stats)
        assert len(result) == 1
        # No recalibration — confidence stays at 80
        conf = result[0]["avg_weighted_confidence"]
        assert conf >= 70.0
    finally:
        settings.credibility_recalibration_enabled = old_enabled
        settings.credibility_strategy_floor_enabled = old_floor_enabled
