"""Tests for strategy-level hit-rate floor in credibility system."""

from src.config import get_settings
from src.engines.credibility import EngineStats, compute_weighted_picks


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
    per_strategy: dict[str, dict] | None = None,
    weight: float = 1.0,
) -> EngineStats:
    stats = EngineStats(engine_name=name)
    stats.weight = weight
    stats.has_enough_data = True
    if per_strategy:
        stats.per_strategy = per_strategy
    return stats


def test_strategy_floor_drops_low_hit_rate_picks():
    """Picks from strategies below the hit-rate floor are excluded."""
    settings = get_settings()
    old_enabled = settings.credibility_strategy_floor_enabled
    old_rate = settings.credibility_strategy_floor_hit_rate
    old_min = settings.credibility_strategy_floor_min_picks
    try:
        settings.credibility_strategy_floor_enabled = True
        settings.credibility_strategy_floor_hit_rate = 0.20
        settings.credibility_strategy_floor_min_picks = 5

        picks = [
            _make_pick("AAPL", "gemini_stst", "momentum", 60),
            _make_pick("GOOG", "gemini_stst", "mean_reversion", 70),
            _make_pick("MSFT", "koocore_d", "swing", 65),
        ]
        engine_stats = {
            "gemini_stst": _make_engine_stats("gemini_stst", per_strategy={
                "momentum": {"picks": 6, "hits": 0, "hit_rate": 0.0, "avg_return": -3.0},
                "mean_reversion": {"picks": 15, "hits": 3, "hit_rate": 0.20, "avg_return": -0.8},
            }),
            "koocore_d": _make_engine_stats("koocore_d", per_strategy={
                "swing": {"picks": 2, "hits": 1, "hit_rate": 0.50, "avg_return": 6.0},
            }),
        }

        result = compute_weighted_picks(picks, engine_stats)
        tickers = [r["ticker"] for r in result]

        # momentum at 0% (6 picks >= 5 min) should be dropped
        assert "AAPL" not in tickers
        # mean_reversion at 20% (== floor) should survive (not strictly less)
        assert "GOOG" in tickers
        # swing at 50% (only 2 picks < 5 min) should survive (insufficient data)
        assert "MSFT" in tickers
    finally:
        settings.credibility_strategy_floor_enabled = old_enabled
        settings.credibility_strategy_floor_hit_rate = old_rate
        settings.credibility_strategy_floor_min_picks = old_min


def test_strategy_floor_disabled_passes_all():
    """When disabled, no picks are dropped regardless of hit rate."""
    settings = get_settings()
    old_enabled = settings.credibility_strategy_floor_enabled
    try:
        settings.credibility_strategy_floor_enabled = False

        picks = [
            _make_pick("AAPL", "gemini_stst", "momentum", 60),
        ]
        engine_stats = {
            "gemini_stst": _make_engine_stats("gemini_stst", per_strategy={
                "momentum": {"picks": 10, "hits": 0, "hit_rate": 0.0, "avg_return": -5.0},
            }),
        }

        result = compute_weighted_picks(picks, engine_stats)
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
    finally:
        settings.credibility_strategy_floor_enabled = old_enabled


def test_strategy_floor_respects_min_picks_threshold():
    """Strategies with fewer than min_picks resolved are not subject to the floor."""
    settings = get_settings()
    old_enabled = settings.credibility_strategy_floor_enabled
    old_rate = settings.credibility_strategy_floor_hit_rate
    old_min = settings.credibility_strategy_floor_min_picks
    try:
        settings.credibility_strategy_floor_enabled = True
        settings.credibility_strategy_floor_hit_rate = 0.25
        settings.credibility_strategy_floor_min_picks = 10

        picks = [
            _make_pick("AAPL", "gemini_stst", "momentum", 60),
        ]
        engine_stats = {
            "gemini_stst": _make_engine_stats("gemini_stst", per_strategy={
                # 0% hit rate but only 4 picks — below min_picks threshold
                "momentum": {"picks": 4, "hits": 0, "hit_rate": 0.0, "avg_return": -5.0},
            }),
        }

        result = compute_weighted_picks(picks, engine_stats)
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
    finally:
        settings.credibility_strategy_floor_enabled = old_enabled
        settings.credibility_strategy_floor_hit_rate = old_rate
        settings.credibility_strategy_floor_min_picks = old_min


def test_strategy_floor_unknown_engine_passes():
    """Picks from engines not in engine_stats (unknown) pass through."""
    settings = get_settings()
    old_enabled = settings.credibility_strategy_floor_enabled
    try:
        settings.credibility_strategy_floor_enabled = True

        picks = [
            _make_pick("AAPL", "new_engine", "some_strategy", 60),
        ]
        # No stats for new_engine
        engine_stats = {}

        result = compute_weighted_picks(picks, engine_stats)
        assert len(result) == 1
    finally:
        settings.credibility_strategy_floor_enabled = old_enabled
