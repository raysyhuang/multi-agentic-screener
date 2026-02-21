"""Credibility Tracker — dynamic engine weighting based on historical accuracy.

Each engine earns its weight through proven track record:
  engine_weight = base_weight * hit_rate_multiplier * calibration_bonus * recency_decay

Convergence multipliers boost conviction when multiple engines agree:
  1 engine: 0.9x (single-source penalty)
  2 engines: 1.3x combined weight
  3+ engines: configurable (default 1.0x)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import select, and_

from src.config import get_settings
from src.db.models import EnginePickOutcome
from src.db.session import get_session

logger = logging.getLogger(__name__)

_DEFAULT_UNKNOWN_ENGINE_WEIGHT = 0.3


@dataclass
class EngineStats:
    """Performance statistics for a single engine."""

    engine_name: str
    total_picks: int = 0
    resolved_picks: int = 0
    hits: int = 0  # picks that hit target
    hit_rate: float = 0.0
    avg_return_pct: float = 0.0
    avg_confidence: float = 0.0
    avg_actual_return: float = 0.0
    brier_score: float = 1.0  # lower is better calibrated
    weight: float = 1.0
    has_enough_data: bool = False
    per_strategy: dict[str, dict] = field(default_factory=dict)


@dataclass
class CredibilitySnapshot:
    """Snapshot of all engine credibility data."""

    engine_stats: dict[str, EngineStats] = field(default_factory=dict)
    avg_hit_rate: float = 0.0
    snapshot_date: date = field(default_factory=date.today)


def _compute_brier_score(picks: list[dict]) -> float:
    """Compute Brier score measuring confidence calibration.

    Brier score = mean((predicted_probability - actual_outcome)^2)
    Lower is better. Perfect calibration = 0, random = 0.25.
    """
    if not picks:
        return 1.0

    total = 0.0
    for p in picks:
        predicted = p["confidence"] / 100.0  # normalize to 0-1
        actual = 1.0 if p["hit_target"] else 0.0
        total += (predicted - actual) ** 2

    return total / len(picks)


async def compute_credibility_snapshot() -> CredibilitySnapshot:
    """Compute current credibility stats for all engines from resolved outcomes."""
    settings = get_settings()
    lookback = settings.credibility_lookback_days
    min_picks = settings.credibility_min_picks_for_weight
    cutoff_date = date.today() - timedelta(days=lookback)

    snapshot = CredibilitySnapshot(snapshot_date=date.today())

    async with get_session() as session:
        # Fetch all resolved outcomes within lookback window
        result = await session.execute(
            select(EnginePickOutcome).where(
                and_(
                    EnginePickOutcome.outcome_resolved == True,  # noqa: E712
                    EnginePickOutcome.run_date >= cutoff_date,
                )
            )
        )
        outcomes = result.scalars().all()

    if not outcomes:
        logger.info("No resolved engine outcomes found — all engines get default weight 1.0")
        return snapshot

    # Group by engine
    by_engine: dict[str, list[dict]] = defaultdict(list)
    for o in outcomes:
        by_engine[o.engine_name].append({
            "ticker": o.ticker,
            "strategy": o.strategy,
            "confidence": o.confidence,
            "hit_target": o.hit_target or False,
            "actual_return_pct": o.actual_return_pct or 0.0,
            "run_date": o.run_date,
            "days_held": o.days_held,
        })

    # Compute per-engine stats
    all_hit_rates: list[float] = []
    for engine_name, picks in by_engine.items():
        stats = EngineStats(engine_name=engine_name)
        stats.total_picks = len(picks)
        stats.resolved_picks = len(picks)
        stats.hits = sum(1 for p in picks if p["hit_target"])
        stats.hit_rate = stats.hits / stats.resolved_picks if stats.resolved_picks > 0 else 0.0
        stats.avg_return_pct = (
            sum(p["actual_return_pct"] for p in picks) / len(picks)
            if picks
            else 0.0
        )
        stats.avg_confidence = (
            sum(p["confidence"] for p in picks) / len(picks)
            if picks
            else 0.0
        )
        stats.brier_score = _compute_brier_score(picks)
        stats.has_enough_data = stats.resolved_picks >= min_picks

        # Per-strategy breakdown
        by_strategy: dict[str, list[dict]] = defaultdict(list)
        for p in picks:
            by_strategy[p["strategy"]].append(p)
        for strat, strat_picks in by_strategy.items():
            strat_hits = sum(1 for p in strat_picks if p["hit_target"])
            stats.per_strategy[strat] = {
                "picks": len(strat_picks),
                "hits": strat_hits,
                "hit_rate": strat_hits / len(strat_picks) if strat_picks else 0.0,
                "avg_return": (
                    sum(p["actual_return_pct"] for p in strat_picks) / len(strat_picks)
                ),
            }

        all_hit_rates.append(stats.hit_rate)
        snapshot.engine_stats[engine_name] = stats

    snapshot.avg_hit_rate = sum(all_hit_rates) / len(all_hit_rates) if all_hit_rates else 0.0

    # Compute dynamic weights
    for stats in snapshot.engine_stats.values():
        stats.weight = _compute_weight(stats, snapshot.avg_hit_rate)

    logger.info(
        "Credibility snapshot: %d engines, avg_hit_rate=%.1f%%",
        len(snapshot.engine_stats), snapshot.avg_hit_rate * 100,
    )
    for name, stats in snapshot.engine_stats.items():
        logger.info(
            "  %s: hits=%d/%d (%.1f%%), weight=%.2f, brier=%.3f, enough_data=%s",
            name, stats.hits, stats.resolved_picks, stats.hit_rate * 100,
            stats.weight, stats.brier_score, stats.has_enough_data,
        )

    return snapshot


def _compute_weight(stats: EngineStats, avg_hit_rate: float) -> float:
    """Compute dynamic weight for an engine.

    weight = base_weight * hit_rate_multiplier * calibration_bonus * data_maturity

    Cold-start engines (< min_picks resolved) get a ramped weight:
    weight scales linearly from 0.1 (0 picks) to 1.0 (min_picks).
    This prevents untested engines from getting equal influence.
    """
    settings = get_settings()
    min_picks = settings.credibility_min_picks_for_weight

    if not stats.has_enough_data:
        # Ramp weight linearly: 0 picks → 0.1, min_picks → 1.0
        if stats.resolved_picks <= 0:
            return 0.1
        ramp = stats.resolved_picks / min_picks  # 0.0 to ~1.0
        return round(max(0.1, min(1.0, ramp)), 3)

    base_weight = 1.0

    # Hit rate multiplier: engine_hit_rate / avg_hit_rate
    if avg_hit_rate > 0:
        hit_rate_multiplier = stats.hit_rate / avg_hit_rate
    else:
        hit_rate_multiplier = 1.0

    # Calibration bonus: reward well-calibrated confidence scores
    calibration_bonus = 1.2 if stats.brier_score < 0.15 else 1.0

    # Clamp weight to reasonable range [0.3, 3.0]
    weight = base_weight * hit_rate_multiplier * calibration_bonus
    weight = max(0.3, min(3.0, weight))

    return round(weight, 3)


def compute_convergence_multiplier(engine_count: int) -> float:
    """Get the conviction multiplier for a given number of agreeing engines."""
    settings = get_settings()
    if engine_count >= 4:
        return settings.convergence_4_engine_multiplier
    elif engine_count == 3:
        return settings.convergence_3_engine_multiplier
    elif engine_count == 2:
        return settings.convergence_2_engine_multiplier
    return settings.convergence_1_engine_multiplier


def _collect_strategy_tags(picks: list[dict]) -> list[str]:
    """Collect all unique strategy tags from picks' metadata.strategies."""
    tags: list[str] = []
    for pick in picks:
        meta = pick.get("metadata") or {}
        pick_tags = meta.get("strategies", [])
        if isinstance(pick_tags, list):
            tags.extend(pick_tags)
    return tags


def _compute_effective_signal_count(
    cross_engine_count: int,
    all_strategy_tags: list[str],
) -> float:
    """Compute effective signal count giving partial credit for same-engine extras.

    Cross-engine agreement counts as full signals. Additional unique strategy
    tags from the same engine (e.g. kc_weekly + kc_pro30 both from KooCore-D)
    count at 0.5 each.
    """
    unique_tags = set(all_strategy_tags)
    independent_count = len(unique_tags)
    same_engine_extras = max(0, independent_count - cross_engine_count)
    return cross_engine_count + (same_engine_extras * 0.5)


def compute_weighted_picks(
    all_picks: list[dict],
    engine_stats: dict[str, EngineStats],
) -> list[dict]:
    """Compute weighted conviction scores for picks across engines.

    Groups picks by ticker, applies engine weights, convergence multipliers,
    and strategy-level convergence scoring.
    Returns sorted list of weighted picks (highest conviction first).
    """
    settings = get_settings()

    # Group by ticker
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for pick in all_picks:
        by_ticker[pick["ticker"]].append(pick)

    weighted_results: list[dict] = []
    for ticker, ticker_picks in by_ticker.items():
        # One vote per engine per ticker: if an engine emits duplicates for the
        # same ticker, keep only its highest-confidence pick to avoid bias.
        by_engine: dict[str, dict] = {}
        for pick in ticker_picks:
            engine_name = pick["engine_name"]
            current = by_engine.get(engine_name)
            if current is None or pick["confidence"] > current["confidence"]:
                by_engine[engine_name] = pick
        deduped_picks = list(by_engine.values())

        engine_count = len(deduped_picks)
        convergence_mult = compute_convergence_multiplier(engine_count)

        # Collect strategy tags across all picks for this ticker
        all_strategy_tags = _collect_strategy_tags(ticker_picks)
        independent_signal_count = len(set(all_strategy_tags))
        effective_signal_count = _compute_effective_signal_count(
            engine_count, all_strategy_tags,
        )

        # Compute weighted confidence
        total_weighted_conf = 0.0
        total_weight = 0.0
        engines_agreeing: list[str] = []
        strategies: list[str] = []

        for pick in deduped_picks:
            engine_name = pick["engine_name"]
            stats = engine_stats.get(engine_name)
            if stats:
                weight = stats.weight
            elif engine_name == "multi_agentic_screener":
                # Preserve baseline influence for the host engine.
                weight = 1.0
            else:
                # Unknown external engines should not get full influence.
                weight = _DEFAULT_UNKNOWN_ENGINE_WEIGHT

            total_weighted_conf += pick["confidence"] * weight
            total_weight += weight
            engines_agreeing.append(engine_name)
            strategies.append(pick.get("strategy", "unknown"))

        avg_weighted_conf = total_weighted_conf / total_weight if total_weight > 0 else 0
        combined_score = avg_weighted_conf * convergence_mult

        # Aggregate best entry/stop/target from highest-weighted engine
        best_pick = max(
            deduped_picks,
            key=lambda p: (
                engine_stats[p["engine_name"]].weight
                if p["engine_name"] in engine_stats
                else (1.0 if p["engine_name"] == "multi_agentic_screener"
                      else _DEFAULT_UNKNOWN_ENGINE_WEIGHT)
            ) * p["confidence"],
        )

        weighted_results.append({
            "ticker": ticker,
            "combined_score": round(combined_score, 2),
            "avg_weighted_confidence": round(avg_weighted_conf, 2),
            "convergence_multiplier": convergence_mult,
            "engine_count": engine_count,
            "engines": engines_agreeing,
            "strategies": strategies,
            "strategy_tags": sorted(set(all_strategy_tags)),
            "independent_signal_count": independent_signal_count,
            "effective_signal_count": round(effective_signal_count, 1),
            "entry_price": best_pick.get("entry_price"),
            "stop_loss": best_pick.get("stop_loss"),
            "target_price": best_pick.get("target_price"),
            "holding_period_days": best_pick.get("holding_period_days"),
            "thesis": best_pick.get("thesis"),
            "risk_factors": best_pick.get("risk_factors", []),
        })

    # Sort by combined score descending
    weighted_results.sort(key=lambda x: x["combined_score"], reverse=True)
    return weighted_results
