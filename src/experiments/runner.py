"""Shadow Track Runner — executes parameter variations in parallel with production.

Shares all expensive data (engine picks, credibility snapshots, regime context).
Only varies the cheap math stages: scoring, synthesis weighting, guardian sizing.
All shadow track picks are paper-only.
"""

from __future__ import annotations

import copy
import logging
import time
from datetime import date

from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config import get_settings
from src.db.models import ShadowTrack, ShadowTrackPick
from src.db.session import get_session
from src.engines.credibility import (
    EngineStats,
    compute_weighted_picks,
)
from src.experiments.config import sync_tracks_to_db

logger = logging.getLogger(__name__)


async def run_shadow_tracks(
    all_picks: list[dict],
    cred_snapshot_stats: dict[str, EngineStats],
    regime_context: dict,
    run_date: date | None = None,
) -> dict[str, list[dict]]:
    """Run all active shadow tracks against today's engine picks.

    Args:
        all_picks: Flattened engine picks (already fetched in Step 10).
        cred_snapshot_stats: Engine credibility stats (already computed).
        regime_context: Regime from Step 1 (regime, breadth_score, etc.).
        run_date: Override run date (defaults to today).

    Returns:
        Dict of track_name -> list of weighted pick dicts.
    """
    today = run_date or date.today()
    settings = get_settings()
    t0 = time.monotonic()

    if not all_picks:
        logger.info("Shadow tracks: no picks to process, skipping")
        return {}

    # Sync YAML definitions to DB, get active tracks
    active_tracks = await sync_tracks_to_db()
    if not active_tracks:
        logger.info("Shadow tracks: no active tracks configured")
        return {}

    logger.info(
        "Shadow tracks: running %d tracks against %d picks",
        len(active_tracks), len(all_picks),
    )

    results: dict[str, list[dict]] = {}

    # Always run baseline (no overrides) as control group
    baseline_picks = compute_weighted_picks(all_picks, cred_snapshot_stats)
    results["_baseline"] = baseline_picks

    # Persist baseline picks — ensures leaderboard deltas are computed from real data
    baseline_track = await _ensure_baseline_track()
    if baseline_track:
        await _save_track_picks(
            track_id=baseline_track.id,
            track_name="_baseline",
            picks=baseline_picks,
            regime_context=regime_context,
            run_date=today,
        )

    for track in active_tracks:
        try:
            track_picks = _run_single_track(
                track=track,
                all_picks=all_picks,
                cred_snapshot_stats=cred_snapshot_stats,
                regime_context=regime_context,
                settings=settings,
            )
            results[track.name] = track_picks

            # Persist picks to DB
            await _save_track_picks(
                track_id=track.id,
                track_name=track.name,
                picks=track_picks,
                regime_context=regime_context,
                run_date=today,
            )
        except Exception as e:
            logger.error("Shadow track '%s' failed: %s", track.name, e, exc_info=True)
            results[track.name] = []

    elapsed = time.monotonic() - t0
    logger.info(
        "Shadow tracks complete: %d tracks in %.1fs, total picks=%d",
        len(active_tracks), elapsed,
        sum(len(v) for v in results.values()),
    )

    return results


def _run_single_track(
    track: ShadowTrack,
    all_picks: list[dict],
    cred_snapshot_stats: dict[str, EngineStats],
    regime_context: dict,
    settings,
) -> list[dict]:
    """Execute a single shadow track's parameter variation.

    Deep-copies picks and applies the track's config overrides to
    the weighting/scoring functions.
    """
    overrides = track.config or {}

    # Run weighted picks with track-specific convergence multipliers
    weighted = compute_weighted_picks(
        all_picks,
        cred_snapshot_stats,
        config_overrides=overrides,
    )

    # Apply min_confidence filter if overridden
    min_conf = overrides.get("min_confidence")
    if min_conf is not None:
        before = len(weighted)
        weighted = [p for p in weighted if p.get("avg_weighted_confidence", 0) >= min_conf]
        if len(weighted) < before:
            logger.debug(
                "Track '%s': min_confidence=%.1f filtered %d→%d picks",
                track.name, min_conf, before, len(weighted),
            )

    # Apply guardian overrides for position sizing
    guardian_overrides = {
        k: v for k, v in overrides.items()
        if k.startswith("guardian_")
    }
    if guardian_overrides:
        weighted = _apply_guardian_sizing(
            weighted, guardian_overrides, regime_context, settings,
        )

    return weighted


def _apply_guardian_sizing(
    picks: list[dict],
    guardian_overrides: dict,
    regime_context: dict,
    settings,
) -> list[dict]:
    """Apply guardian-style sizing adjustments to shadow track picks.

    Simplified version of the full Capital Guardian — applies regime
    sizing and max portfolio heat overrides.
    """
    regime = regime_context.get("regime", "bull").lower()

    bear_sizing = guardian_overrides.get("guardian_bear_sizing", settings.guardian_bear_sizing)
    choppy_sizing = guardian_overrides.get("guardian_choppy_sizing", settings.guardian_choppy_sizing)
    max_heat = guardian_overrides.get(
        "guardian_max_portfolio_heat_pct", settings.guardian_max_portfolio_heat_pct
    )

    regime_factor = {
        "bull": 1.0,
        "bear": bear_sizing,
        "choppy": choppy_sizing,
    }.get(regime, 0.75)

    # Apply regime scaling to combined scores as proxy for weight
    for pick in picks:
        pick["weight_pct"] = round(
            pick.get("combined_score", 50.0) / 100.0 * regime_factor * 5.0, 2
        )

    # Cap total weight at max portfolio heat
    total_weight = sum(p.get("weight_pct", 0) for p in picks)
    if total_weight > max_heat and total_weight > 0:
        scale = max_heat / total_weight
        for pick in picks:
            pick["weight_pct"] = round(pick.get("weight_pct", 0) * scale, 2)

    return picks


async def _ensure_baseline_track() -> ShadowTrack | None:
    """Ensure a '_baseline' track row exists in the DB (generation=0, no overrides)."""
    async with get_session() as session:
        result = await session.execute(
            select(ShadowTrack).where(ShadowTrack.name == "_baseline")
        )
        existing = result.scalar_one_or_none()
        if existing:
            return existing

        baseline = ShadowTrack(
            name="_baseline",
            generation=0,
            parent_track=None,
            status="active",
            config={},
            description="Production baseline — no parameter overrides. Control group.",
        )
        session.add(baseline)
        await session.flush()
        logger.info("Created _baseline track (id=%d)", baseline.id)
        return baseline


async def _save_track_picks(
    track_id: int,
    track_name: str,
    picks: list[dict],
    regime_context: dict,
    run_date: date,
) -> None:
    """Persist shadow track picks to the database.

    Uses INSERT ... ON CONFLICT DO NOTHING to handle re-runs gracefully.
    """
    if not picks:
        return

    async with get_session() as session:
        for pick in picks:
            ticker = pick.get("ticker", "")
            if not ticker:
                continue

            engines = pick.get("engines", [])
            stmt = pg_insert(ShadowTrackPick).values(
                track_id=track_id,
                run_date=run_date,
                ticker=ticker,
                direction="LONG",  # default; cross-engine synthesis is long-biased
                strategy=pick.get("strategies", ["unknown"])[0] if pick.get("strategies") else "unknown",
                entry_price=pick.get("entry_price", 0.0) or 0.0,
                stop_loss=pick.get("stop_loss"),
                target_price=pick.get("target_price"),
                confidence=pick.get("avg_weighted_confidence", 0.0),
                holding_period=pick.get("holding_period_days", 5) or 5,
                weight_pct=pick.get("weight_pct"),
                source_engines=",".join(engines) if engines else None,
            ).on_conflict_do_nothing(
                constraint="uq_shadow_pick_track_date_ticker"
            )
            await session.execute(stmt)

    logger.debug(
        "Shadow track '%s': saved %d picks for %s",
        track_name, len(picks), run_date,
    )
