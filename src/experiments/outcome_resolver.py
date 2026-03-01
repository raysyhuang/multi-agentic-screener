"""Shadow Track Outcome Resolver — resolves paper picks and recomputes snapshots.

Adapts the existing outcome_resolver pattern from src/engines/outcome_resolver.py.
Reuses _fetch_prices_batch and _compute_pick_outcome for actual trade simulation.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.backtest.metrics import compute_metrics
from src.db.models import ShadowTrack, ShadowTrackPick, ShadowTrackSnapshot
from src.db.session import get_session
from src.engines.outcome_resolver import _fetch_prices_batch, _compute_pick_outcome

logger = logging.getLogger(__name__)


async def resolve_shadow_track_outcomes() -> int:
    """Resolve paper picks whose holding period has expired.

    Returns the number of picks resolved.
    """
    today = date.today()
    resolved_count = 0

    async with get_session() as session:
        # Find unresolved picks where holding period has elapsed
        result = await session.execute(
            select(ShadowTrackPick).where(
                ShadowTrackPick.outcome_resolved == False  # noqa: E712
            )
        )
        unresolved = result.scalars().all()

        if not unresolved:
            logger.info("Shadow tracks: no unresolved picks to process")
            return 0

        # Filter to picks that are due for resolution
        due_picks = []
        for pick in unresolved:
            due_date = pick.run_date + timedelta(days=pick.holding_period + 1)
            if today >= due_date:
                due_picks.append(pick)

        if not due_picks:
            logger.info(
                "Shadow tracks: %d unresolved picks, none due yet", len(unresolved)
            )
            return 0

        logger.info("Shadow tracks: resolving %d due picks", len(due_picks))

        # Batch-fetch prices
        tickers = list(set(p.ticker for p in due_picks))
        price_data = _fetch_prices_batch(tickers)

        no_price_count = 0
        insufficient_bars = 0

        for pick in due_picks:
            prices = price_data.get(pick.ticker)
            if not prices:
                no_price_count += 1
                continue

            # Check for enough trading bars
            trading_bars = [p for p in prices if p["date"] > pick.run_date]
            if len(trading_bars) < pick.holding_period:
                insufficient_bars += 1
                continue

            # Compute outcome
            resolution = _compute_pick_outcome(
                ticker=pick.ticker,
                entry_price=pick.entry_price,
                target_price=pick.target_price,
                stop_loss=pick.stop_loss,
                entry_date=pick.run_date,
                hold_days=pick.holding_period,
                prices=prices,
            )

            # Update pick in-session
            pick.outcome_resolved = True
            pick.actual_return = resolution["actual_return_pct"]
            pick.exit_reason = resolution["exit_reason"]
            pick.days_held = resolution["days_held"]
            pick.max_favorable = resolution["max_favorable_pct"]
            pick.max_adverse = resolution["max_adverse_pct"]
            if resolution["days_held"] and pick.run_date:
                pick.exit_date = pick.run_date + timedelta(days=resolution["days_held"])

            resolved_count += 1

        await session.commit()

    logger.info(
        "Shadow tracks resolved: %d picks (no_price=%d, insufficient_bars=%d)",
        resolved_count, no_price_count, insufficient_bars,
    )

    # Recompute snapshots for affected tracks
    if resolved_count > 0:
        await recompute_snapshots()

    return resolved_count


async def recompute_snapshots() -> None:
    """Recompute cumulative metrics for all active tracks with resolved picks."""
    today = date.today()

    async with get_session() as session:
        # Get all active tracks
        tracks_result = await session.execute(
            select(ShadowTrack).where(ShadowTrack.status == "active")
        )
        tracks = tracks_result.scalars().all()

        for track in tracks:
            # Get all resolved picks for this track
            picks_result = await session.execute(
                select(ShadowTrackPick).where(
                    and_(
                        ShadowTrackPick.track_id == track.id,
                        ShadowTrackPick.outcome_resolved == True,  # noqa: E712
                    )
                )
            )
            resolved_picks = picks_result.scalars().all()

            # Get total pick count
            total_result = await session.execute(
                select(ShadowTrackPick).where(ShadowTrackPick.track_id == track.id)
            )
            total_picks = len(total_result.scalars().all())

            if not resolved_picks:
                continue

            # Compute metrics from resolved returns
            returns = [p.actual_return for p in resolved_picks if p.actual_return is not None]
            if not returns:
                continue

            metrics = compute_metrics(returns)

            # Upsert snapshot
            stmt = pg_insert(ShadowTrackSnapshot).values(
                track_id=track.id,
                snapshot_date=today,
                total_picks=total_picks,
                resolved_picks=len(resolved_picks),
                win_rate=metrics.win_rate,
                avg_return_pct=metrics.avg_return_pct,
                total_return=metrics.total_return_pct,
                sharpe_ratio=metrics.sharpe_ratio,
                profit_factor=metrics.profit_factor,
                max_drawdown=metrics.max_drawdown_pct,
            ).on_conflict_do_update(
                constraint="uq_shadow_snapshot_track_date",
                set_={
                    "total_picks": total_picks,
                    "resolved_picks": len(resolved_picks),
                    "win_rate": metrics.win_rate,
                    "avg_return_pct": metrics.avg_return_pct,
                    "total_return": metrics.total_return_pct,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "profit_factor": metrics.profit_factor,
                    "max_drawdown": metrics.max_drawdown_pct,
                },
            )
            await session.execute(stmt)

            logger.info(
                "Shadow track '%s' snapshot: %d/%d resolved, wr=%.1f%%, "
                "sharpe=%.2f, total_return=%.2f%%",
                track.name, len(resolved_picks), total_picks,
                metrics.win_rate * 100, metrics.sharpe_ratio,
                metrics.total_return_pct,
            )
