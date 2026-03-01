"""Shadow Track Leaderboard — ranks tracks by composite performance score.

Composite = 0.30*sharpe + 0.20*profit_factor + 0.20*win_rate
          + 0.15*calmar + 0.15*avg_return  (min-max normalized)

Deflated Sharpe Ratio (Bailey & López de Prado) penalizes for multiple
testing — critical since we run 20+ tracks simultaneously.

Tracks with < 5 resolved picks are flagged as 'insufficient data'.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import select, and_, func

from src.backtest.metrics import compute_metrics, deflated_sharpe_ratio
from src.db.models import ShadowTrack, ShadowTrackPick
from src.db.session import get_session

logger = logging.getLogger(__name__)

MIN_RESOLVED_FOR_RANKING = 5


@dataclass
class TrackScorecard:
    """Performance scorecard for a single shadow track."""

    name: str
    track_id: int
    status: str
    generation: int
    parent_track: str | None
    config: dict
    description: str | None = None

    # Metrics
    total_picks: int = 0
    resolved_picks: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0

    # DSR — probability the Sharpe is real after correcting for multiple testing
    deflated_sharpe: float = 0.0

    # Composite
    composite_score: float = 0.0
    has_sufficient_data: bool = False

    # Delta vs baseline
    delta_sharpe: float = 0.0
    delta_win_rate: float = 0.0
    delta_avg_return: float = 0.0


async def compute_leaderboard(lookback_days: int = 14) -> list[TrackScorecard]:
    """Rank active tracks by composite score vs baseline.

    Args:
        lookback_days: Only consider picks from the last N days.

    Returns:
        List of TrackScorecard sorted by composite_score descending.
    """
    cutoff = date.today() - timedelta(days=lookback_days)
    scorecards: list[TrackScorecard] = []

    async with get_session() as session:
        # Get all tracks (not just active — include eliminated for history)
        tracks_result = await session.execute(select(ShadowTrack))
        tracks = tracks_result.scalars().all()

        for track in tracks:
            # Get resolved picks within lookback window
            picks_result = await session.execute(
                select(ShadowTrackPick).where(
                    and_(
                        ShadowTrackPick.track_id == track.id,
                        ShadowTrackPick.outcome_resolved == True,  # noqa: E712
                        ShadowTrackPick.run_date >= cutoff,
                    )
                )
            )
            resolved = picks_result.scalars().all()

            # Total picks (including unresolved)
            total_result = await session.execute(
                select(func.count(ShadowTrackPick.id)).where(
                    and_(
                        ShadowTrackPick.track_id == track.id,
                        ShadowTrackPick.run_date >= cutoff,
                    )
                )
            )
            total_picks = total_result.scalar() or 0

            sc = TrackScorecard(
                name=track.name,
                track_id=track.id,
                status=track.status,
                generation=track.generation,
                parent_track=track.parent_track,
                config=track.config,
                description=track.description,
                total_picks=total_picks,
                resolved_picks=len(resolved),
                has_sufficient_data=len(resolved) >= MIN_RESOLVED_FOR_RANKING,
            )

            if resolved:
                returns = [
                    p.actual_return for p in resolved if p.actual_return is not None
                ]
                if returns:
                    metrics = compute_metrics(returns)
                    sc.win_rate = metrics.win_rate
                    sc.avg_return_pct = metrics.avg_return_pct
                    sc.total_return_pct = metrics.total_return_pct
                    sc.sharpe_ratio = metrics.sharpe_ratio
                    sc.sortino_ratio = metrics.sortino_ratio
                    sc.calmar_ratio = metrics.calmar_ratio
                    sc.profit_factor = metrics.profit_factor
                    sc.max_drawdown_pct = metrics.max_drawdown_pct
                    sc._returns = returns  # stash for DSR computation

            scorecards.append(sc)

    # Count total active tracks for DSR multiple-testing correction
    num_tracks = max(1, len([s for s in scorecards if s.status == "active"]))

    # Compute Deflated Sharpe Ratio for each track
    for sc in scorecards:
        returns = getattr(sc, "_returns", None)
        if returns and sc.sharpe_ratio > 0 and len(returns) >= 10:
            sc.deflated_sharpe = deflated_sharpe_ratio(
                observed_sharpe=sc.sharpe_ratio,
                num_trials=num_tracks,
                returns=returns,
            )
        if hasattr(sc, "_returns"):
            del sc._returns

    # Compute composite scores (min-max normalized)
    _compute_composite_scores(scorecards)

    # Sort by composite score descending
    scorecards.sort(key=lambda s: s.composite_score, reverse=True)

    # Compute deltas vs baseline (first _baseline track or lowest generation)
    _compute_baseline_deltas(scorecards)

    return scorecards


def _compute_composite_scores(scorecards: list[TrackScorecard]) -> None:
    """Compute min-max normalized composite scores.

    Composite = 0.30*sharpe + 0.20*profit_factor + 0.20*win_rate
              + 0.15*calmar + 0.15*avg_return
    """
    eligible = [s for s in scorecards if s.has_sufficient_data]
    if not eligible:
        return

    # Weights
    weights = {
        "sharpe": 0.30,
        "profit_factor": 0.20,
        "win_rate": 0.20,
        "calmar": 0.15,
        "avg_return": 0.15,
    }

    # Extract raw values
    metrics_map = {
        "sharpe": [s.sharpe_ratio for s in eligible],
        "profit_factor": [s.profit_factor for s in eligible],
        "win_rate": [s.win_rate for s in eligible],
        "calmar": [s.calmar_ratio for s in eligible],
        "avg_return": [s.avg_return_pct for s in eligible],
    }

    # Min-max normalize each metric
    normalized: dict[str, list[float]] = {}
    for metric_name, values in metrics_map.items():
        mn, mx = min(values), max(values)
        rng = mx - mn
        if rng == 0:
            normalized[metric_name] = [0.5] * len(values)
        else:
            normalized[metric_name] = [(v - mn) / rng for v in values]

    # Compute weighted composite with DSR penalty
    for i, sc in enumerate(eligible):
        raw_score = sum(
            weights[m] * normalized[m][i]
            for m in weights
        )
        # DSR penalty: if DSR < 0.50, the Sharpe is likely noise — scale down
        # DSR >= 0.95 = no penalty; DSR ~0.50 = 25% penalty; DSR ~0 = 50% penalty
        dsr_factor = 0.5 + 0.5 * min(1.0, sc.deflated_sharpe / 0.95) if sc.deflated_sharpe > 0 else 1.0
        sc.composite_score = round(raw_score * dsr_factor, 4)


def _compute_baseline_deltas(scorecards: list[TrackScorecard]) -> None:
    """Compute delta-vs-baseline for each track.

    Baseline is the track with name '_baseline' or the first generation-1 track.
    """
    baseline = None
    for sc in scorecards:
        if sc.name == "_baseline":
            baseline = sc
            break

    if baseline is None:
        # Use first gen-1 track as proxy
        for sc in scorecards:
            if sc.generation == 1:
                baseline = sc
                break

    if baseline is None:
        return

    for sc in scorecards:
        sc.delta_sharpe = round(sc.sharpe_ratio - baseline.sharpe_ratio, 4)
        sc.delta_win_rate = round(sc.win_rate - baseline.win_rate, 4)
        sc.delta_avg_return = round(sc.avg_return_pct - baseline.avg_return_pct, 4)
