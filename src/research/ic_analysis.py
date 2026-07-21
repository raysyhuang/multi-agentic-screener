"""IC (Information Coefficient) analysis for engine confidence calibration.

IC = Spearman rank correlation between confidence and actual returns.
A high hit_rate engine with random confidence ordering is less useful
than one where high-confidence picks consistently outperform low-confidence ones.

This is an offline analysis tool, not a pipeline step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from scipy.stats import spearmanr
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import EnginePickOutcome

logger = logging.getLogger(__name__)


@dataclass
class ICResult:
    """Result of an IC computation."""

    ic: float  # Spearman rank correlation coefficient
    p_value: float
    n: int  # number of observations
    engine_name: str = ""


@dataclass
class EngineICReport:
    """Full IC report for a single engine."""

    engine_name: str
    ic: float
    p_value: float
    n: int
    hit_rate: float
    brier_score: float


# ── Core Functions ───────────────────────────────────────────────────────────


def compute_ic(
    confidences: list[float],
    returns: list[float],
) -> ICResult:
    """Compute Spearman rank IC between confidence scores and actual returns.

    Returns ICResult with ic=0, p_value=1 for degenerate inputs
    (< 5 observations, zero variance).
    """
    n = len(confidences)
    if n != len(returns):
        raise ValueError(
            f"Length mismatch: confidences={n}, returns={len(returns)}"
        )

    if n < 5:
        return ICResult(ic=0.0, p_value=1.0, n=n)

    # Zero variance guard — spearmanr returns NaN for constant input
    if len(set(confidences)) <= 1 or len(set(returns)) <= 1:
        return ICResult(ic=0.0, p_value=1.0, n=n)

    corr, p_val = spearmanr(confidences, returns)
    return ICResult(
        ic=round(float(corr), 4),
        p_value=round(float(p_val), 4),
        n=n,
    )


def _compute_brier(confidences: list[float], hit_targets: list[bool]) -> float:
    """Brier score from confidence (0-100) and binary outcomes."""
    if not confidences:
        return 1.0
    total = 0.0
    for conf, hit in zip(confidences, hit_targets):
        pred = conf / 100.0
        actual = 1.0 if hit else 0.0
        total += (pred - actual) ** 2
    return round(total / len(confidences), 4)


async def compute_engine_ic(
    engine_name: str,
    session: AsyncSession,
    lookback_days: int = 90,
    asof_date: date | None = None,
) -> EngineICReport:
    """Compute IC for a single engine from resolved outcomes in the DB.

    Args:
        engine_name: Engine to analyze.
        session: Async SQLAlchemy session.
        lookback_days: How far back to look for resolved outcomes.
        asof_date: Reference date (defaults to today).
    """
    reference = asof_date or date.today()
    cutoff = reference - timedelta(days=lookback_days)

    result = await session.execute(
        select(EnginePickOutcome).where(
            and_(
                EnginePickOutcome.engine_name == engine_name,
                EnginePickOutcome.outcome_resolved == True,  # noqa: E712
                EnginePickOutcome.run_date >= cutoff,
                EnginePickOutcome.run_date <= reference,
            )
        )
    )
    outcomes = result.scalars().all()

    confidences = [o.confidence for o in outcomes]
    returns = [o.actual_return_pct or 0.0 for o in outcomes]
    hit_targets = [o.hit_target or False for o in outcomes]

    ic_result = compute_ic(confidences, returns)
    hit_count = sum(1 for h in hit_targets if h)

    return EngineICReport(
        engine_name=engine_name,
        ic=ic_result.ic,
        p_value=ic_result.p_value,
        n=ic_result.n,
        hit_rate=round(hit_count / len(hit_targets), 4) if hit_targets else 0.0,
        brier_score=_compute_brier(confidences, hit_targets),
    )

