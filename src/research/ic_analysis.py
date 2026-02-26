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


@dataclass
class PairwiseCorrelation:
    """Pairwise prediction correlation between two engines."""

    engine_a: str
    engine_b: str
    correlation: float
    p_value: float
    n_overlap: int


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


async def compute_cross_engine_independence(
    session: AsyncSession,
    lookback_days: int = 90,
    asof_date: date | None = None,
) -> list[PairwiseCorrelation]:
    """Compute pairwise prediction correlation on overlapping (run_date, ticker) pairs.

    For each engine pair, finds picks where both engines picked the same ticker
    on the same date, then computes Spearman correlation of their confidences.
    Low correlation = more independent = better for ensemble diversification.
    """
    reference = asof_date or date.today()
    cutoff = reference - timedelta(days=lookback_days)

    result = await session.execute(
        select(EnginePickOutcome).where(
            and_(
                EnginePickOutcome.outcome_resolved == True,  # noqa: E712
                EnginePickOutcome.run_date >= cutoff,
                EnginePickOutcome.run_date <= reference,
            )
        )
    )
    outcomes = result.scalars().all()

    # Index by (run_date, ticker) → {engine: confidence}
    by_key: dict[tuple[date, str], dict[str, float]] = {}
    for o in outcomes:
        key = (o.run_date, o.ticker)
        by_key.setdefault(key, {})[o.engine_name] = o.confidence

    # Collect engine names
    all_engines = sorted({o.engine_name for o in outcomes})

    correlations: list[PairwiseCorrelation] = []
    for i, eng_a in enumerate(all_engines):
        for eng_b in all_engines[i + 1:]:
            # Find overlapping picks
            confs_a: list[float] = []
            confs_b: list[float] = []
            for engines_map in by_key.values():
                if eng_a in engines_map and eng_b in engines_map:
                    confs_a.append(engines_map[eng_a])
                    confs_b.append(engines_map[eng_b])

            n_overlap = len(confs_a)
            if n_overlap < 5:
                correlations.append(PairwiseCorrelation(
                    engine_a=eng_a,
                    engine_b=eng_b,
                    correlation=0.0,
                    p_value=1.0,
                    n_overlap=n_overlap,
                ))
                continue

            if len(set(confs_a)) <= 1 or len(set(confs_b)) <= 1:
                correlations.append(PairwiseCorrelation(
                    engine_a=eng_a,
                    engine_b=eng_b,
                    correlation=0.0,
                    p_value=1.0,
                    n_overlap=n_overlap,
                ))
                continue

            corr, p_val = spearmanr(confs_a, confs_b)
            correlations.append(PairwiseCorrelation(
                engine_a=eng_a,
                engine_b=eng_b,
                correlation=round(float(corr), 4),
                p_value=round(float(p_val), 4),
                n_overlap=n_overlap,
            ))

    return correlations
