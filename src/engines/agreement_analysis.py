"""Engine agreement/convergence analysis — measures whether multi-engine
convergence predicts better outcomes."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from itertools import combinations

from sqlalchemy import select

from src.db.models import EnginePickOutcome
from src.db.session import get_session

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceBucket:
    engine_count: int
    total_picks: int
    hits: int
    hit_rate: float
    avg_return_pct: float
    avg_mfe_pct: float
    avg_mae_pct: float


@dataclass
class EnginePairStats:
    engine_a: str
    engine_b: str
    agreement_count: int
    agreement_hit_rate: float
    agreement_avg_return: float


@dataclass
class AgreementReport:
    generated_at: datetime
    lookback_days: int
    total_resolved_picks: int
    by_convergence: list[ConvergenceBucket] = field(default_factory=list)
    engine_pairs: list[EnginePairStats] = field(default_factory=list)


async def compute_agreement_report(
    lookback_days: int = 90,
) -> AgreementReport:
    """Analyze engine agreement vs outcomes over the lookback window."""
    cutoff = date.today() - timedelta(days=lookback_days)

    async with get_session() as session:
        rows = (
            await session.execute(
                select(EnginePickOutcome).where(
                    EnginePickOutcome.outcome_resolved.is_(True),
                    EnginePickOutcome.run_date >= cutoff,
                )
            )
        ).scalars().all()

    if not rows:
        return AgreementReport(
            generated_at=datetime.now(timezone.utc),
            lookback_days=lookback_days,
            total_resolved_picks=0,
        )

    # Group by (run_date, ticker) to find convergence
    groups: dict[tuple[date, str], list[EnginePickOutcome]] = defaultdict(list)
    for row in rows:
        groups[(row.run_date, row.ticker)].append(row)

    # --- Convergence buckets ---
    bucket_data: dict[int, list[EnginePickOutcome]] = defaultdict(list)
    for (rd, ticker), picks in groups.items():
        engine_count = len(set(p.engine_name for p in picks))
        for p in picks:
            bucket_data[engine_count].append(p)

    by_convergence: list[ConvergenceBucket] = []
    for ec in sorted(bucket_data.keys()):
        picks = bucket_data[ec]
        hits = sum(1 for p in picks if p.hit_target)
        returns = [p.actual_return_pct for p in picks if p.actual_return_pct is not None]
        mfes = [p.max_favorable_pct for p in picks if p.max_favorable_pct is not None]
        maes = [p.max_adverse_pct for p in picks if p.max_adverse_pct is not None]
        by_convergence.append(ConvergenceBucket(
            engine_count=ec,
            total_picks=len(picks),
            hits=hits,
            hit_rate=hits / len(picks) if picks else 0.0,
            avg_return_pct=sum(returns) / len(returns) if returns else 0.0,
            avg_mfe_pct=sum(mfes) / len(mfes) if mfes else 0.0,
            avg_mae_pct=sum(maes) / len(maes) if maes else 0.0,
        ))

    # --- Engine pair stats ---
    # For each (run_date, ticker), record which engines picked it
    engine_sets: dict[tuple[date, str], set[str]] = {}
    for (rd, ticker), picks in groups.items():
        engine_sets[(rd, ticker)] = set(p.engine_name for p in picks)

    # Get all unique engine names
    all_engines = sorted(set(p.engine_name for p in rows))
    engine_pairs: list[EnginePairStats] = []

    for ea, eb in combinations(all_engines, 2):
        # Find (run_date, ticker) where both picked
        agreement_keys = [
            k for k, engines in engine_sets.items() if ea in engines and eb in engines
        ]
        if not agreement_keys:
            engine_pairs.append(EnginePairStats(
                engine_a=ea, engine_b=eb,
                agreement_count=0, agreement_hit_rate=0.0, agreement_avg_return=0.0,
            ))
            continue

        # Gather all picks for those agreed keys (from both engines)
        agreed_picks: list[EnginePickOutcome] = []
        for k in agreement_keys:
            agreed_picks.extend(groups[k])

        hits = sum(1 for p in agreed_picks if p.hit_target)
        returns = [p.actual_return_pct for p in agreed_picks if p.actual_return_pct is not None]
        engine_pairs.append(EnginePairStats(
            engine_a=ea,
            engine_b=eb,
            agreement_count=len(agreement_keys),
            agreement_hit_rate=hits / len(agreed_picks) if agreed_picks else 0.0,
            agreement_avg_return=sum(returns) / len(returns) if returns else 0.0,
        ))

    return AgreementReport(
        generated_at=datetime.now(timezone.utc),
        lookback_days=lookback_days,
        total_resolved_picks=len(rows),
        by_convergence=by_convergence,
        engine_pairs=engine_pairs,
    )


def format_agreement_report(report: AgreementReport) -> str:
    """Format an AgreementReport for CLI/Telegram output."""
    lines = [
        "=" * 60,
        "ENGINE AGREEMENT ANALYSIS",
        f"Generated: {report.generated_at:%Y-%m-%d %H:%M} UTC",
        f"Lookback: {report.lookback_days} days | Resolved picks: {report.total_resolved_picks}",
        "=" * 60,
    ]

    if not report.by_convergence:
        lines.append("\nNo resolved picks found in lookback window.")
        return "\n".join(lines)

    lines.append("\nCONVERGENCE BUCKETS:")
    lines.append(f"  {'Engines':>8} {'Picks':>6} {'Hits':>5} {'Hit%':>7} {'AvgRet%':>8} {'AvgMFE%':>8} {'AvgMAE%':>8}")
    lines.append("  " + "-" * 55)
    for b in report.by_convergence:
        lines.append(
            f"  {b.engine_count:>8} {b.total_picks:>6} {b.hits:>5} "
            f"{b.hit_rate:>6.1%} {b.avg_return_pct:>+7.2f} "
            f"{b.avg_mfe_pct:>+7.2f} {b.avg_mae_pct:>+7.2f}"
        )

    if report.engine_pairs:
        lines.append("\nENGINE PAIR AGREEMENT:")
        lines.append(f"  {'Pair':<30} {'Agreed':>7} {'Hit%':>7} {'AvgRet%':>8}")
        lines.append("  " + "-" * 55)
        for p in report.engine_pairs:
            pair_name = f"{p.engine_a} + {p.engine_b}"
            lines.append(
                f"  {pair_name:<30} {p.agreement_count:>7} "
                f"{p.agreement_hit_rate:>6.1%} {p.agreement_avg_return:>+7.2f}"
            )

    return "\n".join(lines)
