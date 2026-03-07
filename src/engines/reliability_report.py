"""Engine reliability report — operational health summary from engine_runs
and historical coverage from external_engine_results."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select

from src.db.models import EngineRun, ExternalEngineResult, EnginePickOutcome
from src.db.session import get_session

logger = logging.getLogger(__name__)

# Known engines for gap detection
KNOWN_ENGINES = ("koocore_d", "gemini_stst", "top3_7d")


@dataclass
class EngineRunStats:
    """Aggregated engine_runs stats for one engine."""

    engine_name: str
    total_runs: int
    successes: int
    failures: int
    success_rate: float
    failure_breakdown: dict[str, int]  # status -> count
    avg_fetch_duration_ms: float | None
    avg_picks_count: float | None
    avg_candidates_screened: float | None
    last_success_date: date | None
    last_failure_date: date | None
    last_error: str | None


@dataclass
class EngineCoverageStats:
    """Historical coverage from external_engine_results."""

    engine_name: str
    total_days_reported: int
    first_report_date: date | None
    last_report_date: date | None
    avg_picks_per_day: float
    missing_dates: list[date]  # trading days with no report in lookback


@dataclass
class EngineHitRateBreakdown:
    """Per-engine hit rate by strategy."""

    engine_name: str
    total_resolved: int
    total_hits: int
    overall_hit_rate: float
    avg_return_pct: float
    by_strategy: dict[str, dict]  # strategy -> {resolved, hits, hit_rate, avg_return}


@dataclass
class ReliabilityReport:
    generated_at: datetime
    lookback_days: int
    run_stats: list[EngineRunStats] = field(default_factory=list)
    coverage_stats: list[EngineCoverageStats] = field(default_factory=list)
    hit_rate_breakdown: list[EngineHitRateBreakdown] = field(default_factory=list)


def _trading_days_in_range(start: date, end: date) -> list[date]:
    """Return weekdays (Mon-Fri) between start and end inclusive."""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon=0 .. Fri=4
            days.append(current)
        current += timedelta(days=1)
    return days


async def _compute_run_stats(cutoff: date) -> list[EngineRunStats]:
    """Query engine_runs for success/failure stats."""
    async with get_session() as session:
        rows = (
            await session.execute(
                select(EngineRun).where(EngineRun.run_date >= cutoff)
            )
        ).scalars().all()

    if not rows:
        return []

    by_engine: dict[str, list[EngineRun]] = defaultdict(list)
    for r in rows:
        by_engine[r.engine_name].append(r)

    stats = []
    for engine_name, runs in sorted(by_engine.items()):
        successes = [r for r in runs if r.status == "success"]
        failures = [r for r in runs if r.status != "success"]

        failure_breakdown: dict[str, int] = defaultdict(int)
        for r in failures:
            failure_breakdown[r.status] += 1

        durations = [r.fetch_duration_ms for r in successes if r.fetch_duration_ms is not None]
        picks = [r.picks_count for r in successes if r.picks_count is not None]
        screened = [r.candidates_screened for r in successes if r.candidates_screened is not None]

        success_dates = [r.run_date for r in successes]
        failure_dates = [r.run_date for r in failures]

        last_error = None
        if failures:
            latest_fail = max(failures, key=lambda r: r.run_date)
            last_error = latest_fail.error_message

        stats.append(EngineRunStats(
            engine_name=engine_name,
            total_runs=len(runs),
            successes=len(successes),
            failures=len(failures),
            success_rate=len(successes) / len(runs) if runs else 0.0,
            failure_breakdown=dict(failure_breakdown),
            avg_fetch_duration_ms=sum(durations) / len(durations) if durations else None,
            avg_picks_count=sum(picks) / len(picks) if picks else None,
            avg_candidates_screened=sum(screened) / len(screened) if screened else None,
            last_success_date=max(success_dates) if success_dates else None,
            last_failure_date=max(failure_dates) if failure_dates else None,
            last_error=last_error,
        ))

    return stats


async def _compute_coverage_stats(cutoff: date) -> list[EngineCoverageStats]:
    """Query external_engine_results for historical coverage."""
    today = date.today()
    trading_days = _trading_days_in_range(cutoff, today)

    async with get_session() as session:
        rows = (
            await session.execute(
                select(ExternalEngineResult).where(
                    ExternalEngineResult.run_date >= cutoff,
                )
            )
        ).scalars().all()

    by_engine: dict[str, list[ExternalEngineResult]] = defaultdict(list)
    for r in rows:
        by_engine[r.engine_name].append(r)

    stats = []
    for engine_name in KNOWN_ENGINES:
        engine_rows = by_engine.get(engine_name, [])
        reported_dates = {r.run_date for r in engine_rows}
        missing = [d for d in trading_days if d not in reported_dates]

        picks_counts = [r.picks_count for r in engine_rows]
        dates = [r.run_date for r in engine_rows]

        stats.append(EngineCoverageStats(
            engine_name=engine_name,
            total_days_reported=len(reported_dates),
            first_report_date=min(dates) if dates else None,
            last_report_date=max(dates) if dates else None,
            avg_picks_per_day=sum(picks_counts) / len(picks_counts) if picks_counts else 0.0,
            missing_dates=sorted(missing)[-10:],  # last 10 missing dates
        ))

    return stats


async def _compute_hit_rate_breakdown(cutoff: date) -> list[EngineHitRateBreakdown]:
    """Per-engine hit rates from engine_pick_outcomes."""
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
        return []

    by_engine: dict[str, list[EnginePickOutcome]] = defaultdict(list)
    for r in rows:
        by_engine[r.engine_name].append(r)

    breakdowns = []
    for engine_name, picks in sorted(by_engine.items()):
        hits = sum(1 for p in picks if p.hit_target)
        returns = [p.actual_return_pct for p in picks if p.actual_return_pct is not None]

        # By strategy
        by_strat: dict[str, list[EnginePickOutcome]] = defaultdict(list)
        for p in picks:
            by_strat[p.strategy].append(p)

        strat_stats = {}
        for strat, strat_picks in sorted(by_strat.items()):
            s_hits = sum(1 for p in strat_picks if p.hit_target)
            s_returns = [p.actual_return_pct for p in strat_picks if p.actual_return_pct is not None]
            strat_stats[strat] = {
                "resolved": len(strat_picks),
                "hits": s_hits,
                "hit_rate": s_hits / len(strat_picks) if strat_picks else 0.0,
                "avg_return": sum(s_returns) / len(s_returns) if s_returns else 0.0,
            }

        breakdowns.append(EngineHitRateBreakdown(
            engine_name=engine_name,
            total_resolved=len(picks),
            total_hits=hits,
            overall_hit_rate=hits / len(picks) if picks else 0.0,
            avg_return_pct=sum(returns) / len(returns) if returns else 0.0,
            by_strategy=strat_stats,
        ))

    return breakdowns


async def compute_reliability_report(lookback_days: int = 90) -> ReliabilityReport:
    """Build a full reliability report across all three data sources."""
    cutoff = date.today() - timedelta(days=lookback_days)

    run_stats = await _compute_run_stats(cutoff)
    coverage_stats = await _compute_coverage_stats(cutoff)
    hit_rate_breakdown = await _compute_hit_rate_breakdown(cutoff)

    return ReliabilityReport(
        generated_at=datetime.now(timezone.utc),
        lookback_days=lookback_days,
        run_stats=run_stats,
        coverage_stats=coverage_stats,
        hit_rate_breakdown=hit_rate_breakdown,
    )


def format_reliability_report(report: ReliabilityReport) -> str:
    """Format a ReliabilityReport for CLI output."""
    lines = [
        "=" * 65,
        "ENGINE RELIABILITY REPORT",
        f"Generated: {report.generated_at:%Y-%m-%d %H:%M} UTC",
        f"Lookback: {report.lookback_days} days",
        "=" * 65,
    ]

    # --- Section 1: Engine Runs (operational) ---
    lines.append("\n--- ENGINE RUNS (operational health) ---")
    if not report.run_stats:
        lines.append("  No engine_runs data yet. Will populate after next pipeline run.")
    else:
        for s in report.run_stats:
            lines.append(f"\n  {s.engine_name}:")
            lines.append(f"    Runs: {s.total_runs}  Success: {s.successes}  Failed: {s.failures}  Rate: {s.success_rate:.0%}")
            if s.failure_breakdown:
                breakdown = ", ".join(f"{k}={v}" for k, v in sorted(s.failure_breakdown.items()))
                lines.append(f"    Failure types: {breakdown}")
            if s.avg_fetch_duration_ms is not None:
                lines.append(f"    Avg batch fetch: {s.avg_fetch_duration_ms:.0f}ms")
            if s.avg_picks_count is not None:
                lines.append(f"    Avg picks: {s.avg_picks_count:.1f}  Avg screened: {s.avg_candidates_screened:.0f}" if s.avg_candidates_screened else f"    Avg picks: {s.avg_picks_count:.1f}")
            if s.last_success_date:
                lines.append(f"    Last success: {s.last_success_date}")
            if s.last_failure_date:
                lines.append(f"    Last failure: {s.last_failure_date}")
                if s.last_error:
                    lines.append(f"    Last error: {s.last_error[:120]}")

    # --- Section 2: Coverage ---
    lines.append("\n--- ENGINE COVERAGE (historical presence) ---")
    if not report.coverage_stats:
        lines.append("  No coverage data.")
    else:
        for c in report.coverage_stats:
            lines.append(f"\n  {c.engine_name}:")
            lines.append(f"    Days reported: {c.total_days_reported}")
            if c.first_report_date and c.last_report_date:
                lines.append(f"    Range: {c.first_report_date} to {c.last_report_date}")
            lines.append(f"    Avg picks/day: {c.avg_picks_per_day:.1f}")
            if c.missing_dates:
                missing_str = ", ".join(str(d) for d in c.missing_dates[-5:])
                suffix = f" (+{len(c.missing_dates) - 5} more)" if len(c.missing_dates) > 5 else ""
                lines.append(f"    Recent gaps: {missing_str}{suffix}")

    # --- Section 3: Hit rates ---
    lines.append("\n--- ENGINE HIT RATES (per-engine performance) ---")
    if not report.hit_rate_breakdown:
        lines.append("  No resolved picks in lookback window.")
    else:
        for h in report.hit_rate_breakdown:
            lines.append(f"\n  {h.engine_name}:")
            lines.append(f"    Resolved: {h.total_resolved}  Hits: {h.total_hits}  Hit rate: {h.overall_hit_rate:.1%}  Avg return: {h.avg_return_pct:+.2f}%")
            if h.by_strategy:
                lines.append("    By strategy:")
                for strat, ss in h.by_strategy.items():
                    lines.append(
                        f"      {strat:<25} n={ss['resolved']:>3}  hit={ss['hit_rate']:>5.1%}  ret={ss['avg_return']:>+6.2f}%"
                    )

    return "\n".join(lines)
