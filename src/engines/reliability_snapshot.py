"""Minimal engine reliability snapshot derived from engine_runs."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from sqlalchemy import select

from src.db.models import EngineRun
from src.db.session import get_session
from src.utils.trading_calendar import trading_days_between

KNOWN_ENGINES = ("koocore_d", "gemini_stst", "top3_7d")


def _reason_from_error_message(error_message: str | None, status: str) -> str:
    if status == "success":
        return "success"
    if not error_message:
        return "unknown"
    prefix = str(error_message).split(":", 1)[0].strip().lower()
    return prefix or "unknown"


def _latest_attempt_per_day(rows: list[EngineRun]) -> list[EngineRun]:
    by_day: dict[date, EngineRun] = {}
    for row in rows:
        existing = by_day.get(row.run_date)
        if existing is None or row.attempt > existing.attempt:
            by_day[row.run_date] = row
    return sorted(by_day.values(), key=lambda r: r.run_date, reverse=True)


def _window_success_rate(rows: list[EngineRun], asof: date, window_days: int) -> tuple[int, int, float | None]:
    cutoff = asof - timedelta(days=window_days)
    in_window = [r for r in rows if r.run_date >= cutoff]
    total = len(in_window)
    if total == 0:
        return 0, 0, None
    successes = sum(1 for r in in_window if r.status == "success")
    return successes, total, successes / total


async def get_engine_reliability_snapshot(
    *,
    days: int = 30,
    asof_date: date | None = None,
) -> dict[str, Any]:
    """Return per-engine reliability summary for dashboard display."""
    asof = asof_date or date.today()
    cutoff = asof - timedelta(days=max(1, days))

    async with get_session() as session:
        rows = (
            await session.execute(
                select(EngineRun).where(EngineRun.run_date >= cutoff)
            )
        ).scalars().all()

    by_engine: dict[str, list[EngineRun]] = defaultdict(list)
    for row in rows:
        by_engine[row.engine_name].append(row)

    engine_names = sorted(set(KNOWN_ENGINES) | set(by_engine.keys()))
    engines: list[dict[str, Any]] = []
    for engine_name in engine_names:
        engine_rows = _latest_attempt_per_day(by_engine.get(engine_name, []))
        latest = engine_rows[0] if engine_rows else None

        last_success_date = next((r.run_date for r in engine_rows if r.status == "success"), None)

        consecutive_failures = 0
        for row in engine_rows:
            if row.status == "success":
                break
            consecutive_failures += 1

        s7, t7, rate7 = _window_success_rate(engine_rows, asof, 7)
        s30, t30, rate30 = _window_success_rate(engine_rows, asof, 30)

        latest_status = latest.status if latest else "no_data"
        latest_reason = _reason_from_error_message(latest.error_message if latest else None, latest_status)
        last_success_age_trading_days = (
            trading_days_between(last_success_date, asof)
            if last_success_date is not None
            else None
        )

        engines.append({
            "engine_name": engine_name,
            "latest_status": latest_status,
            "latest_reason": latest_reason,
            "latest_run_date": str(latest.run_date) if latest else None,
            "last_success_date": str(last_success_date) if last_success_date else None,
            "last_success_age_trading_days": last_success_age_trading_days,
            "consecutive_failures": consecutive_failures,
            "success_rate_7d": rate7,
            "success_rate_30d": rate30,
            "successes_7d": s7,
            "runs_7d": t7,
            "successes_30d": s30,
            "runs_30d": t30,
        })

    return {
        "as_of": str(asof),
        "days": days,
        "engines": engines,
    }
