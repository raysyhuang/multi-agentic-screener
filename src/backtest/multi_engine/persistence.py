"""Persistence helpers for multi-engine backtest reports.

Stores full report JSON in Postgres so the dashboard can access results on
ephemeral hosts (e.g., Heroku) where local backtest JSON files are unavailable.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from sqlalchemy import select

from src.config import get_settings
from src.db.models import MultiEngineBacktestRun
from src.db.session import get_session

logger = logging.getLogger(__name__)


def _parse_iso_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    # Ensures date/datetime values inside nested report payload are JSON-safe.
    return json.loads(json.dumps(value, default=str))


def _run_row_to_metadata(row: MultiEngineBacktestRun) -> dict[str, Any]:
    report = row.report or {}
    return {
        "filename": row.filename,
        "run_date": report.get("run_date"),
        "date_range": report.get("date_range"),
        "trading_days": report.get("trading_days"),
        "engines": report.get("engines"),
        "total_trades_all_tracks": report.get("total_trades_all_tracks"),
        "elapsed_s": report.get("elapsed_s"),
        "storage": "database",
    }


async def persist_multi_engine_backtest_report(report: dict[str, Any], filename: str) -> bool:
    """Upsert a multi-engine backtest report into the DB.

    Returns False (without raising) when DB persistence is unavailable, so
    filesystem backtest output remains usable in local/offline workflows.
    """
    settings = get_settings()
    if not settings.database_url:
        return False

    safe_report = _json_safe(report)
    dr = safe_report.get("date_range") or {}

    try:
        async with get_session() as session:
            row = (
                await session.execute(
                    select(MultiEngineBacktestRun).where(MultiEngineBacktestRun.filename == filename)
                )
            ).scalar_one_or_none()
            if row is None:
                row = MultiEngineBacktestRun(filename=filename, report=safe_report)
                session.add(row)

            row.report = safe_report
            row.run_date = _parse_iso_date(safe_report.get("run_date"))
            row.start_date = _parse_iso_date(dr.get("start"))
            row.end_date = _parse_iso_date(dr.get("end"))
            row.trading_days = safe_report.get("trading_days")
            row.total_trades_all_tracks = safe_report.get("total_trades_all_tracks")
            row.engines = safe_report.get("engines")
        return True
    except Exception as exc:
        logger.warning("Failed to persist multi-engine backtest report %s: %s", filename, exc)
        return False


async def list_persisted_multi_engine_backtest_runs() -> list[dict[str, Any]]:
    """Return lightweight metadata for all persisted runs, newest first."""
    settings = get_settings()
    if not settings.database_url:
        return []
    try:
        async with get_session() as session:
            rows = (
                await session.execute(
                    select(MultiEngineBacktestRun).order_by(
                        MultiEngineBacktestRun.start_date.desc().nullslast(),
                        MultiEngineBacktestRun.end_date.desc().nullslast(),
                        MultiEngineBacktestRun.id.desc(),
                    )
                )
            ).scalars().all()
        return [_run_row_to_metadata(r) for r in rows]
    except Exception as exc:
        logger.warning("Failed to list persisted multi-engine backtest runs: %s", exc)
        return []


async def load_persisted_multi_engine_backtest_report(filename: str) -> dict[str, Any] | None:
    """Load one persisted report JSON by filename."""
    settings = get_settings()
    if not settings.database_url:
        return None
    try:
        async with get_session() as session:
            row = (
                await session.execute(
                    select(MultiEngineBacktestRun).where(MultiEngineBacktestRun.filename == filename)
                )
            ).scalar_one_or_none()
        if row is None:
            return None
        return row.report or None
    except Exception as exc:
        logger.warning("Failed to load persisted multi-engine backtest report %s: %s", filename, exc)
        return None
