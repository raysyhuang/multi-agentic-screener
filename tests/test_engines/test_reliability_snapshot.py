from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.engines.reliability_snapshot import get_engine_reliability_snapshot


def _mock_get_session_with(rows):
    mock_gs = MagicMock()
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = rows
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_gs


def _run_row(
    *,
    engine_name: str,
    run_date: date,
    attempt: int,
    status: str,
    error_message: str | None = None,
):
    row = MagicMock()
    row.engine_name = engine_name
    row.run_date = run_date
    row.attempt = attempt
    row.status = status
    row.error_message = error_message
    return row


@pytest.mark.asyncio
async def test_snapshot_includes_latest_status_streak_and_rates():
    rows = [
        _run_row(engine_name="koocore_d", run_date=date(2026, 3, 7), attempt=1, status="failed", error_message="no_response: timeout"),
        _run_row(engine_name="koocore_d", run_date=date(2026, 3, 6), attempt=1, status="failed", error_message="no_response: timeout"),
        _run_row(engine_name="koocore_d", run_date=date(2026, 3, 5), attempt=1, status="success"),
        _run_row(engine_name="gemini_stst", run_date=date(2026, 3, 7), attempt=1, status="success"),
    ]

    mock_gs = _mock_get_session_with(rows)
    with patch("src.engines.reliability_snapshot.get_session", mock_gs):
        snapshot = await get_engine_reliability_snapshot(days=30, asof_date=date(2026, 3, 7))

    koocore = next(e for e in snapshot["engines"] if e["engine_name"] == "koocore_d")
    assert koocore["latest_status"] == "failed"
    assert koocore["latest_reason"] == "no_response"
    assert koocore["consecutive_failures"] == 2
    assert koocore["last_success_date"] == "2026-03-05"
    assert koocore["success_rate_30d"] == pytest.approx(1 / 3)

    gem = next(e for e in snapshot["engines"] if e["engine_name"] == "gemini_stst")
    assert gem["latest_status"] == "success"
    assert gem["latest_reason"] == "success"
    assert gem["consecutive_failures"] == 0


@pytest.mark.asyncio
async def test_snapshot_dedupes_to_latest_attempt_per_day():
    rows = [
        _run_row(engine_name="top3_7d", run_date=date(2026, 3, 7), attempt=1, status="failed", error_message="no_artifacts: empty"),
        _run_row(engine_name="top3_7d", run_date=date(2026, 3, 7), attempt=2, status="success"),
    ]

    mock_gs = _mock_get_session_with(rows)
    with patch("src.engines.reliability_snapshot.get_session", mock_gs):
        snapshot = await get_engine_reliability_snapshot(days=30, asof_date=date(2026, 3, 7))

    top3 = next(e for e in snapshot["engines"] if e["engine_name"] == "top3_7d")
    assert top3["latest_status"] == "success"
    assert top3["consecutive_failures"] == 0
