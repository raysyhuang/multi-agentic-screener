"""Tests for performance window filtering behavior."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.output.performance as perf


@pytest.mark.asyncio
async def test_get_equity_curve_filters_closed_trades_by_exit_date(monkeypatch):
    """Equity curve window should be based on close date, not entry date."""
    captured: dict[str, str] = {}

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [
        SimpleNamespace(
            pnl_pct=1.25,
            exit_date=date(2026, 2, 20),
            entry_date=date(2025, 10, 1),
            still_open=False,
        )
    ]

    class _FakeSession:
        async def execute(self, statement):
            captured["where"] = str(statement.whereclause)
            return mock_result

    @asynccontextmanager
    async def _fake_get_session():
        yield _FakeSession()

    monkeypatch.setattr(perf, "get_session", _fake_get_session)

    curve = await perf.get_equity_curve(days=90)

    assert len(curve) == 1
    where = captured.get("where", "").lower()
    assert "exit_date" in where
    assert "entry_date" not in where


@pytest.mark.asyncio
async def test_get_return_distribution_filters_by_exit_date(monkeypatch):
    """Return distribution window should be based on trade close date."""
    captured: dict[str, str] = {}

    mock_result = MagicMock()
    mock_result.all.return_value = []

    class _FakeSession:
        async def execute(self, statement):
            captured["where"] = str(statement.whereclause)
            return mock_result

    @asynccontextmanager
    async def _fake_get_session():
        yield _FakeSession()

    monkeypatch.setattr(perf, "get_session", _fake_get_session)

    data = await perf.get_return_distribution(days=90)

    assert data == {}
    where = captured.get("where", "").lower()
    assert "exit_date" in where
    assert "entry_date" not in where


@pytest.mark.asyncio
async def test_get_regime_matrix_filters_by_exit_date(monkeypatch):
    """Regime matrix window should be based on trade close date."""
    captured: dict[str, str] = {}

    mock_result = MagicMock()
    mock_result.all.return_value = []

    class _FakeSession:
        async def execute(self, statement):
            captured["where"] = str(statement.whereclause)
            return mock_result

    @asynccontextmanager
    async def _fake_get_session():
        yield _FakeSession()

    monkeypatch.setattr(perf, "get_session", _fake_get_session)

    data = await perf.get_regime_matrix(days=180)

    assert data == []
    where = captured.get("where", "").lower()
    assert "exit_date" in where
    assert "entry_date" not in where
