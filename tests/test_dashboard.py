"""Tests for the dashboard endpoints and static files."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# We need to mock the DB initialization before importing the app
_mock_session_cm = None


def _make_mock_session(execute_side_effect=None):
    """Create a mock async session context manager."""
    mock_session = AsyncMock()
    if execute_side_effect:
        mock_session.execute = AsyncMock(side_effect=execute_side_effect)
    else:
        # Default: return empty results
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)
    return mock_session


@pytest.fixture
def app_client():
    """Create a test client with mocked DB."""
    from contextlib import asynccontextmanager

    mock_session = _make_mock_session()

    @asynccontextmanager
    async def mock_get_session():
        yield mock_session

    with (
        patch("api.app.init_db", new_callable=AsyncMock),
        patch("api.app.close_db", new_callable=AsyncMock),
        patch("api.app.get_session", mock_get_session),
    ):
        from api.app import app
        yield app, mock_session


@pytest.mark.asyncio
async def test_dashboard_returns_200(app_client):
    """/dashboard should return 200 with HTML content."""
    app, _ = app_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Dashboard" in resp.text


@pytest.mark.asyncio
async def test_dashboard_signals_empty(app_client):
    """/api/dashboard/signals returns empty when no runs."""
    app, _ = app_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/dashboard/signals")
    assert resp.status_code == 200
    data = resp.json()
    assert data["signals"] == []
    assert data["run_date"] is None


@pytest.mark.asyncio
async def test_dashboard_signals_with_data():
    """Signals endpoint returns correct shape with mock DB data."""
    from contextlib import asynccontextmanager

    # Create mock run and signals
    mock_run = MagicMock()
    mock_run.run_date = date(2025, 1, 15)
    mock_run.regime = "bull"

    mock_signal = MagicMock()
    mock_signal.ticker = "AAPL"
    mock_signal.direction = "LONG"
    mock_signal.signal_model = "breakout"
    mock_signal.entry_price = 195.0
    mock_signal.stop_loss = 190.0
    mock_signal.target_1 = 210.0
    mock_signal.holding_period_days = 5
    mock_signal.confidence = 75.0
    mock_signal.interpreter_thesis = "Strong momentum"
    mock_signal.regime = "bull"
    mock_signal.risk_gate_decision = "APPROVE"

    call_count = 0

    def execute_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        if call_count == 1:
            result.scalar_one_or_none.return_value = mock_run
        else:
            result.scalars.return_value.all.return_value = [mock_signal]
        return result

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=execute_side_effect)

    @asynccontextmanager
    async def mock_get_session():
        yield mock_session

    with (
        patch("api.app.init_db", new_callable=AsyncMock),
        patch("api.app.close_db", new_callable=AsyncMock),
        patch("api.app.get_session", mock_get_session),
    ):
        from api.app import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dashboard/signals")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_date"] == "2025-01-15"
    assert data["regime"] == "bull"
    assert len(data["signals"]) == 1
    assert data["signals"][0]["ticker"] == "AAPL"
    assert data["signals"][0]["confidence"] == 75.0


@pytest.mark.asyncio
async def test_dashboard_performance_empty(app_client):
    """/api/dashboard/performance returns minimal data when no trades."""
    app, mock_session = app_client

    # Mock get_performance_summary to return no-data response
    with patch("api.app.get_performance_summary", new_callable=AsyncMock) as mock_perf:
        mock_perf.return_value = {"total_signals": 0, "message": "No closed trades in period"}
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dashboard/performance")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_signals"] == 0


@pytest.mark.asyncio
async def test_dashboard_performance_with_equity_curve():
    """Performance endpoint returns equity_curve with cumulative P&L."""
    from contextlib import asynccontextmanager

    mock_outcome1 = MagicMock()
    mock_outcome1.pnl_pct = 2.5
    mock_outcome1.exit_date = date(2025, 1, 10)
    mock_outcome1.entry_date = date(2025, 1, 5)
    mock_outcome1.still_open = False

    mock_outcome2 = MagicMock()
    mock_outcome2.pnl_pct = -1.0
    mock_outcome2.exit_date = date(2025, 1, 12)
    mock_outcome2.entry_date = date(2025, 1, 7)
    mock_outcome2.still_open = False

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_outcome1, mock_outcome2]
    mock_session.execute = AsyncMock(return_value=mock_result)

    @asynccontextmanager
    async def mock_get_session():
        yield mock_session

    perf_data = {
        "total_signals": 2,
        "overall": {"trades": 2, "win_rate": 0.5, "avg_pnl": 0.75},
        "risk_metrics": {"sharpe_ratio": 1.2, "sortino_ratio": 1.5, "max_drawdown_pct": 1.0,
                         "profit_factor": 2.5, "calmar_ratio": 1.5, "expectancy": 0.75,
                         "payoff_ratio": 2.5, "avg_win_pct": 2.5, "avg_loss_pct": -1.0,
                         "max_consecutive_wins": 1, "max_consecutive_losses": 1},
        "by_model": {}, "by_regime": {}, "by_confidence": {},
        "confidence_calibration": [],
    }

    with (
        patch("api.app.init_db", new_callable=AsyncMock),
        patch("api.app.close_db", new_callable=AsyncMock),
        patch("api.app.get_session", mock_get_session),
        patch("api.app.get_performance_summary", new_callable=AsyncMock, return_value=perf_data),
    ):
        from api.app import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dashboard/performance")

    assert resp.status_code == 200
    data = resp.json()
    assert "equity_curve" in data
    curve = data["equity_curve"]
    assert len(curve) == 2
    # Cumulative: 2.5, then 2.5 + (-1.0) = 1.5
    assert curve[0]["value"] == 2.5
    assert curve[1]["value"] == 1.5


@pytest.mark.asyncio
async def test_cache_stats_returns_shape(app_client):
    """/api/cache-stats returns expected shape."""
    app, _ = app_client

    mock_stats = {
        "hits": 0, "misses": 0, "stores": 0,
        "evictions": 0, "hit_rate": 0.0, "total_entries": 0,
    }

    with patch("src.data.aggregator.DataAggregator") as MockAgg:
        mock_agg = MagicMock()
        mock_agg.get_cache_stats.return_value = mock_stats
        MockAgg.return_value = mock_agg

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/cache-stats")

    assert resp.status_code == 200
    data = resp.json()
    expected_keys = {"hits", "misses", "stores", "evictions", "hit_rate", "total_entries"}
    assert set(data.keys()) == expected_keys


def test_static_files_exist():
    """All dashboard static files should exist."""
    static_dir = Path(__file__).resolve().parent.parent / "static"
    assert (static_dir / "dashboard.html").is_file()
    assert (static_dir / "dashboard.css").is_file()
    assert (static_dir / "dashboard.js").is_file()


@pytest.mark.asyncio
async def test_index_has_dashboard_link(app_client):
    """The index page should have a link to /dashboard."""
    app, _ = app_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "/dashboard" in resp.text
