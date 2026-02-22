"""Tests for data aggregator with mocked clients."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.data.aggregator import DataAggregator


@pytest.fixture
def aggregator():
    with (
        patch("src.data.aggregator.PolygonClient"),
        patch("src.data.aggregator.FMPClient"),
        patch("src.data.aggregator.YFinanceClient"),
        patch("src.data.aggregator.FREDClient"),
        patch("src.data.aggregator.DataCache"),
    ):
        agg = DataAggregator()
        agg._cache_enabled = False
        return agg


@pytest.mark.asyncio
async def test_get_ohlcv_falls_back_on_failure(aggregator):
    """Test fallback chain: Polygon fails → FMP fails → yfinance succeeds."""
    expected_df = pd.DataFrame({
        "date": [date(2024, 1, 2)],
        "open": [100.0], "high": [105.0], "low": [99.0], "close": [103.0], "volume": [1e6],
    })

    aggregator.polygon.get_ohlcv = AsyncMock(side_effect=Exception("Polygon down"))
    aggregator.fmp.get_daily_prices = AsyncMock(return_value=pd.DataFrame())
    aggregator.yfinance.get_ohlcv = AsyncMock(return_value=expected_df)

    df = await aggregator.get_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 5))
    assert not df.empty
    assert df.iloc[0]["close"] == 103.0


@pytest.mark.asyncio
async def test_get_ohlcv_uses_polygon_first(aggregator):
    """Polygon is the primary source."""
    expected_df = pd.DataFrame({
        "date": [date(2024, 1, 2)],
        "open": [100.0], "high": [105.0], "low": [99.0], "close": [103.0], "volume": [1e6],
    })

    aggregator.polygon.get_ohlcv = AsyncMock(return_value=expected_df)

    df = await aggregator.get_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 5))
    assert not df.empty
    aggregator.polygon.get_ohlcv.assert_called_once()


@pytest.mark.asyncio
async def test_get_bulk_ohlcv_handles_mixed_results(aggregator):
    good_df = pd.DataFrame({
        "date": [date(2024, 1, 2)],
        "open": [100.0], "high": [105.0], "low": [99.0], "close": [103.0], "volume": [1e6],
    })

    aggregator.polygon.get_ohlcv = AsyncMock(side_effect=[good_df, Exception("Failed")])
    aggregator.fmp.get_daily_prices = AsyncMock(return_value=pd.DataFrame())
    aggregator.yfinance.get_ohlcv = AsyncMock(return_value=pd.DataFrame())

    result = await aggregator.get_bulk_ohlcv(["AAPL", "FAIL"], date(2024, 1, 1), date(2024, 1, 5))
    assert "AAPL" in result
    assert "FAIL" in result


@pytest.mark.asyncio
async def test_get_universe_calls_fmp_screener(aggregator):
    aggregator.fmp.get_stock_screener = AsyncMock(return_value=[
        {"symbol": "AAPL", "price": 195}, {"symbol": "MSFT", "price": 380},
    ])

    result = await aggregator.get_universe()
    assert len(result) == 2
    aggregator.fmp.get_stock_screener.assert_called_once()


@pytest.mark.asyncio
async def test_get_ticker_fundamentals_aggregates(aggregator):
    aggregator.fmp.get_earnings_surprise = AsyncMock(return_value=[{"actual": 2.0, "estimate": 1.8}])
    aggregator.fmp.get_insider_trading = AsyncMock(return_value=[{"type": "P"}])
    aggregator.fmp.get_company_profile = AsyncMock(return_value={"symbol": "AAPL"})
    aggregator.fmp.get_analyst_estimates = AsyncMock(return_value=[{"estimatedEpsAvg": 2.1}])
    aggregator.fmp.get_ratios = AsyncMock(return_value={"priceEarningsRatio": 15.0})

    result = await aggregator.get_ticker_fundamentals("AAPL")
    assert "earnings_surprises" in result
    assert "insider_transactions" in result
    assert "profile" in result
    assert "analyst_estimates" in result
    assert "ratios" in result
