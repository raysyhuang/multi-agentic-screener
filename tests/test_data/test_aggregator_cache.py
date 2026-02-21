"""Tests for DataAggregator cache integration."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock, patch, MagicMock

import pandas as pd
import pytest

from src.data.aggregator import DataAggregator
from src.data.cache import DataCache, df_to_json


@pytest.fixture
def aggregator(tmp_path):
    """Create an aggregator with mocked clients and a temp cache DB."""
    with (
        patch("src.data.aggregator.PolygonClient"),
        patch("src.data.aggregator.FMPClient"),
        patch("src.data.aggregator.YFinanceClient"),
        patch("src.data.aggregator.FREDClient"),
        patch("src.data.aggregator.DataCache") as MockCache,
    ):
        mock_cache = MagicMock(spec=DataCache)
        mock_cache.get.return_value = None
        mock_cache.get_stats.return_value = {
            "hits": 0, "misses": 0, "stores": 0,
            "evictions": 0, "hit_rate": 0.0, "total_entries": 0,
        }
        MockCache.return_value = mock_cache
        agg = DataAggregator()
        agg._cache = mock_cache
        yield agg


@pytest.fixture
def ohlcv_df():
    return pd.DataFrame({
        "date": [date(2024, 1, 2)],
        "open": [100.0], "high": [105.0], "low": [99.0],
        "close": [103.0], "volume": [1e6],
    })


# ---------------------------------------------------------------------------
# Cache hit skips API call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ohlcv_cache_hit_skips_api(aggregator, ohlcv_df):
    """When cache has data, the fallback chain should not be called."""
    cached_json = df_to_json(ohlcv_df)
    aggregator._cache.get.return_value = cached_json

    df = await aggregator.get_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 5))

    assert not df.empty
    assert df.iloc[0]["close"] == 103.0
    aggregator.polygon.get_ohlcv.assert_not_called()
    aggregator.fmp.get_daily_prices.assert_not_called()


@pytest.mark.asyncio
async def test_fundamentals_cache_hit_skips_api(aggregator):
    cached = json.dumps({
        "earnings_surprises": [{"actual": 2.0}],
        "insider_transactions": [],
        "profile": {"symbol": "AAPL"},
    })
    aggregator._cache.get.return_value = cached

    result = await aggregator.get_ticker_fundamentals("AAPL")

    assert result["profile"]["symbol"] == "AAPL"
    aggregator.fmp.get_earnings_surprise.assert_not_called()


# ---------------------------------------------------------------------------
# Cache miss falls through and stores
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ohlcv_cache_miss_calls_api_and_stores(aggregator, ohlcv_df):
    """On cache miss, API is called and result is stored."""
    aggregator._cache.get.return_value = None
    aggregator.polygon.get_ohlcv = AsyncMock(return_value=ohlcv_df)

    df = await aggregator.get_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 5))

    assert not df.empty
    aggregator.polygon.get_ohlcv.assert_called_once()
    aggregator._cache.put.assert_called_once()


@pytest.mark.asyncio
async def test_news_cache_miss_stores(aggregator):
    aggregator._cache.get.return_value = None
    aggregator.polygon.get_news = AsyncMock(return_value=[{"title": "Breaking news"}])

    result = await aggregator.get_ticker_news("AAPL")

    assert len(result) == 1
    aggregator._cache.put.assert_called_once()


# ---------------------------------------------------------------------------
# Cache disabled bypasses entirely
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_disabled_bypasses(aggregator, ohlcv_df):
    aggregator._cache_enabled = False
    aggregator.polygon.get_ohlcv = AsyncMock(return_value=ohlcv_df)

    df = await aggregator.get_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 5))

    assert not df.empty
    aggregator._cache.get.assert_not_called()
    aggregator._cache.put.assert_not_called()


# ---------------------------------------------------------------------------
# get_cache_stats returns expected shape
# ---------------------------------------------------------------------------

def test_get_cache_stats_shape(aggregator):
    stats = aggregator.get_cache_stats()
    expected_keys = {"hits", "misses", "stores", "evictions", "hit_rate", "total_entries"}
    assert set(stats.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Macro context preserves DataFrames through serialization
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_macro_cache_preserves_dataframes(aggregator):
    """Macro context cached with serialized DFs should restore them."""
    spy_df = pd.DataFrame({
        "date": [date(2024, 1, 2)],
        "close": [450.0], "open": [449.0], "high": [452.0], "low": [448.0], "volume": [5e7],
    })
    qqq_df = pd.DataFrame({
        "date": [date(2024, 1, 2)],
        "close": [380.0], "open": [379.0], "high": [382.0], "low": [378.0], "volume": [3e7],
    })

    cached_payload = json.dumps({
        "vix": 15.2,
        "yield_10y": 4.1,
        "spy_prices": df_to_json(spy_df),
        "qqq_prices": df_to_json(qqq_df),
    })
    aggregator._cache.get.return_value = cached_payload

    result = await aggregator.get_macro_context()

    assert isinstance(result["spy_prices"], pd.DataFrame)
    assert isinstance(result["qqq_prices"], pd.DataFrame)
    assert result["spy_prices"].iloc[0]["close"] == 450.0
    assert result["vix"] == 15.2


# ---------------------------------------------------------------------------
# Universe cache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_universe_cache_hit(aggregator):
    cached = json.dumps([{"symbol": "AAPL"}, {"symbol": "MSFT"}])
    aggregator._cache.get.return_value = cached

    result = await aggregator.get_universe()

    assert len(result) == 2
    aggregator.fmp.get_stock_screener.assert_not_called()


@pytest.mark.asyncio
async def test_fundamentals_all_fail_does_not_cache_empty_payload(aggregator):
    """Avoid caching pure fallback empties from failed FMP calls."""
    aggregator._cache.get.return_value = None
    aggregator.fmp.get_earnings_surprise = AsyncMock(side_effect=Exception("fmp blocked"))
    aggregator.fmp.get_insider_trading = AsyncMock(side_effect=Exception("fmp blocked"))
    aggregator.fmp.get_company_profile = AsyncMock(side_effect=Exception("fmp blocked"))

    result = await aggregator.get_ticker_fundamentals("AAPL")

    assert result == {
        "earnings_surprises": [],
        "insider_transactions": [],
        "profile": {},
    }
    aggregator._cache.put.assert_not_called()
