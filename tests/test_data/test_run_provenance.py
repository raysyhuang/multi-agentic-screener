"""The live path must record which provider actually served the data.

CLAUDE.md requires any artifact claiming a data source to stamp its provenance,
but the only implementation of that rule — `get_last_ohlcv_provenance()` — lives
in `src/research/signal_backtest.py` and covers the research path. The live
pipeline runs `DataAggregator`, whose fallback chain (Polygon -> FMP ->
yfinance) is silent: a Polygon outage degrades an entire run to yfinance bars
and nothing in the output says so.

These tests pin the live-path counterpart.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.aggregator import DataAggregator


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [100]},
        index=pd.to_datetime(["2026-08-10"]),
    )


@pytest.fixture
def agg(monkeypatch):
    a = DataAggregator()
    a._cache_enabled = False
    return a


@pytest.mark.asyncio
async def test_polygon_success_is_attributed_to_polygon(agg, monkeypatch) -> None:
    async def ok(*a, **k):
        return _bars()

    monkeypatch.setattr(agg.polygon, "get_ohlcv", ok)
    await agg.get_ohlcv("AAPL", date(2026, 1, 1), date(2026, 8, 10))

    prov = agg.get_data_provenance()
    assert prov["ohlcv_by_source"] == {"polygon": 1}
    assert prov["ohlcv_failed_tickers"] == []


@pytest.mark.asyncio
async def test_silent_fallback_to_yfinance_is_recorded(agg, monkeypatch) -> None:
    """The failure this exists to surface: the run completes on fallback data."""

    async def boom(*a, **k):
        raise RuntimeError("polygon down")

    async def empty(*a, **k):
        return pd.DataFrame()

    async def ok(*a, **k):
        return _bars()

    monkeypatch.setattr(agg.polygon, "get_ohlcv", boom)
    monkeypatch.setattr(agg.fmp, "get_daily_prices", empty)
    monkeypatch.setattr(agg.yfinance, "get_ohlcv", ok)

    df = await agg.get_ohlcv("AAPL", date(2026, 1, 1), date(2026, 8, 10))

    assert not df.empty, "the run still succeeds — that is precisely the problem"
    assert agg.get_data_provenance()["ohlcv_by_source"] == {"yfinance": 1}


@pytest.mark.asyncio
async def test_total_failure_names_the_ticker(agg, monkeypatch) -> None:
    async def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(agg.polygon, "get_ohlcv", boom)
    monkeypatch.setattr(agg.fmp, "get_daily_prices", boom)
    monkeypatch.setattr(agg.yfinance, "get_ohlcv", boom)

    await agg.get_ohlcv("ZZZZ", date(2026, 1, 1), date(2026, 8, 10))

    prov = agg.get_data_provenance()
    assert prov["ohlcv_failed_tickers"] == ["ZZZZ"]
    assert prov["ohlcv_by_source"] == {}


@pytest.mark.asyncio
async def test_reset_scopes_counters_to_one_run(agg, monkeypatch) -> None:
    async def ok(*a, **k):
        return _bars()

    monkeypatch.setattr(agg.polygon, "get_ohlcv", ok)
    await agg.get_ohlcv("AAPL", date(2026, 1, 1), date(2026, 8, 10))
    agg.reset_data_provenance()

    assert agg.get_data_provenance()["ohlcv_by_source"] == {}


@pytest.mark.asyncio
async def test_universe_source_reflects_the_provider_that_served(agg, monkeypatch) -> None:
    async def screener(*a, **k):
        return [{"symbol": "AAPL"}]

    monkeypatch.setattr(agg.fmp, "get_stock_screener", screener)
    await agg.get_universe()
    assert agg.get_data_provenance()["universe_source"] == "fmp"


@pytest.mark.asyncio
async def test_universe_falls_back_and_says_so(agg, monkeypatch) -> None:
    async def boom(*a, **k):
        raise RuntimeError("fmp down")

    async def polygon_universe():
        return [{"symbol": "AAPL"}]

    monkeypatch.setattr(agg.fmp, "get_stock_screener", boom)
    monkeypatch.setattr(agg, "_build_polygon_universe", polygon_universe)
    await agg.get_universe()
    assert agg.get_data_provenance()["universe_source"] == "polygon"
