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
from src.data.cache import df_to_json


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
async def test_cache_hit_is_attributed_to_the_provider_that_served_it(monkeypatch) -> None:
    """A cache hit is a delivery mechanism, not a data source.

    Reporting hits as source "cache" would hide a run built entirely on bars
    yfinance served during an earlier Polygon outage — the run would look
    unattributed rather than degraded. The origin was always stored in
    `cache_entries.source`; it was simply never read back.
    """
    agg = DataAggregator()
    agg._cache_enabled = True
    served: dict = {}

    def fake_get_with_source(key):
        served["key"] = key
        return df_to_json(_bars()), "yfinance"

    monkeypatch.setattr(agg._cache, "get_with_source", fake_get_with_source)

    await agg.get_ohlcv("AAPL", date(2026, 1, 1), date(2026, 8, 10))

    prov = agg.get_data_provenance()
    assert prov["ohlcv_by_source"] == {"yfinance": 1}, "origin, not 'cache'"
    assert prov["ohlcv_cache_hits"] == 1, "but cache reliance stays visible"


@pytest.mark.asyncio
async def test_cache_hit_without_a_recorded_origin_is_not_silently_attributed(
    monkeypatch,
) -> None:
    agg = DataAggregator()
    agg._cache_enabled = True
    monkeypatch.setattr(
        agg._cache, "get_with_source", lambda key: (df_to_json(_bars()), "")
    )

    await agg.get_ohlcv("AAPL", date(2026, 1, 1), date(2026, 8, 10))
    assert agg.get_data_provenance()["ohlcv_by_source"] == {"unknown": 1}


@pytest.mark.asyncio
async def test_total_universe_failure_is_explicit_not_blank(agg, monkeypatch) -> None:
    """`[]` with a blank source reads as "has not run yet". Say which it is."""

    async def boom(*a, **k):
        raise RuntimeError("fmp down")

    async def polygon_boom():
        raise RuntimeError("polygon down")

    monkeypatch.setattr(agg.fmp, "get_stock_screener", boom)
    monkeypatch.setattr(agg, "_build_polygon_universe", polygon_boom)

    assert await agg.get_universe() == []

    prov = agg.get_data_provenance()
    assert prov["universe_source"] == "unavailable"
    assert any("fmp" in e for e in prov["universe_errors"])
    assert any("polygon" in e for e in prov["universe_errors"])


@pytest.mark.asyncio
async def test_bulk_task_exception_names_the_ticker(agg, monkeypatch) -> None:
    """A task dying outside get_ohlcv's own handling must not vanish.

    get_ohlcv swallows provider errors itself, so an exception surfacing in
    get_bulk_ohlcv means the task died some other way — cancellation, teardown,
    a bug. It was previously turned into an empty frame while the provenance
    record went on claiming nothing had failed.
    """

    async def explode(ticker, *a, **k):
        raise RuntimeError("task died")

    monkeypatch.setattr(agg, "get_ohlcv", explode)

    out = await agg.get_bulk_ohlcv(["AAPL", "MSFT"], date(2026, 1, 1), date(2026, 8, 10))

    assert all(df.empty for df in out.values())
    assert agg.get_data_provenance()["ohlcv_failed_tickers"] == ["AAPL", "MSFT"]


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
