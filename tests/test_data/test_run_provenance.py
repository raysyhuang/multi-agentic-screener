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

import json
from datetime import date

import pandas as pd
import pytest
from unittest.mock import AsyncMock

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


@pytest.mark.parametrize("failing_provider", ["polygon", "fmp", "yfinance"])
@pytest.mark.asyncio
async def test_a_failed_cache_write_never_double_counts_a_ticker(
    monkeypatch, failing_provider
) -> None:
    """A cache write is bookkeeping about a fetch that already succeeded.

    Performing it inside the provider's try made a failed serialization or a
    locked SQLite file indistinguishable from the provider being down: the
    handler recorded a circuit failure, execution fell through, and the NEXT
    provider incremented the counter too. One ticker attributed to two
    providers, totals exceeding the ticker count — corruption of the exact
    record this work exists to make trustworthy.
    """
    agg = DataAggregator()
    agg._cache_enabled = True

    async def ok(*a, **k):
        return _bars()

    def exploding_put(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(agg._cache, "get_with_source", lambda key: (None, ""))
    monkeypatch.setattr(agg._cache, "put", exploding_put)
    monkeypatch.setattr(agg.polygon, "get_ohlcv", ok)
    monkeypatch.setattr(agg.fmp, "get_daily_prices", ok)
    monkeypatch.setattr(agg.yfinance, "get_ohlcv", ok)
    if failing_provider != "polygon":
        monkeypatch.setattr(agg.polygon, "get_ohlcv", AsyncMock(side_effect=RuntimeError("down")))
    if failing_provider == "yfinance":
        monkeypatch.setattr(agg.fmp, "get_daily_prices", AsyncMock(side_effect=RuntimeError("down")))

    df = await agg.get_ohlcv("AAPL", date(2026, 1, 1), date(2026, 8, 10))

    assert not df.empty, "a cache-write failure must not lose the fetched data"
    prov = agg.get_data_provenance()
    assert sum(prov["ohlcv_by_source"].values()) == 1, (
        f"one ticker, one attribution — got {prov['ohlcv_by_source']}"
    )
    assert prov["ohlcv_by_source"] == {failing_provider: 1}
    assert prov["ohlcv_failed_tickers"] == []


@pytest.mark.asyncio
async def test_polygon_universe_survives_a_failed_cache_write(agg, monkeypatch) -> None:
    """Blocker mirror of the FMP branch: a good universe must not be discarded.

    The write failure was caught by the provider handler, flipping the source to
    "unavailable" and returning [] — a false zero-universe run off a universe
    that had been fetched perfectly well.
    """
    agg._cache_enabled = True
    universe = [{"symbol": "AAPL"}, {"symbol": "MSFT"}]

    async def fmp_down(*a, **k):
        raise RuntimeError("fmp down")

    async def polygon_universe():
        return universe

    def exploding_put(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(agg._cache, "get_with_source", lambda key: (None, ""))
    monkeypatch.setattr(agg._cache, "put", exploding_put)
    monkeypatch.setattr(agg.fmp, "get_stock_screener", fmp_down)
    monkeypatch.setattr(agg, "_build_polygon_universe", polygon_universe)

    assert await agg.get_universe() == universe

    prov = agg.get_data_provenance()
    assert prov["universe_source"] == "polygon", "not 'unavailable'"


@pytest.mark.asyncio
async def test_fmp_universe_survives_a_failed_cache_write(agg, monkeypatch) -> None:
    agg._cache_enabled = True
    universe = [{"symbol": "AAPL"}]

    async def screener(*a, **k):
        return universe

    def exploding_put(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(agg._cache, "get_with_source", lambda key: (None, ""))
    monkeypatch.setattr(agg._cache, "put", exploding_put)
    monkeypatch.setattr(agg.fmp, "get_stock_screener", screener)

    assert await agg.get_universe() == universe
    assert agg.get_data_provenance()["universe_source"] == "fmp"


@pytest.mark.asyncio
async def test_universe_cache_hit_reports_the_origin_provider(agg, monkeypatch) -> None:
    """"cache" is not an answer to "where did the universe come from"."""
    agg._cache_enabled = True
    monkeypatch.setattr(
        agg._cache, "get_with_source", lambda key: (json.dumps([{"symbol": "AAPL"}]), "fmp")
    )

    assert await agg.get_universe() == [{"symbol": "AAPL"}]

    prov = agg.get_data_provenance()
    assert prov["universe_source"] == "fmp"
    assert prov["universe_cache_hit"] is True


@pytest.mark.asyncio
async def test_cached_macro_still_attributes_the_benchmark_bars(agg, monkeypatch) -> None:
    """SPY/QQQ drive regime — a cached snapshot must not erase where they came from.

    Returning the payload early skips the provenance-aware OHLCV path entirely,
    so without rehydration the record would claim no benchmark data was used
    while the regime decision rested on exactly that data.
    """
    agg._cache_enabled = True
    payload = {
        "vix": 15.0,
        "spy_prices": df_to_json(_bars()),
        "qqq_prices": df_to_json(_bars()),
        "_benchmark_provenance": {"SPY": "yfinance", "QQQ": "yfinance"},
    }
    monkeypatch.setattr(agg._cache, "get", lambda key: json.dumps(payload))

    macro = await agg.get_macro_context()

    assert not macro["spy_prices"].empty
    assert "_benchmark_provenance" not in macro, "internal key must not leak out"

    prov = agg.get_data_provenance()
    assert prov["macro_source"] == "cache"
    assert prov["macro_cache_hit"] is True
    assert prov["ohlcv_by_source"] == {"yfinance": 2}


@pytest.mark.asyncio
async def test_failed_benchmarks_stay_failed_across_a_cache_replay(
    agg, monkeypatch
) -> None:
    """An empty attribution is a recorded FAILURE, not a missing legacy field.

    Truthiness testing (`if stored:`) conflated the two and fabricated two
    "unknown" benchmark bars for a snapshot that had recorded a genuine SPY/QQQ
    failure — inventing provenance for data that does not exist. The cached
    empty frames go on feeding the regime calculation on every replay, so the
    failure has to persist with them rather than vanishing after the first run.
    """
    agg._cache_enabled = True
    payload = {
        "vix": 15.0,
        "spy_prices": df_to_json(_bars()),
        "qqq_prices": df_to_json(_bars()),
        "_benchmark_provenance": {"SPY": "", "QQQ": ""},
    }
    monkeypatch.setattr(agg._cache, "get", lambda key: json.dumps(payload))

    await agg.get_macro_context()

    prov = agg.get_data_provenance()
    assert prov["ohlcv_by_source"] == {}, "must not invent 'unknown' bars"
    assert prov["ohlcv_failed_tickers"] == ["QQQ", "SPY"]


@pytest.mark.asyncio
async def test_empty_attribution_dict_is_not_treated_as_legacy(agg, monkeypatch) -> None:
    """`{}` is falsey but is not `None` — it must not trigger the legacy path.

    And it must not quietly record nothing either: a benchmark with no readable
    source is a failed benchmark, which is what the shape implies.
    """
    agg._cache_enabled = True
    payload = {
        "vix": 15.0,
        "spy_prices": df_to_json(_bars()),
        "qqq_prices": df_to_json(_bars()),
        "_benchmark_provenance": {},
    }
    monkeypatch.setattr(agg._cache, "get", lambda key: json.dumps(payload))

    await agg.get_macro_context()

    prov = agg.get_data_provenance()
    assert prov["ohlcv_by_source"] == {}, "no fabricated bars"
    assert prov["ohlcv_failed_tickers"] == ["QQQ", "SPY"], "but the gap is recorded"


@pytest.mark.parametrize(
    "hostile",
    [
        {"polygon": 2},                    # the shape a previous revision wrote
        {"SPY": 2, "QQQ": 3},              # non-string values
        {"SPY": None, "QQQ": ["polygon"]},
        {"evil": {"nested": "thing"}},
    ],
)
@pytest.mark.asyncio
async def test_unrecognised_cached_shapes_cannot_become_provider_labels(
    agg, monkeypatch, hostile
) -> None:
    """Cached JSON is untrusted input — never iterate whatever it holds.

    An earlier revision of this branch stored `{provider: count}`. Iterating a
    cached mapping's items would turn the count `2` into a provider label and
    write fabricated provenance into the run record.
    """
    agg._cache_enabled = True
    payload = {
        "vix": 15.0,
        "spy_prices": df_to_json(_bars()),
        "qqq_prices": df_to_json(_bars()),
        "_benchmark_provenance": hostile,
    }
    monkeypatch.setattr(agg._cache, "get", lambda key: json.dumps(payload))

    await agg.get_macro_context()

    prov = agg.get_data_provenance()
    assert prov["ohlcv_by_source"] == {}, f"{hostile} leaked a provider label"
    assert prov["ohlcv_failed_tickers"] == ["QQQ", "SPY"]


@pytest.mark.asyncio
async def test_benchmark_attribution_survives_concurrent_fetching(
    agg, monkeypatch
) -> None:
    """Per-ticker attribution, not a counter diff taken around the gather.

    A diff assumes nothing else fetches during the macro gather — true today
    only because macro runs first. If anything ran alongside it, the other
    tickers' providers would be attributed to the benchmarks. This fetches a
    third ticker concurrently and pins that the benchmarks keep their own.
    """

    async def by_ticker(ticker, *a, **k):
        # Fetch an unrelated ticker from INSIDE the SPY fetch, so NVDA is in
        # flight during the SPY/QQQ gather rather than merely before it — the
        # ordering a counter diff taken around the gather would get wrong.
        if ticker == "SPY":
            await agg.get_ohlcv("NVDA", date(2026, 1, 1), date(2026, 8, 10))
        return _bars()

    async def snapshot():
        return {"vix": 15.0}

    stored: dict = {}
    agg._cache_enabled = True
    monkeypatch.setattr(agg._cache, "get", lambda key: None)
    # Stub the OHLCV cache read too — otherwise SPY/QQQ/NVDA fall through to the
    # real repo-root SQLite cache and the result depends on what is on disk.
    monkeypatch.setattr(agg._cache, "get_with_source", lambda key: (None, ""))
    monkeypatch.setattr(
        agg._cache, "put", lambda key, data, ttl, **kw: stored.update(json.loads(data))
    )
    monkeypatch.setattr(agg.polygon, "get_ohlcv", by_ticker)
    monkeypatch.setattr(agg.fred, "get_macro_snapshot", snapshot)

    await agg.get_macro_context()

    assert stored["_benchmark_provenance"] == {"SPY": "polygon", "QQQ": "polygon"}, (
        "NVDA must not be attributed to the benchmarks"
    )


@pytest.mark.asyncio
async def test_legacy_macro_snapshot_without_attribution_is_marked_unknown(
    agg, monkeypatch
) -> None:
    """Snapshots cached before the field existed: unknown, never silent."""
    agg._cache_enabled = True
    payload = {
        "vix": 15.0,
        "spy_prices": df_to_json(_bars()),
        "qqq_prices": df_to_json(_bars()),
    }
    monkeypatch.setattr(agg._cache, "get", lambda key: json.dumps(payload))

    await agg.get_macro_context()

    assert agg.get_data_provenance()["ohlcv_by_source"] == {"unknown": 2}


@pytest.mark.asyncio
async def test_live_macro_records_its_benchmarks_and_caches_them(agg, monkeypatch) -> None:
    async def ok(*a, **k):
        return _bars()

    async def snapshot():
        return {"vix": 15.0}

    stored: dict = {}
    agg._cache_enabled = True
    monkeypatch.setattr(agg._cache, "get", lambda key: None)
    # Stub the OHLCV cache read too — otherwise SPY/QQQ/NVDA fall through to the
    # real repo-root SQLite cache and the result depends on what is on disk.
    monkeypatch.setattr(agg._cache, "get_with_source", lambda key: (None, ""))
    monkeypatch.setattr(
        agg._cache, "put", lambda key, data, ttl, **kw: stored.update(json.loads(data))
    )
    monkeypatch.setattr(agg.polygon, "get_ohlcv", ok)
    monkeypatch.setattr(agg.fred, "get_macro_snapshot", snapshot)

    await agg.get_macro_context()

    prov = agg.get_data_provenance()
    assert prov["macro_source"] == "live"
    assert prov["macro_cache_hit"] is False
    assert prov["ohlcv_by_source"] == {"polygon": 2}, "SPY + QQQ"
    assert stored["_benchmark_provenance"] == {"SPY": "polygon", "QQQ": "polygon"}, (
        "stored per ticker for the next hit"
    )


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
