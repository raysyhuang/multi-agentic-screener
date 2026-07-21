"""Tests for the windowed intraday (minute-bar) cache — injectable fetcher, no network."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data import intraday_cache


def _bars(day: date, n: int = 3) -> pd.DataFrame:
    base = pd.Timestamp(day)
    return pd.DataFrame({
        "datetime": [base + pd.Timedelta(minutes=i) for i in range(n)],
        "date": [day] * n,
        "open": [100.0 + i for i in range(n)],
        "high": [101.0 + i for i in range(n)],
        "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
        "volume": [1000 * (i + 1) for i in range(n)],
        "vwap": [100.2 + i for i in range(n)],
        "trades": [10 * (i + 1) for i in range(n)],
    })


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(intraday_cache, "CACHE_DIR", tmp_path / "intraday")


@pytest.mark.asyncio
async def test_get_intraday_day_fetches_then_caches():
    day = date(2024, 1, 3)
    calls = {"n": 0}

    async def fake_fetch(ticker, f, t):
        calls["n"] += 1
        return _bars(day)

    df1 = await intraday_cache.get_intraday_day("AAPL", day, fetcher=fake_fetch)
    assert len(df1) == 3 and calls["n"] == 1

    # Second call must hit the disk cache, not the fetcher.
    df2 = await intraday_cache.get_intraday_day("AAPL", day, fetcher=fake_fetch)
    assert len(df2) == 3 and calls["n"] == 1


@pytest.mark.asyncio
async def test_empty_day_is_cached_not_refetched():
    day = date(2024, 1, 6)  # a Saturday — no bars
    calls = {"n": 0}

    async def fake_fetch(ticker, f, t):
        calls["n"] += 1
        return pd.DataFrame()

    df1 = await intraday_cache.get_intraday_day("AAPL", day, fetcher=fake_fetch)
    assert df1.empty and calls["n"] == 1
    # Empty marker cached → no refetch.
    df2 = await intraday_cache.get_intraday_day("AAPL", day, fetcher=fake_fetch)
    assert df2.empty and calls["n"] == 1


@pytest.mark.asyncio
async def test_window_concatenates_and_skips_empty():
    d1, d2, d3 = date(2024, 1, 3), date(2024, 1, 6), date(2024, 1, 4)

    async def fake_fetch(ticker, f, t):
        return _bars(f) if f.weekday() < 5 else pd.DataFrame()

    df = await intraday_cache.get_intraday_window("AAPL", [d1, d2, d3], fetcher=fake_fetch)
    # d2 (Saturday) contributes nothing; d1 and d3 contribute 3 each.
    assert len(df) == 6
    # Sorted by datetime ascending.
    assert df["datetime"].is_monotonic_increasing
