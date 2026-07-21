"""Tests for the point-in-time earnings cache — injectable fetcher, no network."""

from __future__ import annotations


import pytest

from src.data import earnings_cache


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(earnings_cache, "CACHE_DIR", tmp_path / "earnings")


@pytest.mark.asyncio
async def test_fetches_then_caches():
    calls = {"n": 0}

    async def fake(ticker):
        calls["n"] += 1
        return [{"date": "2025-01-30", "epsActual": 2.0, "epsEstimated": 1.8}]

    r1 = await earnings_cache.get_earnings("AAPL", fetcher=fake)
    assert len(r1) == 1 and calls["n"] == 1
    # Second call hits disk cache.
    r2 = await earnings_cache.get_earnings("AAPL", fetcher=fake)
    assert len(r2) == 1 and calls["n"] == 1


@pytest.mark.asyncio
async def test_empty_is_cached():
    calls = {"n": 0}

    async def fake(ticker):
        calls["n"] += 1
        return []

    assert await earnings_cache.get_earnings("XYZ", fetcher=fake) == []
    assert await earnings_cache.get_earnings("XYZ", fetcher=fake) == []
    assert calls["n"] == 1  # empty result cached, not refetched


@pytest.mark.asyncio
async def test_fetch_error_not_cached(tmp_path):
    calls = {"n": 0}

    async def boom(ticker):
        calls["n"] += 1
        raise RuntimeError("fmp down")

    assert await earnings_cache.get_earnings("ERR", fetcher=boom) == []
    # Error path must NOT write the cache → retried next time.
    assert not earnings_cache._cache_path("ERR").exists()
    assert await earnings_cache.get_earnings("ERR", fetcher=boom) == []
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_no_cache_forces_refetch():
    calls = {"n": 0}

    async def fake(ticker):
        calls["n"] += 1
        return [{"date": "2025-01-30", "epsActual": 1.0, "epsEstimated": 1.0}]

    await earnings_cache.get_earnings("AAPL", fetcher=fake)
    await earnings_cache.get_earnings("AAPL", fetcher=fake, no_cache=True)
    assert calls["n"] == 2
