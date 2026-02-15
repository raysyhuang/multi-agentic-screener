"""Tests for the SQLite data cache layer."""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import pytest

from src.data.cache import (
    DataCache,
    TTL_HISTORICAL_OHLCV,
    TTL_RECENT_OHLCV,
    classify_ohlcv_ttl,
    df_to_json,
    json_to_df,
)


@pytest.fixture
def cache(tmp_path):
    """Create a DataCache backed by a temporary SQLite file."""
    c = DataCache(db_path=tmp_path / "test_cache.sqlite")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

class TestBuildKey:
    def test_deterministic(self):
        k1 = DataCache.build_key("polygon", "AAPL", "ohlcv", from_date="2024-01-01")
        k2 = DataCache.build_key("polygon", "AAPL", "ohlcv", from_date="2024-01-01")
        assert k1 == k2

    def test_param_sensitivity(self):
        k1 = DataCache.build_key("polygon", "AAPL", "ohlcv", from_date="2024-01-01")
        k2 = DataCache.build_key("polygon", "AAPL", "ohlcv", from_date="2024-01-02")
        assert k1 != k2

    def test_source_sensitivity(self):
        k1 = DataCache.build_key("polygon", "AAPL", "ohlcv")
        k2 = DataCache.build_key("fmp", "AAPL", "ohlcv")
        assert k1 != k2

    def test_ticker_sensitivity(self):
        k1 = DataCache.build_key("polygon", "AAPL", "ohlcv")
        k2 = DataCache.build_key("polygon", "MSFT", "ohlcv")
        assert k1 != k2

    def test_param_order_irrelevant(self):
        k1 = DataCache.build_key("p", "T", "e", a="1", b="2")
        k2 = DataCache.build_key("p", "T", "e", b="2", a="1")
        assert k1 == k2


# ---------------------------------------------------------------------------
# Put / get roundtrip
# ---------------------------------------------------------------------------

class TestPutGet:
    def test_roundtrip(self, cache):
        cache.put("k1", '{"data": 42}', ttl_seconds=3600)
        result = cache.get("k1")
        assert result == '{"data": 42}'

    def test_ttl_expiry(self, cache):
        cache.put("k2", '{"val": 1}', ttl_seconds=1)
        time.sleep(1.1)
        result = cache.get("k2")
        assert result is None

    def test_miss_returns_none(self, cache):
        assert cache.get("nonexistent") is None

    def test_overwrite(self, cache):
        cache.put("k3", '"old"', ttl_seconds=3600)
        cache.put("k3", '"new"', ttl_seconds=3600)
        assert cache.get("k3") == '"new"'


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStats:
    def test_initial_stats(self, cache):
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["stores"] == 0
        assert stats["hit_rate"] == 0.0

    def test_hit_miss_counting(self, cache):
        cache.put("s1", '"data"', ttl_seconds=3600)
        cache.get("s1")  # hit
        cache.get("s1")  # hit
        cache.get("miss_key")  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["stores"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_stats_shape(self, cache):
        stats = cache.get_stats()
        expected_keys = {"hits", "misses", "stores", "evictions", "hit_rate", "total_entries"}
        assert set(stats.keys()) == expected_keys


# ---------------------------------------------------------------------------
# DataFrame serialization
# ---------------------------------------------------------------------------

class TestDataFrameSerialization:
    def test_roundtrip(self):
        df = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "close": [100.0, 101.5],
            "volume": [1_000_000, 1_200_000],
        })
        json_str = df_to_json(df)
        restored = json_to_df(json_str)
        assert list(restored.columns) == ["date", "close", "volume"]
        assert restored.iloc[0]["close"] == 100.0
        assert restored.iloc[1]["date"] == date(2024, 1, 2)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        json_str = df_to_json(df)
        restored = json_to_df(json_str)
        assert restored.empty


# ---------------------------------------------------------------------------
# classify_ohlcv_ttl
# ---------------------------------------------------------------------------

class TestClassifyOhlcvTtl:
    def test_historical_date(self):
        old = date.today() - timedelta(days=30)
        assert classify_ohlcv_ttl(old) == TTL_HISTORICAL_OHLCV

    def test_recent_date(self):
        recent = date.today()
        assert classify_ohlcv_ttl(recent) == TTL_RECENT_OHLCV

    def test_none_defaults_to_recent(self):
        assert classify_ohlcv_ttl(None) == TTL_RECENT_OHLCV

    def test_boundary_at_3_days(self):
        exactly_3 = date.today() - timedelta(days=3)
        assert classify_ohlcv_ttl(exactly_3) == TTL_RECENT_OHLCV
        four_days = date.today() - timedelta(days=4)
        assert classify_ohlcv_ttl(four_days) == TTL_HISTORICAL_OHLCV


# ---------------------------------------------------------------------------
# clear_expired
# ---------------------------------------------------------------------------

class TestClearExpired:
    def test_bulk_eviction(self, cache):
        cache.put("exp1", '"a"', ttl_seconds=1)
        cache.put("exp2", '"b"', ttl_seconds=1)
        cache.put("keep", '"c"', ttl_seconds=3600)
        time.sleep(1.1)

        deleted = cache.clear_expired()
        assert deleted == 2
        assert cache.get("keep") == '"c"'

    def test_no_expired(self, cache):
        cache.put("alive", '"x"', ttl_seconds=3600)
        deleted = cache.clear_expired()
        assert deleted == 0
