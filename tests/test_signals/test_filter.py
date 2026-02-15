"""Tests for universe filtering."""

import pandas as pd
import pytest

from src.signals.filter import filter_universe, filter_by_ohlcv


def test_filter_universe_basic():
    candidates = [
        {"symbol": "AAPL", "price": 150, "volume": 1_000_000, "exchangeShortName": "NASDAQ", "type": "stock"},
        {"symbol": "PENNY", "price": 2, "volume": 1_000_000, "exchangeShortName": "NYSE", "type": "stock"},
        {"symbol": "LOWVOL", "price": 50, "volume": 100_000, "exchangeShortName": "NYSE", "type": "stock"},
        {"symbol": "ETFX", "price": 100, "volume": 2_000_000, "exchangeShortName": "NYSE", "type": "ETF"},
        {"symbol": "WRNTS.W", "price": 10, "volume": 500_000, "exchangeShortName": "NYSE", "type": "stock"},
        {"symbol": "GOOD", "price": 75, "volume": 800_000, "exchangeShortName": "NYSE", "type": "stock"},
    ]
    result = filter_universe(candidates)
    tickers = [r["symbol"] for r in result]

    assert "AAPL" in tickers
    assert "GOOD" in tickers
    assert "PENNY" not in tickers  # too cheap
    assert "LOWVOL" not in tickers  # too low volume
    assert "ETFX" not in tickers  # ETF excluded
    assert "WRNTS.W" not in tickers  # warrant excluded


def test_filter_universe_empty():
    assert filter_universe([]) == []


def test_filter_by_ohlcv_passes(sample_ohlcv):
    assert filter_by_ohlcv("TEST", sample_ohlcv) is True


def test_filter_by_ohlcv_short_data():
    df = pd.DataFrame({
        "close": [100, 101, 102],
        "volume": [1e6, 1.1e6, 1.2e6],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
    })
    assert filter_by_ohlcv("TEST", df) is False


def test_filter_by_ohlcv_empty():
    assert filter_by_ohlcv("TEST", pd.DataFrame()) is False
