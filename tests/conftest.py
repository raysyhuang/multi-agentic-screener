"""Shared test fixtures."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate 100 days of synthetic OHLCV data."""
    np.random.seed(42)
    n = 100
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n)]

    close = 100 + np.cumsum(np.random.randn(n) * 1.5)
    close = np.maximum(close, 10)  # keep positive

    df = pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n)) * 1.0,
        "low": close - abs(np.random.randn(n)) * 1.0,
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
    })
    # Ensure high >= close >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def sample_ohlcv_long() -> pd.DataFrame:
    """Generate 250 days of synthetic OHLCV data for SMA(200) testing."""
    np.random.seed(77)
    n = 250
    dates = [date(2024, 4, 1) + timedelta(days=i) for i in range(n)]
    close = 100 + np.cumsum(np.random.randn(n) * 1.2)
    close = np.maximum(close, 10)
    df = pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n)) * 1.0,
        "low": close - abs(np.random.randn(n)) * 1.0,
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def sample_ohlcv_oversold() -> pd.DataFrame:
    """Generate OHLCV data with a clear oversold condition at the end."""
    np.random.seed(99)
    n = 100
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n)]

    # Uptrend then sharp 5-day selloff
    close = np.concatenate([
        100 + np.cumsum(np.random.randn(95) * 0.8),
        np.array([120, 117, 114, 111, 108]),
    ])
    close = np.maximum(close, 10)

    df = pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.3,
        "high": close + abs(np.random.randn(n)) * 0.8,
        "low": close - abs(np.random.randn(n)) * 0.8,
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def sample_spy_df() -> pd.DataFrame:
    """SPY-like uptrending data for regime detection."""
    np.random.seed(10)
    n = 60
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n)]
    close = 450 + np.cumsum(np.random.randn(n) * 2 + 0.3)  # upward bias

    df = pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n)) * 1.5,
        "low": close - abs(np.random.randn(n)) * 1.5,
        "close": close,
        "volume": np.random.randint(50_000_000, 100_000_000, n).astype(float),
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def sample_qqq_df() -> pd.DataFrame:
    """QQQ-like uptrending data for regime detection."""
    np.random.seed(11)
    n = 60
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n)]
    close = 380 + np.cumsum(np.random.randn(n) * 2 + 0.3)

    df = pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n)) * 1.5,
        "low": close - abs(np.random.randn(n)) * 1.5,
        "close": close,
        "volume": np.random.randint(30_000_000, 80_000_000, n).astype(float),
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def sample_news() -> list[dict]:
    """Sample news articles for sentiment scoring."""
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    return [
        {"title": "Company beats earnings estimates, stock surges", "published_utc": (now - timedelta(hours=2)).isoformat()},
        {"title": "FDA approves new drug, shares rally", "published_utc": (now - timedelta(hours=12)).isoformat()},
        {"title": "Analyst upgrades to buy with strong growth outlook", "published_utc": (now - timedelta(hours=24)).isoformat()},
        {"title": "Company announces layoffs amid restructuring", "published_utc": (now - timedelta(hours=48)).isoformat()},
    ]
