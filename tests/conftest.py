"""Shared test fixtures."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Sibling-repo paths (KooCore-D, Gemini STST) ─────────────────────────
# These repos are expected as siblings of the MAS project root for local
# cross-engine tests.  In CI they don't exist, so tests importing from
# them must be skipped.  Two layers of protection:
#   1. Module-level pytest.skip() guards in each test file (immediate)
#   2. The @pytest.mark.sibling_repo marker + auto-skip hook below (belt & suspenders)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIBLING_REPOS = {
    "KooCore-D": _PROJECT_ROOT / ".." / "KooCore-D",
    "Gemini STST": _PROJECT_ROOT / ".." / "Gemini STST",
}
SIBLING_REPOS_AVAILABLE = all(p.exists() for p in SIBLING_REPOS.values())


def pytest_configure(config):
    """Cut the test suite off from the developer's `.env`.

    `Settings` loads `PROJECT_ROOT/.env` (src/config.py), so on a machine with
    real credentials the suite sees populated API keys and a live DATABASE_URL,
    while CI sees neither. Tests that touch settings therefore pass locally and
    fail on the runner — three times in one session, each costing a CI round
    trip: `validate_keys_for_mode()` in the worker, then again in `main()`, and
    an aggregator cache test reaching the real SQLite file.

    The failure is one-directional and that is what makes it expensive: local
    is the permissive environment, so it never catches what CI will reject. The
    fix is to make the strict one the default. A test needing credentials must
    now state that by stubbing settings, rather than inheriting whatever the
    machine happens to carry.

    Verified before landing: the full suite passes with `.env` moved aside, so
    this changes which environment tests see, not which tests pass.
    """
    from src import config as config_mod

    config_mod.Settings.model_config["env_file"] = None
    # Drop any Settings already built from the dotenv during collection.
    config_mod._settings = None


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.sibling_repo tests when repos aren't cloned."""
    if SIBLING_REPOS_AVAILABLE:
        return
    skip_marker = pytest.mark.skip(reason="Sibling repos not found (expected in CI)")
    for item in items:
        if "sibling_repo" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    """Seed numpy's global RNG before every test.

    Synthetic OHLCV is built with `np.random` in conftest and in several test
    modules. Some of those call `np.random.seed()` and some don't, and numpy's
    seed is process-global — so whether an unseeded fixture produced repeatable
    data depended on which seeded fixture happened to run before it. Values
    derived from those prices (health composite score, regime classification,
    ranker scores) therefore drifted between runs.

    That made `TestVelocityInHealthCard::test_velocity_warning_forces_watch`
    fail ~15% of runs (3/20) when the score landed past the EXIT threshold
    instead of WATCH. Seeding per test keeps the data realistic but makes every
    test independent of collection order.
    """
    np.random.seed(1729)


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
