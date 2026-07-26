"""Minute-bar same-bar tie resolver for the backtest exit walk.

On a daily bar where BOTH the stop and the target are touched, daily OHLC can't
say which came first, so the exit engine conservatively takes the stop. That
assumption is the crux of the backtest-vs-live gap (it helped inflate the 82%
sniper number). This fetches that day's Polygon 1-minute bars (sync, disk-cached)
and asks exit_engine.resolve_first_touch which level was actually hit first —
turning the assumption into a measurement.

Only called on AMBIGUOUS bars (both levels in range, no opening gap), so the
number of minute fetches is small and bounded. Requires the $199 intraday plan.
walk_exit stays pure: it calls this via ExitParams.same_bar_resolver, sync.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from src.backtest.exit_engine import resolve_first_touch
from src.config import get_settings

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/intraday")
BASE_URL = "https://api.polygon.io"


def _fetch_minute_lowhigh(ticker: str, day: date) -> list[tuple[float, float]]:
    """Time-ordered (low, high) 1-minute bars for ticker×day (adjusted).

    Disk-cached at data/cache/intraday/{ticker}_{day}.parquet (reuses any cache
    get_intraday_aggs already wrote — both are sorted ascending by time). Empty
    list on any failure → the caller keeps the conservative stop.
    """
    cache = CACHE_DIR / f"{ticker}_{day}.parquet"
    if cache.exists():
        try:
            d = pd.read_parquet(cache)
            if {"low", "high"}.issubset(d.columns) and len(d):
                return list(zip(d["low"].to_numpy(), d["high"].to_numpy()))
            return []
        except Exception:  # noqa: BLE001 — corrupt cache → refetch below
            pass

    poly_sym = ticker.replace("-", ".")  # dash→dot for Polygon share classes
    url = f"{BASE_URL}/v2/aggs/ticker/{poly_sym}/range/1/minute/{day}/{day}"
    key = get_settings().polygon_api_key
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, params={"apiKey": key, "adjusted": "true",
                                           "sort": "asc", "limit": 50000})
            resp.raise_for_status()
            results = resp.json().get("results", [])
    except Exception as e:  # noqa: BLE001 — network/plan error → unresolved
        logger.debug("minute fetch failed %s %s: %s", ticker, day, e)
        return []

    if results:
        df = pd.DataFrame(results).rename(columns={"l": "low", "h": "high", "t": "timestamp"})
        df = df.sort_values("timestamp")[["low", "high"]].reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["low", "high"])
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    except Exception:  # noqa: BLE001 — cache write is best-effort
        pass
    return list(zip(df["low"].to_numpy(), df["high"].to_numpy()))


class MinuteResolver:
    """A per-ticker ExitParams.same_bar_resolver that counts what it did, so a
    backtest can report how much the minute resolution actually changed."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.calls = 0        # ambiguous bars seen
        self.resolved = 0     # had minute data (stop or target)
        self.flipped = 0      # minute data said TARGET (would've been stop)

    def __call__(self, day: date, stop: float, target: float) -> str | None:
        self.calls += 1
        bars = _fetch_minute_lowhigh(self.ticker, day)
        if not bars:
            return None
        verdict = resolve_first_touch(bars, stop, target)
        if verdict is not None:
            self.resolved += 1
        if verdict == "target":
            self.flipped += 1
        return verdict


def make_minute_resolver(ticker: str) -> MinuteResolver:
    return MinuteResolver(ticker)
