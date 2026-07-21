"""Windowed intraday (minute-bar) cache.

Phase-3 data layer. Backtests and fill-realism studies need minute bars only for
specific (ticker, day) trade windows — never the whole universe — so we cache
per (ticker, day) on disk and reuse across strategies and runs. A day with no
bars (weekend/holiday/halt) is cached as an empty marker so we don't refetch it.

The Polygon fetch is injectable (``fetcher``) so this is unit-testable without a
live key or network. Default fetcher is PolygonClient.get_intraday_aggs.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Awaitable, Callable

import pandas as pd

CACHE_DIR = Path("data/cache/intraday")
_EMPTY_COLS = ["datetime", "date", "open", "high", "low", "close", "volume", "vwap", "trades"]

# A fetcher takes (ticker, from_date, to_date) and returns a minute-bar frame.
Fetcher = Callable[[str, date, date], Awaitable[pd.DataFrame]]


def _cache_path(ticker: str, day: date) -> Path:
    return CACHE_DIR / f"{ticker.upper()}_{day.isoformat()}.parquet"


async def _default_fetcher(ticker: str, from_date: date, to_date: date) -> pd.DataFrame:
    from src.data.polygon_client import PolygonClient
    return await PolygonClient().get_intraday_aggs(ticker, from_date, to_date)


async def get_intraday_day(
    ticker: str,
    day: date,
    *,
    fetcher: Fetcher | None = None,
    no_cache: bool = False,
) -> pd.DataFrame:
    """Return the 1-minute bars for one ticker on one day, disk-cached.

    Empty results are cached (as an empty frame) so non-trading days aren't
    refetched. Returns an empty frame with the standard columns when there is no
    data.
    """
    path = _cache_path(ticker, day)
    if not no_cache and path.exists():
        return pd.read_parquet(path)

    fetch = fetcher or _default_fetcher
    df = await fetch(ticker, day, day)
    if df is None or df.empty:
        df = pd.DataFrame(columns=_EMPTY_COLS)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


async def get_intraday_window(
    ticker: str,
    days: list[date],
    *,
    fetcher: Fetcher | None = None,
    no_cache: bool = False,
) -> pd.DataFrame:
    """Return concatenated minute bars for a ticker across several days.

    Fetches each day through the per-day cache, so repeated backtests reuse
    everything already on disk. Days are fetched in order; result is sorted by
    datetime.
    """
    frames = [
        await get_intraday_day(ticker, d, fetcher=fetcher, no_cache=no_cache)
        for d in days
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=_EMPTY_COLS)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("datetime").reset_index(drop=True)
