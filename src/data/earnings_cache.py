"""Point-in-time earnings cache (FMP EPS actual vs estimate, dated).

For post-earnings-drift research and any event-based signal. Pulls each ticker's
historical earnings once and caches to disk (JSON), so repeated backtests don't
re-hit FMP. The fetcher is injectable for unit tests (no network/key needed).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable

CACHE_DIR = Path("data/cache/earnings")

Fetcher = Callable[[str], Awaitable[list[dict]]]


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.json"


async def _default_fetcher(ticker: str) -> list[dict]:
    from src.data.fmp_client import FMPClient
    return await FMPClient().get_earnings_surprise(ticker)


async def get_earnings(
    ticker: str,
    *,
    fetcher: Fetcher | None = None,
    no_cache: bool = False,
) -> list[dict]:
    """Return a ticker's historical earnings rows (date, epsActual, epsEstimated, …).

    Disk-cached; a fetch that returns nothing is cached as [] so it isn't retried
    every run. Any fetch error returns [] WITHOUT caching (so it retries later).
    """
    path = _cache_path(ticker)
    if not no_cache and path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass

    fetch = fetcher or _default_fetcher
    try:
        rows = await fetch(ticker)
    except Exception:
        return []
    if not isinstance(rows, list):
        rows = []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows
