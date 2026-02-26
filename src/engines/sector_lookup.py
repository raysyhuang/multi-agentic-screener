"""Lightweight sector lookup for cross-engine convergence analysis.

Uses a local JSON map for fast, no-API lookups. Falls back to yfinance
for tickers not in the map (cached on first lookup per session).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SECTOR_MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "sector_map.json"
_sector_cache: dict[str, str] = {}
_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    try:
        with open(_SECTOR_MAP_PATH) as f:
            _sector_cache.update(json.load(f))
        logger.debug("Loaded %d tickers from sector map", len(_sector_cache))
    except FileNotFoundError:
        logger.warning("Sector map not found at %s", _SECTOR_MAP_PATH)
    _loaded = True


def get_sector(ticker: str) -> str:
    """Return GICS sector for a ticker, or 'Unknown' if not found."""
    _ensure_loaded()
    upper = ticker.upper()
    if upper in _sector_cache:
        return _sector_cache[upper]

    # Fallback: try yfinance (slow, cached per session)
    try:
        import yfinance as yf
        info = yf.Ticker(upper).info
        sector = info.get("sector", "Unknown")
        _sector_cache[upper] = sector
        return sector
    except Exception:
        _sector_cache[upper] = "Unknown"
        return "Unknown"


def enrich_picks_with_sector(picks: list[dict]) -> list[dict]:
    """Add 'sector' field to each pick dict in-place. Returns the same list."""
    _ensure_loaded()
    for pick in picks:
        ticker = pick.get("ticker", "")
        pick["sector"] = get_sector(ticker)
    return picks
