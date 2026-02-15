"""Universe filtering — gates stocks by price, volume, liquidity, and exclusions."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import get_settings

logger = logging.getLogger(__name__)

# Exclude problematic categories
EXCLUDED_SUFFIXES = {".W", ".U", ".R"}  # warrants, units, rights
EXCLUDED_TYPES = {"ETF", "ETN", "FUND", "REIT"}


def filter_universe(candidates: list[dict]) -> list[dict]:
    """Apply universe gate filters.

    Criteria:
      - min price ($5)
      - min average daily volume (500K shares)
      - exclude warrants, units, rights
      - exclude ETFs/ETNs (we trade individual stocks)
      - must be on NYSE or NASDAQ
    """
    settings = get_settings()
    passed = []

    for stock in candidates:
        ticker = stock.get("symbol", "")
        price = stock.get("price") or stock.get("lastPrice") or 0
        volume = stock.get("volume") or stock.get("avgVolume") or 0
        exchange = (stock.get("exchangeShortName") or stock.get("exchange") or "").upper()
        stock_type = (stock.get("type") or "").upper()

        # Price gate
        if price < settings.min_price:
            continue

        # Volume gate
        if volume < settings.min_avg_daily_volume:
            continue

        # Exchange gate
        if exchange not in ("NYSE", "NASDAQ"):
            continue

        # Suffix exclusion (warrants, units)
        if any(ticker.endswith(s) for s in EXCLUDED_SUFFIXES):
            continue

        # Type exclusion
        if any(t in stock_type for t in EXCLUDED_TYPES):
            continue

        # Ticker sanity: max 5 chars, no special characters
        if len(ticker) > 5 or not ticker.isalpha():
            continue

        passed.append(stock)

    logger.info("Universe filter: %d → %d candidates", len(candidates), len(passed))
    return passed


def filter_by_ohlcv(ticker: str, df: pd.DataFrame) -> bool:
    """Additional filters requiring OHLCV data.

    - Minimum 20 trading days of data
    - No extreme moves (>50% in 1 day = likely corporate action)
    - Minimum average dollar volume ($2M/day)
    """
    if df.empty or len(df) < 20:
        return False

    # Check for extreme moves (corporate actions, splits, etc.)
    daily_returns = df["close"].pct_change().abs()
    if daily_returns.max() > 0.50:
        logger.debug("Excluded %s: extreme daily move (%.1f%%)", ticker, daily_returns.max() * 100)
        return False

    # Minimum dollar volume
    avg_dollar_vol = (df["close"] * df["volume"]).tail(20).mean()
    if avg_dollar_vol < 2_000_000:
        return False

    return True
