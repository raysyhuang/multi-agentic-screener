"""Universe filtering — gates stocks by price, volume, liquidity, and exclusions.

Includes funnel counters (ported from gemini_STST) that track how many tickers
are eliminated at each filter stage. Critical for debugging filter chains.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd

from src.config import get_settings

logger = logging.getLogger(__name__)

# Exclude problematic categories
EXCLUDED_SUFFIXES = {".W", ".U", ".R"}  # warrants, units, rights
EXCLUDED_TYPES = {"ETF", "ETN", "FUND", "REIT"}
_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")
_SPLIT_DROP_RATIOS = (0.50, 0.333, 0.25, 0.20)  # 2:1, 3:1, 4:1, 5:1
_SPLIT_RATIO_TOLERANCE = 0.05


def _is_valid_ticker(ticker: str) -> bool:
    """Allow normal US symbols, including class shares like BRK.B/BF-B."""
    if not ticker:
        return False
    return bool(_VALID_TICKER_RE.match(ticker.upper()))


@dataclass
class FilterFunnel:
    """Tracks how many tickers are eliminated at each filter stage.

    Enables debugging of filter chain effectiveness — e.g., if 80% of tickers
    fail the volume filter, the threshold may be too aggressive.
    """

    total_input: int = 0
    failed_price: int = 0
    failed_volume: int = 0
    failed_exchange: int = 0
    failed_suffix: int = 0
    failed_type: int = 0
    failed_ticker_format: int = 0
    passed: int = 0

    def log_summary(self) -> None:
        """Log filter funnel as a readable summary."""
        logger.info(
            "Filter funnel: %d input → %d passed | "
            "price=%d, volume=%d, exchange=%d, suffix=%d, type=%d, format=%d dropped",
            self.total_input,
            self.passed,
            self.failed_price,
            self.failed_volume,
            self.failed_exchange,
            self.failed_suffix,
            self.failed_type,
            self.failed_ticker_format,
        )

    def to_dict(self) -> dict:
        return {
            "total_input": self.total_input,
            "failed_price": self.failed_price,
            "failed_volume": self.failed_volume,
            "failed_exchange": self.failed_exchange,
            "failed_suffix": self.failed_suffix,
            "failed_type": self.failed_type,
            "failed_ticker_format": self.failed_ticker_format,
            "passed": self.passed,
        }


@dataclass
class OHLCVFunnel:
    """Tracks OHLCV filter chain results."""

    total_input: int = 0
    failed_insufficient_data: int = 0
    failed_extreme_move: int = 0
    failed_dollar_volume: int = 0
    passed: int = 0

    def log_summary(self) -> None:
        logger.info(
            "OHLCV funnel: %d input → %d passed | "
            "data=%d, extreme=%d, dollar_vol=%d dropped",
            self.total_input,
            self.passed,
            self.failed_insufficient_data,
            self.failed_extreme_move,
            self.failed_dollar_volume,
        )

    def to_dict(self) -> dict:
        return {
            "total_input": self.total_input,
            "failed_insufficient_data": self.failed_insufficient_data,
            "failed_extreme_move": self.failed_extreme_move,
            "failed_dollar_volume": self.failed_dollar_volume,
            "passed": self.passed,
        }


def filter_universe(
    candidates: list[dict],
    funnel: FilterFunnel | None = None,
) -> list[dict]:
    """Apply universe gate filters.

    Criteria:
      - min price ($5)
      - min average daily volume (500K shares)
      - exclude warrants, units, rights
      - exclude ETFs/ETNs (we trade individual stocks)
      - must be on NYSE or NASDAQ

    Returns filtered list. Optionally populates a FilterFunnel for diagnostics.
    """
    settings = get_settings()
    if funnel is None:
        funnel = FilterFunnel()
    funnel.total_input = len(candidates)

    passed = []

    for stock in candidates:
        ticker = stock.get("symbol", "")
        price = stock.get("price") or stock.get("lastPrice") or 0
        volume = stock.get("volume") or stock.get("avgVolume") or 0
        exchange = (stock.get("exchangeShortName") or stock.get("exchange") or "").upper()
        stock_type = (stock.get("type") or "").upper()

        # Price gate
        if price < settings.min_price:
            funnel.failed_price += 1
            continue

        # Volume gate
        if volume < settings.min_avg_daily_volume:
            funnel.failed_volume += 1
            continue

        # Exchange gate
        if exchange not in ("NYSE", "NASDAQ"):
            funnel.failed_exchange += 1
            continue

        # Suffix exclusion (warrants, units)
        if any(ticker.endswith(s) for s in EXCLUDED_SUFFIXES):
            funnel.failed_suffix += 1
            continue

        # Type exclusion
        if any(t in stock_type for t in EXCLUDED_TYPES):
            funnel.failed_type += 1
            continue

        # Ticker sanity
        if not _is_valid_ticker(ticker):
            funnel.failed_ticker_format += 1
            continue

        passed.append(stock)

    funnel.passed = len(passed)
    funnel.log_summary()
    return passed


def filter_by_ohlcv(
    ticker: str,
    df: pd.DataFrame,
    funnel: OHLCVFunnel | None = None,
) -> bool:
    """Additional filters requiring OHLCV data.

    - Minimum 20 trading days of data
    - No extreme moves (>50% in 1 day = likely corporate action)
    - Minimum average dollar volume ($2M/day)
    """
    if df is None or df.empty or len(df) < 20:
        if funnel:
            funnel.failed_insufficient_data += 1
        return False

    # Check for extreme moves (corporate actions, splits, etc.)
    daily_returns = df["close"].pct_change().abs()
    if daily_returns.max() > 0.50:
        logger.debug("Excluded %s: extreme daily move (%.1f%%)", ticker, daily_returns.max() * 100)
        if funnel:
            funnel.failed_extreme_move += 1
        return False

    # Detect likely unadjusted split artifacts in the recent window.
    recent_returns = daily_returns.tail(30).dropna()
    for abs_ret in recent_returns:
        if any(abs(abs_ret - ratio) < _SPLIT_RATIO_TOLERANCE for ratio in _SPLIT_DROP_RATIOS):
            logger.debug("Excluded %s: likely unadjusted split artifact (%.1f%% gap)", ticker, abs_ret * 100)
            if funnel:
                funnel.failed_extreme_move += 1
            return False

    # Minimum dollar volume
    avg_dollar_vol = (df["close"] * df["volume"]).tail(20).mean()
    if avg_dollar_vol < 2_000_000:
        if funnel:
            funnel.failed_dollar_volume += 1
        return False

    return True
