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


_TRUE_TOKENS = frozenset({"true", "1", "yes", "y", "t"})
_FALSE_TOKENS = frozenset({"false", "0", "no", "n", "f"})


def _as_bool(value: object) -> tuple[bool, bool]:
    """Coerce a provider's boolean-ish flag. Returns (value, recognised).

    JSON booleans, the strings "true"/"false", and 0/1 are all encodings a
    provider may use, and they are not interchangeable in Python: `bool("false")`
    is True. A gate that reads such a flag raw flips to rejecting everything the
    moment the encoding changes.

    ``None`` and ``""`` are NOT false — they are *unknown*. Treating them as a
    recognised false is how a provider quietly dropping the field would admit
    products with neither an exclusion nor a warning: the gate would report
    itself healthy while evaluating nothing. Only an explicit false-like value
    counts as false.

    An unrecognised value returns ``(False, False)`` — do not exclude, but say
    so. For this gate that is the safe direction: wrongly admitting an ETF costs
    one bad candidate, wrongly excluding everything costs the whole day's
    universe. The caller counts the unrecognised ones so the condition surfaces
    in the funnel instead of being absorbed.
    """
    if isinstance(value, bool):
        return value, True
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value), True
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True, True
        if token in _FALSE_TOKENS:
            return False, True
    return False, False


def _fund_flags(stock: dict) -> tuple[bool, bool, bool]:
    """Read isEtf/isFund from a row. Returns (is_etf, is_fund, evaluated).

    Whether an absent flag is a problem depends on which provider shaped the
    row. The Polygon builder always sets ``type`` (to ``""`` for common stock,
    since its query is already restricted to CS) and never sets these flags —
    there, absence is correct and the ``type`` field is authoritative. An
    FMP-shaped row has no ``type`` key at all, so the flags are the only thing
    standing between an ETF and the universe; absent or empty there means the
    gate evaluated nothing and must say so.
    """
    polygon_shaped = "type" in stock

    is_etf, etf_known = _as_bool(stock.get("isEtf"))
    is_fund, fund_known = _as_bool(stock.get("isFund"))

    if polygon_shaped and "isEtf" not in stock and "isFund" not in stock:
        return is_etf, is_fund, True  # `type` carries the decision

    return is_etf, is_fund, etf_known and fund_known


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
    # Rows whose isEtf/isFund arrived in an encoding we do not recognise. These
    # are NOT excluded (see _as_bool); a non-zero count means the provider
    # changed shape and the ETF gate is running blind on those rows.
    unrecognized_type_flags: int = 0
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
        if self.unrecognized_type_flags:
            logger.warning(
                "Filter funnel: %d rows had unrecognised isEtf/isFund encodings — "
                "the ETF gate did not evaluate them. Provider shape may have changed.",
                self.unrecognized_type_flags,
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
            "unrecognized_type_flags": self.unrecognized_type_flags,
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

        # Type exclusion.
        #
        # `type` is the Polygon-shaped field. The FMP screener — the PRIMARY
        # universe source — does not return it at all; it reports `isEtf` /
        # `isFund` booleans instead. So for six months this gate read None on
        # every FMP row, `stock_type` was "", and nothing was ever excluded.
        # TQQQ (a 3x leveraged ETF, beta 3.7) reached the official candidate
        # pool with a sniper score of 97.5 — leveraged products clear the
        # sniper's ATR% >= 5 floor structurally, and their "relative strength
        # vs SPY" is leveraged beta, not the idiosyncratic strength the signal
        # is trying to measure.
        # The flags are coerced, never read for raw truthiness: JSON `false` and
        # the STRING "false" are both plausible encodings, and `bool("false")` is
        # True. Reading them raw would drop every FMP row on the day the provider
        # changed encoding — a silent zero-universe run, which is far worse than
        # admitting an ETF. Unrecognised values are counted and reported rather
        # than silently deciding either way.
        is_etf, is_fund, evaluated = _fund_flags(stock)
        if not evaluated:
            funnel.unrecognized_type_flags += 1

        if any(t in stock_type for t in EXCLUDED_TYPES) or is_etf or is_fund:
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
