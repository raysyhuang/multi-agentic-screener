"""Quality veto layer — pre-ranking filters for extended tape, dilution, and data sanity.

Three look-ahead-safe vetoes that fire BEFORE ranking and mark candidates with
skip_reason for shadow tracking. All vetoes fail open (do not veto if data is
missing or insufficient).

Shadow-only by default: vetoes mark candidates but do NOT remove them from the
official pick stream unless explicitly enabled (future work, requires
selected-vs-selected evidence per §0.4).

See: outputs/research/quality_veto_FINDINGS.md
STRATEGY_REVIEW §0.4: gate-blocked picks persisted with skip_reason
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# Skip reason labels for the three vetoes. These are persisted into
# Outcome.skip_reason (String(30)) when a candidate is vetoed in shadow mode.
VETO_EXTENDED = "veto_extended"
VETO_DILUTION = "veto_dilution"
VETO_DATA_SANITY = "veto_data_sanity"


@dataclass
class VetoResult:
    """Result of running the veto layer on a single candidate."""
    ticker: str
    vetoed: bool
    veto_reason: str | None = None  # One of VETO_EXTENDED, VETO_DILUTION, VETO_DATA_SANITY
    veto_detail: str | None = None  # Human-readable detail for logging


def _valid(x) -> bool:
    """Check if a value is a valid, finite number (catches None, NaN, inf)."""
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def veto_extended_tape(
    ticker: str,
    df: pd.DataFrame,
    lookback: int = 20,
    cushion_atr_frac: float = 0.1,
) -> VetoResult:
    """Veto if the ticker is trading at/near its recent high (extended tape).

    Look-ahead safe: uses only the ticker's own daily OHLCV history.

    Logic:
        A name is extended if close >= (20d high - cushion), where cushion is
        a small fraction of ATR to avoid hair-trigger vetoes on minor noise.

    Args:
        ticker: Ticker symbol
        df: OHLCV DataFrame (must include 'close', 'high', 'low' columns)
        lookback: Days to look back for the high (default 20)
        cushion_atr_frac: ATR cushion fraction (default 0.1 = 10% of ATR)

    Returns:
        VetoResult with vetoed=True if extended, else vetoed=False

    Fail-open cases:
        - df is None, empty, or has fewer than (lookback + 14) bars
        - close/high data is missing or invalid
    """
    # Fail open: insufficient data
    if df is None or df.empty or len(df) < (lookback + 14):
        logger.debug("Extended veto: %s failed open (insufficient data, len=%d)", ticker, len(df) if df is not None else 0)
        return VetoResult(ticker=ticker, vetoed=False)

    try:
        close = df["close"].iloc[-1]
        recent_high = df["high"].tail(lookback).max()

        if not _valid(close) or not _valid(recent_high):
            logger.debug("Extended veto: %s failed open (invalid close or high)", ticker)
            return VetoResult(ticker=ticker, vetoed=False)

        # Compute ATR(14) for cushion
        # ATR = average of true range over 14 days
        # True range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close_series = df["close"].astype(float)
        prev_close = close_series.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = true_range.tail(14).mean()

        if not _valid(atr_14) or atr_14 <= 0:
            # Fail open if ATR is invalid/zero
            logger.debug("Extended veto: %s failed open (invalid ATR)", ticker)
            return VetoResult(ticker=ticker, vetoed=False)

        cushion = atr_14 * cushion_atr_frac
        threshold = recent_high - cushion

        if close >= threshold:
            pct_from_high = ((recent_high - close) / close) * 100
            detail = f"Close ${close:.2f} at {pct_from_high:.1f}% from {lookback}d high ${recent_high:.2f}"
            logger.info("Extended veto: %s VETOED (%s)", ticker, detail)
            return VetoResult(
                ticker=ticker,
                vetoed=True,
                veto_reason=VETO_EXTENDED,
                veto_detail=detail,
            )

        return VetoResult(ticker=ticker, vetoed=False)

    except Exception as e:  # noqa: BLE001
        # Fail open on any computation error (intentional broad catch)
        logger.warning("Extended veto: %s failed open (error: %s)", ticker, e)
        return VetoResult(ticker=ticker, vetoed=False)


def veto_dilution(
    ticker: str,
    fundamental_data: dict | None,
    dilution_threshold: float = 2.0,
    lookback_years: int = 1,
) -> VetoResult:
    """Veto if shares outstanding increased dramatically (dilution shock).

    Look-ahead safe: uses historical fundamental data that was already reported.

    Logic:
        Veto if (shares_now / shares_1y_ago) > dilution_threshold.
        Default threshold 2.0 = shares doubled or more YoY.

    Args:
        ticker: Ticker symbol
        fundamental_data: Dictionary with 'profile' and/or 'ratios' keys
        dilution_threshold: Ratio threshold (default 2.0 = 2x shares)
        lookback_years: How far back to compare (default 1 year)

    Returns:
        VetoResult with vetoed=True if diluted, else vetoed=False

    Fail-open cases:
        - fundamental_data is None or missing share count fields
        - Only current shares available (no historical comparison)
        - Share count is zero or invalid

    Note:
        Share count fields vary by provider. Common fields:
        - profile: sharesOutstanding, shares
        - ratios: weightedAverageShsOut, weightedAverageShsOutDil
    """
    # Fail open: no fundamental data
    if not fundamental_data:
        logger.debug("Dilution veto: %s failed open (no fundamental data)", ticker)
        return VetoResult(ticker=ticker, vetoed=False)

    # Try to extract current shares outstanding from profile
    profile = fundamental_data.get("profile", {})
    ratios = fundamental_data.get("ratios", {})

    # Current shares: try multiple fields
    shares_now = None
    for field in ["sharesOutstanding", "shares"]:
        if _valid(profile.get(field)):
            shares_now = float(profile[field])
            break

    # If profile doesn't have it, try ratios (most recent)
    if shares_now is None:
        if isinstance(ratios, list) and len(ratios) > 0:
            # ratios might be a list of historical records
            latest = ratios[0]
            for field in ["weightedAverageShsOut", "weightedAverageShsOutDil"]:
                if _valid(latest.get(field)):
                    shares_now = float(latest[field])
                    break
        elif isinstance(ratios, dict):
            # or a single dict
            for field in ["weightedAverageShsOut", "weightedAverageShsOutDil"]:
                if _valid(ratios.get(field)):
                    shares_now = float(ratios[field])
                    break

    if shares_now is None or shares_now <= 0:
        logger.debug("Dilution veto: %s failed open (no valid current shares)", ticker)
        return VetoResult(ticker=ticker, vetoed=False)

    # Historical shares: typically ratios are time-series
    # For now, fail open if we don't have historical data
    # TODO: This needs actual historical share count data, which may require
    # fetching historical financials. For MVP, we fail open if historical
    # comparison is not available in the already-fetched data.

    # If ratios is a list, look for an entry ~1 year ago
    shares_past = None
    if isinstance(ratios, list) and len(ratios) > 4:
        # Assume quarterly data, so 4 quarters back ≈ 1 year
        past_record = ratios[min(4, len(ratios) - 1)]
        for field in ["weightedAverageShsOut", "weightedAverageShsOutDil"]:
            if _valid(past_record.get(field)):
                shares_past = float(past_record[field])
                break

    if shares_past is None or shares_past <= 0:
        logger.debug("Dilution veto: %s failed open (no valid historical shares for comparison)", ticker)
        return VetoResult(ticker=ticker, vetoed=False)

    # Compute dilution ratio
    dilution_ratio = shares_now / shares_past

    if dilution_ratio > dilution_threshold:
        detail = f"Shares {dilution_ratio:.2f}x over {lookback_years}y ({shares_past/1e6:.1f}M → {shares_now/1e6:.1f}M)"
        logger.info("Dilution veto: %s VETOED (%s)", ticker, detail)
        return VetoResult(
            ticker=ticker,
            vetoed=True,
            veto_reason=VETO_DILUTION,
            veto_detail=detail,
        )

    return VetoResult(ticker=ticker, vetoed=False)


def veto_data_sanity(
    ticker: str,
    snapshot_a: dict | None,
    snapshot_b: dict | None,
    tolerance_pct: float = 0.10,
) -> VetoResult:
    """Veto if two data snapshots disagree on key metrics (revenue, shares, EPS).

    Look-ahead safe: compares two already-provided snapshots (e.g., FMP vs Massive,
    or FMP profile vs FMP ratios).

    Logic:
        Veto if both snapshots have the same metric and they disagree by more than
        tolerance_pct (default 10%). Checks revenue, shares outstanding, and EPS.

    Args:
        ticker: Ticker symbol
        snapshot_a: First data snapshot (e.g., FMP profile)
        snapshot_b: Second data snapshot (e.g., FMP ratios or Massive snapshot)
        tolerance_pct: Relative tolerance (default 0.10 = 10%)

    Returns:
        VetoResult with vetoed=True if data is inconsistent, else vetoed=False

    Fail-open cases:
        - Either snapshot is None
        - Metrics are missing in one or both snapshots
        - Metrics are zero or invalid (can't compute relative difference)

    Note:
        Common field mappings:
        - Revenue: revenue, totalRevenue, revenuePerShare (need conversion)
        - Shares: sharesOutstanding, shares, weightedAverageShsOut
        - EPS: eps, epsActual, epsDiluted
    """
    # Fail open: missing snapshots
    if not snapshot_a or not snapshot_b:
        logger.debug("Data sanity veto: %s failed open (missing snapshot)", ticker)
        return VetoResult(ticker=ticker, vetoed=False)

    # Check key metrics for disagreement
    metrics_to_check = [
        # (field_a, field_b, metric_name)
        ("revenue", "revenue", "revenue"),
        ("totalRevenue", "totalRevenue", "revenue"),
        ("sharesOutstanding", "sharesOutstanding", "shares"),
        ("shares", "shares", "shares"),
        ("sharesOutstanding", "weightedAverageShsOut", "shares"),
        ("eps", "eps", "EPS"),
        ("epsActual", "eps", "EPS"),
    ]

    for field_a, field_b, metric_name in metrics_to_check:
        val_a = snapshot_a.get(field_a)
        val_b = snapshot_b.get(field_b)

        if not _valid(val_a) or not _valid(val_b):
            continue

        val_a = float(val_a)
        val_b = float(val_b)

        # Skip if either is zero (can't compute relative diff)
        if val_a == 0 or val_b == 0:
            continue

        # Compute relative difference
        rel_diff = abs(val_a - val_b) / max(abs(val_a), abs(val_b))

        if rel_diff > tolerance_pct:
            detail = f"{metric_name} mismatch: A={val_a:.2e} vs B={val_b:.2e} ({rel_diff*100:.1f}% diff)"
            logger.info("Data sanity veto: %s VETOED (%s)", ticker, detail)
            return VetoResult(
                ticker=ticker,
                vetoed=True,
                veto_reason=VETO_DATA_SANITY,
                veto_detail=detail,
            )

    return VetoResult(ticker=ticker, vetoed=False)


def apply_veto_layer(
    signals: list,
    price_data: dict[str, pd.DataFrame],
    fundamental_data_by_ticker: dict[str, dict],
    shadow_only: bool = True,
) -> tuple[list, list[VetoResult]]:
    """Apply all vetoes to a list of signals.

    Args:
        signals: List of signal objects (MeanReversionSignal, SniperSignal, etc.)
        price_data: Dict of ticker -> OHLCV DataFrame
        fundamental_data_by_ticker: Dict of ticker -> fundamental data dict
        shadow_only: If True, vetoed signals are kept but marked. If False, they
                     are removed from the output list. (default True = shadow mode)

    Returns:
        (filtered_signals, veto_results): Filtered signal list and full veto results
    """
    veto_results = []
    filtered_signals = []

    for sig in signals:
        ticker = sig.ticker
        df = price_data.get(ticker)
        fundamental_data = fundamental_data_by_ticker.get(ticker)

        # Run all three vetoes
        extended = veto_extended_tape(ticker, df)
        dilution = veto_dilution(ticker, fundamental_data)
        # For data sanity, we compare profile vs ratios (both from FMP)
        profile = fundamental_data.get("profile", {}) if fundamental_data else {}
        ratios = fundamental_data.get("ratios", {}) if fundamental_data else {}
        # If ratios is a list (time-series), use the most recent entry
        if isinstance(ratios, list) and len(ratios) > 0:
            ratios = ratios[0]
        elif not isinstance(ratios, dict):
            ratios = {}
        data_sanity = veto_data_sanity(ticker, profile, ratios)

        # Determine if this signal is vetoed
        vetoed = extended.vetoed or dilution.vetoed or data_sanity.vetoed
        veto_reason = None
        if extended.vetoed:
            veto_reason = VETO_EXTENDED
            veto_results.append(extended)
        if dilution.vetoed:
            veto_reason = VETO_DILUTION
            veto_results.append(dilution)
        if data_sanity.vetoed:
            veto_reason = VETO_DATA_SANITY
            veto_results.append(data_sanity)

        if not vetoed:
            # No veto, keep the signal
            veto_results.append(VetoResult(ticker=ticker, vetoed=False))
            filtered_signals.append(sig)
        else:
            # Signal is vetoed
            if shadow_only:
                # Shadow mode: keep the signal but mark it with veto_reason
                # The caller will persist it with skip_reason=veto_reason
                sig.veto_reason = veto_reason  # Attach veto reason to signal object
                filtered_signals.append(sig)
            else:
                # Hard veto: remove from output
                logger.info("Veto layer: %s REMOVED (reason: %s)", ticker, veto_reason)

    vetoed_count = sum(1 for r in veto_results if r.vetoed)
    logger.info(
        "Veto layer: %d/%d signals vetoed (shadow_only=%s)",
        vetoed_count, len(signals), shadow_only
    )

    return filtered_signals, veto_results
