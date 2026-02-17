"""Market regime classifier — gates all signals by current market conditions.

Regime types:
  - BULL: trending up, low volatility, broad participation
  - BEAR: trending down, high volatility, narrow breadth
  - CHOPPY: range-bound, mixed signals

This is the single highest-ROI feature in the system. A momentum breakout
signal firing in a bear/choppy regime will destroy returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.config import get_settings


class Regime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    CHOPPY = "choppy"


@dataclass
class RegimeAssessment:
    regime: Regime
    confidence: float  # 0-1
    vix_level: float | None
    spy_trend: str  # "above_sma20", "below_sma20"
    qqq_trend: str
    breadth_score: float | None  # % of stocks above 20-day SMA
    yield_spread: float | None
    details: dict


def compute_breadth_score(price_data: dict[str, pd.DataFrame]) -> float | None:
    """Compute market breadth: % of tickers with close above their 20-day SMA.

    Args:
        price_data: Dict of ticker -> OHLCV DataFrame with 'close' column.

    Returns:
        Breadth score 0.0-1.0, or None if insufficient data.
    """
    if not price_data:
        return None

    above = 0
    total = 0
    for ticker, df in price_data.items():
        if df is None or df.empty or len(df) < 20:
            continue
        close = df["close"].astype(float)
        sma20 = close.rolling(20).mean().iloc[-1]
        if pd.isna(sma20):
            continue
        total += 1
        if close.iloc[-1] > sma20:
            above += 1

    if total < 10:
        return None

    return round(above / total, 4)


def classify_regime(
    spy_df: pd.DataFrame,
    qqq_df: pd.DataFrame,
    vix: float | None = None,
    yield_spread: float | None = None,
    breadth_score: float | None = None,
) -> RegimeAssessment:
    """Classify current market regime using SPY/QQQ trend + VIX + breadth.

    Baseline logic ported from gemini_STST's 20-day SMA regime detection,
    extended with VIX thresholds and yield curve context.
    """
    settings = get_settings()
    signals = {}

    # --- SPY trend ---
    spy_trend = _classify_trend(spy_df)
    signals["spy_trend"] = spy_trend

    # --- QQQ trend ---
    qqq_trend = _classify_trend(qqq_df)
    signals["qqq_trend"] = qqq_trend

    # --- VIX regime ---
    vix_signal = "neutral"
    if vix is not None:
        if vix >= settings.vix_high_threshold:
            vix_signal = "fear"
        elif vix <= settings.vix_low_threshold:
            vix_signal = "complacent"
        else:
            vix_signal = "neutral"
    signals["vix"] = vix_signal

    # --- Yield curve ---
    yield_signal = "normal"
    if yield_spread is not None:
        if yield_spread < 0:
            yield_signal = "inverted"
        elif yield_spread < 0.5:
            yield_signal = "flat"
    signals["yield_curve"] = yield_signal

    # --- Breadth ---
    breadth_signal = "neutral"
    if breadth_score is not None:
        settings_obj = get_settings()
        if breadth_score >= settings_obj.breadth_bullish_threshold:
            breadth_signal = "broad"
        elif breadth_score <= settings_obj.breadth_bearish_threshold:
            breadth_signal = "narrow"
    signals["breadth"] = breadth_signal
    signals["breadth_score"] = breadth_score

    # --- Composite scoring ---
    bull_score = 0.0
    bear_score = 0.0

    # SPY/QQQ alignment (highest weight)
    if spy_trend == "above_sma20" and qqq_trend == "above_sma20":
        bull_score += 2.0
    elif spy_trend == "below_sma20" and qqq_trend == "below_sma20":
        bear_score += 2.0
    else:
        # Divergence = choppy signal
        pass

    # SPY slope (are we trending or flat?)
    spy_slope = _compute_slope(spy_df)
    signals["spy_slope"] = spy_slope
    if spy_slope > 0.001:
        bull_score += 1.0
    elif spy_slope < -0.001:
        bear_score += 1.0

    # VIX
    if vix_signal == "complacent":
        bull_score += 0.5
    elif vix_signal == "fear":
        bear_score += 1.5  # VIX fear is heavily weighted

    # Yield curve
    if yield_signal == "inverted":
        bear_score += 0.5

    # Breadth
    if breadth_signal == "broad":
        bull_score += 1.0
    elif breadth_signal == "narrow":
        bear_score += 1.0

    # Determine regime
    total = bull_score + bear_score
    if total == 0:
        regime = Regime.CHOPPY
        confidence = 0.5
    elif bull_score > bear_score * 1.5:
        regime = Regime.BULL
        confidence = min(1.0, bull_score / (total + 1))
    elif bear_score > bull_score * 1.5:
        regime = Regime.BEAR
        confidence = min(1.0, bear_score / (total + 1))
    else:
        regime = Regime.CHOPPY
        confidence = 0.4

    return RegimeAssessment(
        regime=regime,
        confidence=round(confidence, 3),
        vix_level=vix,
        spy_trend=spy_trend,
        qqq_trend=qqq_trend,
        breadth_score=breadth_score,
        yield_spread=yield_spread,
        details=signals,
    )


def _classify_trend(df: pd.DataFrame) -> str:
    """Is the latest close above or below the 20-day SMA?"""
    if df.empty or len(df) < 20:
        return "unknown"

    close = df["close"].astype(float)
    sma_20 = close.rolling(20).mean()
    latest_close = close.iloc[-1]
    latest_sma = sma_20.iloc[-1]

    if pd.isna(latest_sma):
        return "unknown"

    return "above_sma20" if latest_close > latest_sma else "below_sma20"


def _compute_slope(df: pd.DataFrame, window: int = 20) -> float:
    """Compute normalized slope of close prices over the window."""
    if df.empty or len(df) < window:
        return 0.0

    close = df["close"].astype(float).iloc[-window:]
    x = np.arange(len(close))
    try:
        coeffs = np.polyfit(x, close.values, 1)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0
    slope = coeffs[0]
    # Normalize by price level
    mean_price = close.mean()
    if mean_price == 0 or pd.isna(mean_price):
        return 0.0
    return slope / mean_price


def get_regime_allowed_models(regime: Regime) -> list[str]:
    """Which signal models are allowed to fire in each regime.

    Key insight: momentum breakouts in bear markets destroy returns.
    """
    if regime == Regime.BULL:
        return ["breakout", "mean_reversion", "catalyst"]
    elif regime == Regime.BEAR:
        return ["mean_reversion"]  # Only counter-trend works in downtrends
    else:  # CHOPPY
        return ["mean_reversion", "catalyst"]  # No momentum, range plays only
