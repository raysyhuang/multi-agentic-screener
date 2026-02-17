"""Momentum breakout signal model.

Identifies stocks breaking out of consolidation on high volume with
technical momentum confirmation. Ported from KooCore-D multi-factor scoring.

Fires on day T close, execution at T+1 open.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


def _valid(x) -> bool:
    """Check if a value is a valid, finite number (catches None, NaN, inf)."""
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


@dataclass
class BreakoutSignal:
    ticker: str
    score: float  # 0-100 composite score
    direction: str  # LONG
    entry_price: float  # expected T+1 open (use latest close as proxy)
    stop_loss: float
    target_1: float
    target_2: float
    holding_period: int
    components: dict  # individual factor scores


def score_breakout(ticker: str, df: pd.DataFrame, features: dict) -> BreakoutSignal | None:
    """Score a ticker for momentum breakout potential.

    Multi-factor model:
      1. Technical momentum (30%): RSI, MACD, price vs MAs
      2. Volume confirmation (25%): RVOL, volume surge
      3. Consolidation breakout (20%): tight range → expansion
      4. Trend alignment (15%): price above key MAs
      5. Volatility context (10%): ATR position sizing
    """
    if df.empty or len(df) < 50:
        return None

    scores = {}

    # --- 1. Technical Momentum (30%) ---
    rsi = features.get("rsi_14")
    roc_5 = features.get("roc_5")
    roc_10 = features.get("roc_10")

    momentum_score = 0.0
    if _valid(rsi):
        if 55 <= rsi <= 75:  # strong but not overbought
            momentum_score += 40
        elif 50 <= rsi < 55:
            momentum_score += 20
        elif rsi > 75:
            momentum_score -= 10  # overbought penalty

    if _valid(roc_5) and roc_5 > 0:
        momentum_score += min(30, roc_5 * 5)
    if _valid(roc_10) and roc_10 > 0:
        momentum_score += min(30, roc_10 * 3)

    scores["momentum"] = max(0, min(100, momentum_score))

    # --- 2. Volume Confirmation (25%) ---
    rvol = features.get("rvol")
    volume_surge = features.get("volume_surge", 0)

    volume_score = 0.0
    if _valid(rvol):
        if rvol >= 3.0:
            volume_score = 100
        elif rvol >= 2.0:
            volume_score = 80
        elif rvol >= 1.5:
            volume_score = 60
        elif rvol >= 1.2:
            volume_score = 40
        else:
            volume_score = 20  # below-average volume = weak confirmation

    if volume_surge:
        volume_score = min(100, volume_score + 20)

    # Gap breakouts with high volume = highest conviction
    if features.get("is_gap_up", 0) and _valid(rvol) and rvol >= 2.0:
        volume_score = min(100, volume_score + 15)

    scores["volume"] = volume_score

    # --- 3. Consolidation Breakout (20%) ---
    is_consolidating = features.get("is_consolidating", 0)
    near_20d_high = features.get("near_20d_high", 0)

    breakout_score = 0.0
    if near_20d_high:
        breakout_score += 60
    if is_consolidating:
        breakout_score += 40  # was tight, now breaking out

    # Check if today broke out of the consolidation range
    close = features.get("close", 0)
    high_20d = features.get("high_20d")
    if close and high_20d and close >= high_20d:
        breakout_score = min(100, breakout_score + 30)

    # Gap-up breakout bonus (Humbled Trader)
    is_gap_up = features.get("is_gap_up", 0)
    gap_pct = features.get("gap_pct")
    if is_gap_up and _valid(gap_pct):
        breakout_score += 25

    scores["breakout"] = min(100, breakout_score)

    # --- 4. Trend Alignment (15%) ---
    pct_above_sma20 = features.get("pct_above_sma20", 0)
    pct_above_sma50 = features.get("pct_above_sma50", 0)
    sma_20_slope = features.get("sma_20_slope")

    trend_score = 0.0
    if _valid(pct_above_sma20) and pct_above_sma20 > 0:
        trend_score += 35
    if _valid(pct_above_sma50) and pct_above_sma50 > 0:
        trend_score += 35

    # Rising 20 SMA confirms momentum direction (Emmanuel)
    if _valid(sma_20_slope) and sma_20_slope > 0.5:
        trend_score += 30

    scores["trend"] = min(100, trend_score)

    # --- 5. Volatility Context (10%) ---
    atr_pct = features.get("atr_pct")
    volatility_score = 50  # neutral default
    if _valid(atr_pct):
        if 2.0 <= atr_pct <= 5.0:  # sweet spot for short-term trades
            volatility_score = 80
        elif atr_pct < 1.5:
            volatility_score = 30  # too quiet
        elif atr_pct > 8.0:
            volatility_score = 20  # too wild

    scores["volatility"] = volatility_score

    # --- Composite ---
    weights = {
        "momentum": 0.30,
        "volume": 0.25,
        "breakout": 0.20,
        "trend": 0.15,
        "volatility": 0.10,
    }
    composite = sum(scores[k] * weights[k] for k in weights)

    # Minimum threshold
    if composite < 50:
        return None

    # --- Position sizing via ATR ---
    close_price = features.get("close", 0)
    atr = features.get("atr_14")
    if not _valid(close_price) or close_price <= 0:
        return None
    # Fallback ATR: 2% of price; enforce minimum floor of 0.5% of price
    if not _valid(atr) or atr <= 0:
        atr = close_price * 0.02
    atr = max(atr, close_price * 0.005)

    stop_loss = close_price - 2.0 * atr
    target_1 = close_price + 2.0 * atr  # 2:1 R:R
    target_2 = close_price + 3.0 * atr  # 3:1 R:R

    return BreakoutSignal(
        ticker=ticker,
        score=round(composite, 2),
        direction="LONG",
        entry_price=round(close_price, 2),
        stop_loss=round(stop_loss, 2),
        target_1=round(target_1, 2),
        target_2=round(target_2, 2),
        holding_period=10,
        components=scores,
    )
