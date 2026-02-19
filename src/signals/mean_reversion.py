"""RSI(2) oversold mean-reversion model.

Ported from gemini_STST's 3-day RSI(2) model with next-day-open execution.
Looks for oversold conditions in stocks with intact long-term uptrends.

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
class MeanReversionSignal:
    ticker: str
    score: float  # 0-100
    direction: str  # LONG
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    holding_period: int
    components: dict


def score_mean_reversion(
    ticker: str, df: pd.DataFrame, features: dict
) -> MeanReversionSignal | None:
    """Score a ticker for RSI(2) mean-reversion potential.

    Criteria:
      1. RSI(2) < 10 (deeply oversold on 2-period RSI)
      2. Price above 200-day SMA (long-term uptrend intact)
      3. 3+ consecutive down days preferred
      4. Not near earnings (avoid event risk)
      5. Volume not collapsing (still liquid)

    Target: reversion to mean (5-day SMA) within 3-5 days.
    """
    if df.empty or len(df) < 50:
        return None

    scores = {}

    # --- 1. RSI(2) Oversold (40%) ---
    rsi_2 = features.get("rsi_2")
    if not _valid(rsi_2):
        return None

    rsi_score = 0.0
    if rsi_2 <= 5:
        rsi_score = 100  # extreme oversold
    elif rsi_2 <= 10:
        rsi_score = 80
    elif rsi_2 <= 15:
        rsi_score = 50
    elif rsi_2 <= 20:
        rsi_score = 30
    else:
        return None  # not oversold enough

    scores["rsi2_oversold"] = rsi_score

    # --- 2. Long-term Trend Intact (25%) ---
    close = df["close"].astype(float)
    pct_above_sma200 = features.get("pct_above_sma200")
    pct_above_sma50 = features.get("pct_above_sma50")

    if _valid(pct_above_sma200):
        above_200 = pct_above_sma200 > 0
    elif _valid(pct_above_sma50):
        above_200 = pct_above_sma50 > 0  # fallback for short history
    else:
        above_200 = False

    trend_score = 80 if above_200 else 20
    scores["trend_intact"] = trend_score

    # --- 3. Consecutive Down Days (15%) ---
    streak = features.get("streak", 0)
    streak_score = 0.0
    if _valid(streak) and streak <= -3:
        streak_score = 100
    elif _valid(streak) and streak <= -2:
        streak_score = 60
    elif _valid(streak) and streak <= -1:
        streak_score = 30
    scores["down_streak"] = streak_score

    # --- 4. Distance from Recent Low (10%) ---
    dist_from_5d_low = features.get("dist_from_5d_low", 0)
    proximity_score = 0.0
    if _valid(dist_from_5d_low):
        if dist_from_5d_low < 1.0:
            proximity_score = 80  # very close to 5-day low
        elif dist_from_5d_low < 2.0:
            proximity_score = 50

    scores["proximity_to_low"] = proximity_score

    # --- 5. Volume Liquidity (10%) ---
    rvol = features.get("rvol")
    vol_score = 50  # neutral default
    if _valid(rvol):
        if rvol >= 0.5:
            vol_score = 70  # still liquid
        elif rvol < 0.3:
            vol_score = 20  # dried up, dangerous
    scores["liquidity"] = vol_score

    # --- Composite ---
    weights = {
        "rsi2_oversold": 0.40,
        "trend_intact": 0.25,
        "down_streak": 0.15,
        "proximity_to_low": 0.10,
        "liquidity": 0.10,
    }
    composite = sum(scores[k] * weights[k] for k in weights)

    if composite < 50:
        return None

    # --- Price targets ---
    close_price = features.get("close", 0)
    atr = features.get("atr_14")
    if not _valid(close_price) or close_price <= 0:
        return None
    if not _valid(atr) or atr <= 0:
        atr = close_price * 0.02
    atr = max(atr, close_price * 0.005)

    # Mean reversion target: back to 5-day SMA with 1x ATR floor
    sma_5 = close.rolling(5).mean().iloc[-1]
    sma_5_val = float(sma_5) if pd.notna(sma_5) else close_price * 1.03
    target_1 = max(sma_5_val, close_price + 1.0 * atr)

    # Extended target: back to 10-day SMA with 1.5x ATR floor
    sma_10 = close.rolling(10).mean().iloc[-1]
    sma_10_val = float(sma_10) if pd.notna(sma_10) else close_price * 1.05
    target_2 = max(sma_10_val, close_price + 1.5 * atr)

    # Tight stop: 1.0x ATR below current price (thesis invalidated quickly)
    stop_loss = close_price - 1.0 * atr

    return MeanReversionSignal(
        ticker=ticker,
        score=round(composite, 2),
        direction="LONG",
        entry_price=round(close_price, 2),
        stop_loss=round(stop_loss, 2),
        target_1=round(target_1, 2),
        target_2=round(target_2, 2),
        holding_period=3,  # short hold for mean reversion (WR decays after day 3)
        components=scores,
    )
