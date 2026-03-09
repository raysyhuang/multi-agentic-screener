"""Sniper track — concentrated high-velocity signal model.

Targets stocks that CAN move 7-10% (ATR% > 3.5%) in setups with energy
buildup (BB squeeze + volume compression). Fewer, bigger, higher-conviction
trades with wider stops and longer holds than mean reversion.

Scoring components:
  - BB squeeze (30%): Bollinger Band width percentile < 20th = max squeeze
  - Volume compression→expansion (25%): declining vol + today spike
  - Relative strength vs SPY (20%): outperforming = momentum base
  - Trend alignment (15%): SMA50/200 stack
  - Momentum base (10%): RSI(14) 40-65 coiled zone

Hard gates: ATR% < 3.5, avg vol < 500K, score < 60.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _valid(x) -> bool:
    """Check if a value is a valid, finite number."""
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


@dataclass
class SniperSignal:
    ticker: str
    score: float          # 0-100
    direction: str        # LONG
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    holding_period: int   # 7 days
    components: dict
    max_entry_price: float | None = None


def score_sniper(
    ticker: str,
    df: pd.DataFrame,
    features: dict,
    fundamental_data: dict | None = None,
    regime: str = "unknown",
    spy_df: pd.DataFrame | None = None,
    atr_pct_floor: float = 3.5,
    min_avg_volume: int = 500_000,
    stop_atr_mult: float = 2.0,
    target_atr_mult: float = 3.0,
    target_2_atr_mult: float = 5.0,
    holding_period: int = 7,
) -> SniperSignal | None:
    """Score a ticker for sniper (BB squeeze + vol compression) potential.

    Returns None if hard gates fail or score < 60.
    """
    if df.empty or len(df) < 60:
        return None

    # --- Hard gate: bear regime block ---
    # Sniper targets volatility expansion which fails in bear (61.5% WR, -0.02% avg).
    # Only fire in bull/choppy where wind is at our back.
    if regime == "bear":
        return None

    scores: dict[str, float] = {}
    close = df["close"].astype(float)

    # --- Hard gate: ATR% floor ---
    atr_pct = features.get("atr_pct")
    if not _valid(atr_pct) or float(atr_pct) < atr_pct_floor:
        return None

    # --- Hard gate: average volume ---
    vol_sma_20 = features.get("vol_sma_20")
    if _valid(vol_sma_20) and float(vol_sma_20) < min_avg_volume:
        return None
    # Fallback: check raw volume average
    if not _valid(vol_sma_20) and len(df) >= 20:
        avg_vol = float(df["volume"].astype(float).tail(20).mean())
        if avg_vol < min_avg_volume:
            return None

    # --- 1. BB Squeeze (30%) ---
    bb_width_col = "BBB_20_2.0"
    bb_score = 50.0  # neutral default
    if bb_width_col in df.columns and len(df) >= 60:
        bb_width = df[bb_width_col].astype(float)
        current_bb = bb_width.iloc[-1]
        if _valid(current_bb):
            # Percentile rank over last 60 bars
            recent_60 = bb_width.tail(60).dropna()
            if len(recent_60) >= 20:
                pctile = float((recent_60 <= current_bb).sum()) / len(recent_60)
                if pctile <= 0.20:
                    bb_score = 100  # tight squeeze
                elif pctile <= 0.40:
                    bb_score = 70   # moderate squeeze
                else:
                    bb_score = max(0, 50 - (pctile - 0.40) * 100)
    scores["bb_squeeze"] = bb_score

    # --- 2. Volume Compression → Expansion (25%) ---
    vol_score = 50.0
    if "volume" in df.columns and len(df) >= 6:
        vol = df["volume"].astype(float)
        recent_5 = vol.tail(6).values  # 5 bars + today
        today_vol = float(vol.iloc[-1])
        avg_5 = float(vol.tail(6).iloc[:-1].mean()) if len(recent_5) >= 2 else 0

        # Check if volume is declining over last 5 bars
        if len(recent_5) >= 5:
            x = np.arange(5, dtype=float)
            slope = float(np.polyfit(x, recent_5[:5], 1)[0])
            vol_declining = slope < 0
        else:
            vol_declining = False

        vol_sma = features.get("vol_sma_20")
        avg_vol_20 = float(vol_sma) if _valid(vol_sma) else avg_5

        if vol_declining and avg_vol_20 > 0 and today_vol > 1.5 * avg_vol_20:
            vol_score = 100  # compression → expansion = ideal
        elif vol_declining:
            vol_score = 60   # compression without expansion yet
        elif avg_vol_20 > 0 and today_vol > 1.5 * avg_vol_20:
            vol_score = 70   # expansion without prior compression
    scores["vol_compression"] = vol_score

    # --- 3. Relative Strength vs SPY (20%) ---
    rs_score = 50.0
    roc_10 = features.get("roc_10")
    if _valid(roc_10) and spy_df is not None and len(spy_df) >= 11:
        spy_close = spy_df["close"].astype(float)
        spy_roc_10 = float((spy_close.iloc[-1] - spy_close.iloc[-11]) / spy_close.iloc[-11] * 100)
        rs_diff = float(roc_10) - spy_roc_10
        if rs_diff > 5:
            rs_score = 100
        elif rs_diff > 2:
            rs_score = 80
        elif rs_diff > 0:
            rs_score = 60
        else:
            rs_score = max(0, 40 + rs_diff * 5)  # penalize underperformers
    scores["relative_strength"] = rs_score

    # --- 4. Trend Alignment (15%) ---
    pct_above_sma50 = features.get("pct_above_sma50")
    pct_above_sma200 = features.get("pct_above_sma200")
    above_50 = _valid(pct_above_sma50) and float(pct_above_sma50) > 0
    above_200 = _valid(pct_above_sma200) and float(pct_above_sma200) > 0

    # Check SMA50 > SMA200 (golden cross alignment)
    sma50 = features.get("sma_50")
    sma200 = features.get("sma_200")
    sma50_above_200 = (_valid(sma50) and _valid(sma200) and float(sma50) > float(sma200))

    if above_50 and sma50_above_200:
        trend_score = 100  # full stack alignment
    elif above_50:
        trend_score = 60   # at least above intermediate trend
    elif above_200:
        trend_score = 40   # above long-term but below intermediate
    else:
        trend_score = 10   # below both
    scores["trend_alignment"] = trend_score

    # --- 5. Momentum Base (10%) ---
    rsi_14 = features.get("rsi_14")
    mom_score = 50.0
    if _valid(rsi_14):
        rsi_val = float(rsi_14)
        if 40 <= rsi_val <= 65:
            mom_score = 100  # coiled, not overbought
        elif 30 <= rsi_val < 40:
            mom_score = 60   # slightly oversold — ok
        elif 65 < rsi_val <= 75:
            mom_score = 50   # slightly overbought
        else:
            mom_score = 20   # extreme — not ideal for sniper
    scores["momentum_base"] = mom_score

    # --- Composite ---
    weights = {
        "bb_squeeze": 0.30,
        "vol_compression": 0.25,
        "relative_strength": 0.20,
        "trend_alignment": 0.15,
        "momentum_base": 0.10,
    }
    composite = sum(scores[k] * weights[k] for k in weights)

    # Regime adjustment (bear already hard-blocked above)
    regime_floors = {
        "choppy": 65,
    }
    floor = regime_floors.get(regime, 60)
    if composite < floor:
        return None

    # --- Price targets ---
    close_price = features.get("close", 0)
    atr = features.get("atr_14")
    if not _valid(close_price) or close_price <= 0:
        return None
    if not _valid(atr) or atr <= 0:
        atr = close_price * 0.03
    atr = max(atr, close_price * 0.005)

    stop_loss = close_price - stop_atr_mult * atr
    target_1 = close_price + target_atr_mult * atr
    target_2 = close_price + target_2_atr_mult * atr

    return SniperSignal(
        ticker=ticker,
        score=round(composite, 2),
        direction="LONG",
        entry_price=round(close_price, 2),
        stop_loss=round(stop_loss, 2),
        target_1=round(target_1, 2),
        target_2=round(target_2, 2),
        holding_period=holding_period,
        components=scores,
    )
