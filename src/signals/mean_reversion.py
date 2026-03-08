"""RSI(2) oversold mean-reversion model.

Ported from gemini_STST's 3-day RSI(2) model with next-day-open execution.
Looks for oversold conditions in stocks with intact long-term uptrends.

Fires on day T close, execution at T+1 open.

Parameters optimized via walk-forward backtest (S&P 500, 2yr, 35K+ trades):
  - RSI(2) threshold: 10 (was 20) — stricter filter, stronger per-trade signal
  - Stop: 0.75x ATR (was 1.0x) — tighter invalidation, Sharpe 0.887 vs 0.641
  - Target: 1.5x ATR floor (was 1.0x) — wider target, Sharpe 0.875 vs 0.629
  - Hold: 3 days (unchanged) — edge decays quickly after day 3
  See outputs/research/sweep_mean_reversion_all.csv for full parameter sweep.
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
    max_entry_price: float | None = None  # gap filter: reject if T+1 open exceeds this


def score_mean_reversion(
    ticker: str,
    df: pd.DataFrame,
    features: dict,
    fundamental_data: dict | None = None,
    regime: str = "unknown",
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
    else:
        return None  # backtest: RSI(2)<=10 is the optimal threshold

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

    # --- 5. Volume Signature (10%) ---
    # Declining volume into the low = selling exhaustion (buy).
    # Spiking volume into the low = distribution (avoid).
    rvol = features.get("rvol")
    vol_score = 50  # neutral default
    if len(df) >= 4 and "volume" in df.columns:
        import numpy as np
        recent_vol = df["volume"].astype(float).iloc[-3:].values
        if len(recent_vol) == 3 and all(v > 0 for v in recent_vol):
            # Linear regression slope over 3 bars
            x = np.arange(3, dtype=float)
            slope = float(np.polyfit(x, recent_vol, 1)[0])
            if slope > 0 and _valid(rvol) and rvol > 1.5:
                vol_score = 10  # spiking volume into low = distribution, penalize
            elif slope < 0:
                vol_score = 80  # selling exhaustion = good setup
            elif _valid(rvol) and rvol >= 0.5:
                vol_score = 70
            elif _valid(rvol) and rvol < 0.3:
                vol_score = 20
    elif _valid(rvol):
        if rvol >= 0.5:
            vol_score = 70
        elif rvol < 0.3:
            vol_score = 20
    scores["liquidity"] = vol_score

    # --- 6. Fundamental quality/value filter (10%) ---
    ratio_profile = {}
    if isinstance(fundamental_data, dict):
        ratio_profile = fundamental_data.get("ratio_profile", {}) or {}
    ratio_score = 50.0
    if isinstance(ratio_profile, dict):
        value_score = ratio_profile.get("value_score")
        quality_score = ratio_profile.get("quality_score")
        if _valid(value_score) and _valid(quality_score):
            ratio_score = max(0.0, min(100.0, (float(value_score) + float(quality_score)) / 2.0))
    scores["fundamental"] = ratio_score

    # --- Composite ---
    weights = {
        "rsi2_oversold": 0.35,
        "trend_intact": 0.22,
        "down_streak": 0.13,
        "proximity_to_low": 0.10,
        "liquidity": 0.10,
        "fundamental": 0.10,
    }
    composite = sum(scores[k] * weights[k] for k in weights)

    # Regime-dependent score floor (choppy is a coinflip — demand higher quality)
    from src.config import get_settings
    _settings = get_settings()
    regime_floors = {
        "choppy": _settings.choppy_min_score,
    }
    floor = regime_floors.get(regime, 50)
    if composite < floor:
        return None

    # --- ATR percentile gate: no MR if ATR14 at 52-week low (need stretch for snapback) ---
    atr_raw = features.get("atr_14")
    if _valid(atr_raw) and len(df) >= 252 and "high" in df.columns and "low" in df.columns:
        close_series = df["close"].astype(float)
        high_series = df["high"].astype(float)
        low_series = df["low"].astype(float)
        # Compute historical ATR14 percentile
        tr = pd.concat([
            high_series - low_series,
            (high_series - close_series.shift(1)).abs(),
            (low_series - close_series.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean().dropna()
        if len(atr_series) >= 252:
            import numpy as np
            atr_pctile = float(np.sum(atr_series.iloc[-252:] <= float(atr_raw))) / 252
            if atr_pctile < _settings.min_atr_percentile_252:
                return None

    # --- Price targets ---
    close_price = features.get("close", 0)
    atr = features.get("atr_14")
    if not _valid(close_price) or close_price <= 0:
        return None
    if not _valid(atr) or atr <= 0:
        atr = close_price * 0.02
    atr = max(atr, close_price * 0.005)

    # Primary target: 1.5x ATR (backtest-optimized, was 1.0x)
    sma_5 = close.rolling(5).mean().iloc[-1]
    sma_5_val = float(sma_5) if pd.notna(sma_5) else close_price * 1.03
    target_1 = max(sma_5_val, close_price + 1.5 * atr)

    # Extended target: back to 10-day SMA with 2.0x ATR floor
    sma_10 = close.rolling(10).mean().iloc[-1]
    sma_10_val = float(sma_10) if pd.notna(sma_10) else close_price * 1.05
    target_2 = max(sma_10_val, close_price + 2.0 * atr)

    # Tight stop: 0.75x ATR (backtest-optimized, was 1.0x — if it doesn't bounce fast, thesis is wrong)
    stop_loss = close_price - 0.75 * atr

    # Gap filter: max acceptable T+1 open price
    max_entry = round(close_price + _settings.entry_gap_max_atr * atr, 2)

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
        max_entry_price=max_entry,
    )
