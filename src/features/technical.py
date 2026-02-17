"""Technical feature engineering — ATR, RSI, VWAP deviation, RVOL, MAs, volume surge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


def compute_all_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on an OHLCV DataFrame.

    Expects columns: open, high, low, close, volume (and optionally vwap).
    Returns the same DataFrame with new indicator columns appended.
    """
    if df.empty or len(df) < 20:
        return df

    df = df.copy()

    # --- Trend / Moving Averages ---
    df["sma_10"] = ta.sma(df["close"], length=10)
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["sma_50"] = ta.sma(df["close"], length=50)
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)

    # Price vs MAs (guard against division by zero when SMA is NaN/0)
    df["pct_above_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"].replace(0, pd.NA) * 100
    df["pct_above_sma50"] = (df["close"] - df["sma_50"]) / df["sma_50"].replace(0, pd.NA) * 100

    # --- Momentum ---
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["rsi_2"] = ta.rsi(df["close"], length=2)  # for mean-reversion model

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df = pd.concat([df, macd], axis=1)

    # Rate of change
    df["roc_5"] = ta.roc(df["close"], length=5)
    df["roc_10"] = ta.roc(df["close"], length=10)

    # --- Volatility ---
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    if atr is not None:
        df["atr_14"] = atr
        df["atr_pct"] = df["atr_14"] / df["close"].replace(0, pd.NA) * 100  # ATR as % of price

    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None and not bb.empty:
        df = pd.concat([df, bb], axis=1)

    # --- Volume ---
    df["vol_sma_20"] = ta.sma(df["volume"], length=20)
    df["rvol"] = df["volume"] / df["vol_sma_20"].replace(0, pd.NA)  # relative volume

    # Volume surge: today's volume vs 20-day avg
    df["volume_surge"] = (df["volume"] > df["vol_sma_20"] * 2.0).astype(int)

    # On-balance volume
    obv = ta.obv(df["close"], df["volume"])
    if obv is not None:
        df["obv"] = obv

    # --- VWAP deviation ---
    if "vwap" in df.columns:
        df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, pd.NA) * 100
    else:
        # Approximate VWAP using typical price * volume
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        df["vwap_approx"] = cum_tp_vol / cum_vol.replace(0, pd.NA)
        df["vwap_dev"] = (df["close"] - df["vwap_approx"]) / df["vwap_approx"].replace(0, pd.NA) * 100

    # --- Breakout signals ---
    df["high_20d"] = df["high"].rolling(20).max()
    df["low_20d"] = df["low"].rolling(20).min()
    df["near_20d_high"] = (df["close"] >= df["high_20d"] * 0.98).astype(int)
    df["near_20d_low"] = (df["close"] <= df["low_20d"] * 1.02).astype(int)

    # Consolidation detection: 10-day range < 1.5x ATR
    df["range_10d"] = df["high"].rolling(10).max() - df["low"].rolling(10).min()
    if "atr_14" in df.columns:
        df["is_consolidating"] = (df["range_10d"] < df["atr_14"] * 1.5).astype(int)

    return df


def compute_rsi2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Specific features for the RSI(2) mean-reversion model."""
    if df.empty or len(df) < 10:
        return df

    df = df.copy()
    df["rsi_2"] = ta.rsi(df["close"], length=2)

    # Streak counting: consecutive down/up days
    df["daily_return"] = df["close"].pct_change()
    streak = []
    current = 0
    for ret in df["daily_return"]:
        if pd.isna(ret):
            streak.append(0)
            continue
        if ret < 0:
            current = current - 1 if current < 0 else -1
        elif ret > 0:
            current = current + 1 if current > 0 else 1
        else:
            current = 0
        streak.append(current)
    df["streak"] = streak

    # Distance from 5-day low
    df["low_5d"] = df["low"].rolling(5).min()
    df["dist_from_5d_low"] = (df["close"] - df["low_5d"]) / df["low_5d"].replace(0, pd.NA) * 100

    return df


def latest_features(df: pd.DataFrame) -> dict:
    """Extract the most recent row's features as a flat dictionary."""
    if df.empty:
        return {}
    row = df.iloc[-1]
    return {k: (None if pd.isna(v) else float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in row.items()}
