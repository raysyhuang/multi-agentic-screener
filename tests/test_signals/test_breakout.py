"""Tests for breakout signal model."""


from src.features.technical import compute_all_technical_features, latest_features
from src.signals.breakout import score_breakout


def test_breakout_returns_signal_for_strong_setup(sample_ohlcv):
    df = compute_all_technical_features(sample_ohlcv)
    feat = latest_features(df)
    # Force strong features
    feat["rsi_14"] = 65
    feat["rvol"] = 2.5
    feat["near_20d_high"] = 1
    feat["is_consolidating"] = 1
    feat["roc_5"] = 5.0
    feat["roc_10"] = 8.0
    feat["pct_above_sma20"] = 3.0
    feat["pct_above_sma50"] = 5.0
    feat["atr_pct"] = 3.0
    feat["atr_14"] = feat.get("close", 100) * 0.03

    result = score_breakout("TEST", df, feat)
    assert result is not None
    assert result.score >= 50
    assert result.direction == "LONG"
    assert result.stop_loss < result.entry_price
    assert result.target_1 > result.entry_price


def test_breakout_returns_none_for_weak_setup(sample_ohlcv):
    df = compute_all_technical_features(sample_ohlcv)
    feat = latest_features(df)
    # Force weak features
    feat["rsi_14"] = 30
    feat["rvol"] = 0.5
    feat["near_20d_high"] = 0
    feat["roc_5"] = -3.0
    feat["roc_10"] = -5.0
    feat["pct_above_sma20"] = -5.0
    feat["pct_above_sma50"] = -10.0

    result = score_breakout("WEAK", df, feat)
    assert result is None


def test_breakout_empty_df():
    import pandas as pd
    result = score_breakout("TEST", pd.DataFrame(), {})
    assert result is None
