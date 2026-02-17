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


def test_breakout_gap_up_boost(sample_ohlcv):
    """Gap-up with volume should score higher than without."""
    df = compute_all_technical_features(sample_ohlcv)
    feat_base = latest_features(df)
    # Moderate setup
    feat_base["rsi_14"] = 60
    feat_base["rvol"] = 2.5
    feat_base["near_20d_high"] = 1
    feat_base["is_consolidating"] = 1
    feat_base["roc_5"] = 4.0
    feat_base["roc_10"] = 6.0
    feat_base["pct_above_sma20"] = 2.0
    feat_base["pct_above_sma50"] = 3.0
    feat_base["atr_pct"] = 3.0
    feat_base["atr_14"] = feat_base.get("close", 100) * 0.03

    # Without gap
    feat_no_gap = {**feat_base, "is_gap_up": 0, "gap_pct": 1.0}
    result_no_gap = score_breakout("NOGAP", df, feat_no_gap)

    # With gap
    feat_gap = {**feat_base, "is_gap_up": 1, "gap_pct": 5.0}
    result_gap = score_breakout("GAP", df, feat_gap)

    assert result_no_gap is not None
    assert result_gap is not None
    assert result_gap.score > result_no_gap.score, "Gap-up should boost breakout score"


def test_breakout_empty_df():
    import pandas as pd
    result = score_breakout("TEST", pd.DataFrame(), {})
    assert result is None
