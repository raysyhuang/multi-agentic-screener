"""Tests for mean reversion signal model."""


from src.features.technical import (
    compute_all_technical_features,
    compute_rsi2_features,
    latest_features,
)
from src.signals.mean_reversion import score_mean_reversion


def test_mean_reversion_fires_on_oversold(sample_ohlcv_oversold):
    df = compute_all_technical_features(sample_ohlcv_oversold)
    df = compute_rsi2_features(df)
    feat = latest_features(df)

    # Force oversold conditions
    feat["rsi_2"] = 5
    feat["streak"] = -4
    feat["dist_from_5d_low"] = 0.5
    feat["rvol"] = 1.0
    feat["atr_14"] = feat.get("close", 100) * 0.02

    result = score_mean_reversion("OVERSOLD", df, feat)
    assert result is not None
    assert result.direction == "LONG"
    assert result.holding_period == 5
    assert result.stop_loss < result.entry_price


def test_mean_reversion_skips_not_oversold(sample_ohlcv):
    df = compute_all_technical_features(sample_ohlcv)
    df = compute_rsi2_features(df)
    feat = latest_features(df)
    feat["rsi_2"] = 60  # not oversold

    result = score_mean_reversion("NOTOVERSOLD", df, feat)
    assert result is None


def test_mean_reversion_empty_df():
    import pandas as pd
    result = score_mean_reversion("TEST", pd.DataFrame(), {})
    assert result is None
