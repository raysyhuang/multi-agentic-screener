"""Tests for technical feature engineering."""

import pandas as pd

from src.features.technical import (
    compute_all_technical_features,
    compute_rsi2_features,
    latest_features,
)


def test_compute_all_features_adds_expected_columns(sample_ohlcv):
    df = compute_all_technical_features(sample_ohlcv)

    expected = [
        "sma_10", "sma_20", "sma_50", "sma_200", "ema_9", "ema_21",
        "rsi_14", "rsi_2", "atr_14", "atr_pct",
        "vol_sma_20", "rvol", "volume_surge",
        "high_20d", "low_20d", "near_20d_high", "near_20d_low",
        "roc_5", "roc_10", "pct_above_sma20", "pct_above_sma200",
        "sma_20_slope", "gap_pct", "is_gap_up", "is_gap_down",
    ]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"


def test_compute_all_features_empty_df():
    df = compute_all_technical_features(pd.DataFrame())
    assert df.empty


def test_compute_all_features_short_df():
    df = pd.DataFrame({
        "open": [100, 101], "high": [102, 103],
        "low": [99, 100], "close": [101, 102], "volume": [1e6, 1.1e6],
    })
    result = compute_all_technical_features(df)
    # Should return the same short df without crashing
    assert len(result) == 2


def test_rsi_values_bounded(sample_ohlcv):
    df = compute_all_technical_features(sample_ohlcv)
    rsi = df["rsi_14"].dropna()
    assert rsi.min() >= 0
    assert rsi.max() <= 100


def test_rvol_positive(sample_ohlcv):
    df = compute_all_technical_features(sample_ohlcv)
    rvol = df["rvol"].dropna()
    assert (rvol > 0).all()


def test_compute_rsi2_features(sample_ohlcv):
    df = compute_rsi2_features(sample_ohlcv)
    assert "rsi_2" in df.columns
    assert "streak" in df.columns
    assert "dist_from_5d_low" in df.columns


def test_latest_features_returns_dict(sample_ohlcv):
    df = compute_all_technical_features(sample_ohlcv)
    feat = latest_features(df)
    assert isinstance(feat, dict)
    assert "close" in feat
    assert "rsi_14" in feat


def test_sma200_with_long_data(sample_ohlcv_long):
    df = compute_all_technical_features(sample_ohlcv_long)
    assert "sma_200" in df.columns
    # Last 50 rows should have non-NaN SMA(200) values
    tail = df["sma_200"].iloc[-50:]
    assert tail.notna().all(), "SMA(200) should be non-NaN for last 50 rows of 250-day data"


def test_gap_detection(sample_ohlcv):
    df = compute_all_technical_features(sample_ohlcv)
    assert "gap_pct" in df.columns
    assert "is_gap_up" in df.columns
    assert "is_gap_down" in df.columns
    # gap_pct should be NaN for first row, numeric elsewhere
    assert df["gap_pct"].iloc[1:].notna().all()
    # is_gap_up and is_gap_down should be 0 or 1
    assert df["is_gap_up"].isin([0, 1]).all()
    assert df["is_gap_down"].isin([0, 1]).all()


def test_sma20_slope(sample_ohlcv):
    df = compute_all_technical_features(sample_ohlcv)
    assert "sma_20_slope" in df.columns
    # After warmup (20 for SMA + 10 for shift = 30 bars), values should be non-NaN
    tail = df["sma_20_slope"].iloc[30:]
    assert tail.notna().all(), "sma_20_slope should be non-NaN after warmup period"


def test_latest_features_empty():
    assert latest_features(pd.DataFrame()) == {}
