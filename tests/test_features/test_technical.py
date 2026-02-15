"""Tests for technical feature engineering."""

import pandas as pd
import pytest

from src.features.technical import (
    compute_all_technical_features,
    compute_rsi2_features,
    latest_features,
)


def test_compute_all_features_adds_expected_columns(sample_ohlcv):
    df = compute_all_technical_features(sample_ohlcv)

    expected = [
        "sma_10", "sma_20", "sma_50", "ema_9", "ema_21",
        "rsi_14", "rsi_2", "atr_14", "atr_pct",
        "vol_sma_20", "rvol", "volume_surge",
        "high_20d", "low_20d", "near_20d_high", "near_20d_low",
        "roc_5", "roc_10", "pct_above_sma20",
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


def test_latest_features_empty():
    assert latest_features(pd.DataFrame()) == {}
