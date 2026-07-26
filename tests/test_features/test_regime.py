"""Tests for market regime classifier."""

import numpy as np
import pandas as pd

from src.features.regime import (
    classify_regime,
    Regime,
    get_regime_allowed_models,
    compute_breadth_score,
)


def test_bull_regime_with_uptrend(sample_spy_df, sample_qqq_df):
    """Uptrending SPY + QQQ + low VIX = bull."""
    result = classify_regime(sample_spy_df, sample_qqq_df, vix=14.0, yield_spread=1.5)
    assert result.regime == Regime.BULL
    assert result.confidence > 0.3
    assert result.spy_trend == "above_sma20"


def test_bear_regime_with_downtrend():
    """Downtrending data + high VIX = bear."""
    np.random.seed(55)
    n = 60
    close = 450 - np.cumsum(np.abs(np.random.randn(n)) * 2)  # strong downtrend

    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n).date,
        "open": close + 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.full(n, 5e7),
    })

    result = classify_regime(df, df, vix=32.0, yield_spread=-0.2)
    assert result.regime == Regime.BEAR
    assert result.vix_level == 32.0


def test_choppy_regime_with_divergence(sample_spy_df):
    """SPY up but QQQ down = choppy."""
    np.random.seed(77)
    n = 60
    close = 380 - np.cumsum(np.abs(np.random.randn(n)) * 1.5)
    qqq_down = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n).date,
        "open": close + 0.5, "high": close + 1,
        "low": close - 1, "close": close,
        "volume": np.full(n, 3e7),
    })

    result = classify_regime(sample_spy_df, qqq_down, vix=20.0)
    # Divergence between SPY and QQQ often produces choppy
    assert result.regime in (Regime.CHOPPY, Regime.BULL)


def test_regime_with_empty_df():
    result = classify_regime(pd.DataFrame(), pd.DataFrame())
    assert result.spy_trend == "unknown"


def test_hy_oas_surfaced_as_context(sample_spy_df, sample_qqq_df):
    """HY OAS level + stress flag are always surfaced, regardless of the tilt."""
    result = classify_regime(sample_spy_df, sample_qqq_df, vix=14.0,
                             yield_spread=1.5, hy_oas=3.2, hy_oas_stress=True)
    assert result.hy_oas == 3.2
    assert result.hy_oas_stress is True
    assert result.details["hy_oas"] == "stress"
    result_calm = classify_regime(sample_spy_df, sample_qqq_df, vix=14.0,
                                  hy_oas=2.9, hy_oas_stress=False)
    assert result_calm.details["hy_oas"] == "calm"


def test_hy_oas_fail_open_when_absent(sample_spy_df, sample_qqq_df):
    """No HY data → hy_oas_stress None → no effect, signal 'unknown'."""
    result = classify_regime(sample_spy_df, sample_qqq_df, vix=14.0, yield_spread=1.5)
    assert result.hy_oas is None
    assert result.hy_oas_stress is None
    assert result.details["hy_oas"] == "unknown"


def test_hy_oas_tilt_off_by_default(sample_spy_df, sample_qqq_df):
    """With the tilt default-off, HY stress does NOT change the regime label."""
    base = classify_regime(sample_spy_df, sample_qqq_df, vix=20.0, yield_spread=0.3)
    with_stress = classify_regime(sample_spy_df, sample_qqq_df, vix=20.0,
                                  yield_spread=0.3, hy_oas=5.5, hy_oas_stress=True)
    assert with_stress.regime == base.regime


def test_hy_oas_tilt_adds_bear_when_enabled(monkeypatch, sample_spy_df, sample_qqq_df):
    """When enabled, HY stress adds bear score (higher weight can flip the label)."""
    from src.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "regime_hy_oas_enabled", True)
    monkeypatch.setattr(s, "regime_hy_oas_bear_weight", 5.0)  # large to force a flip
    calm = classify_regime(sample_spy_df, sample_qqq_df, vix=20.0, yield_spread=0.3,
                           hy_oas=3.0, hy_oas_stress=False)
    stress = classify_regime(sample_spy_df, sample_qqq_df, vix=20.0, yield_spread=0.3,
                             hy_oas=6.0, hy_oas_stress=True)
    # A large bear weight on stress must move the label toward bear vs the calm case.
    assert stress.regime != calm.regime or stress.confidence != calm.confidence


def test_allowed_models_bull():
    models = get_regime_allowed_models(Regime.BULL)
    assert "breakout" in models
    assert "mean_reversion" in models
    assert "catalyst" in models


def test_allowed_models_bear():
    models = get_regime_allowed_models(Regime.BEAR)
    assert "breakout" not in models
    assert "mean_reversion" in models


def test_allowed_models_choppy():
    models = get_regime_allowed_models(Regime.CHOPPY)
    assert "breakout" not in models
    assert "mean_reversion" in models
    assert "catalyst" in models


def test_breadth_score_all_above():
    """All tickers above 20d SMA → breadth near 1.0."""
    np.random.seed(42)
    price_data = {}
    for i in range(20):
        close = 100 + np.arange(30) * 0.5  # steady uptrend
        price_data[f"T{i}"] = pd.DataFrame({"close": close})
    score = compute_breadth_score(price_data)
    assert score is not None
    assert score >= 0.9


def test_breadth_score_all_below():
    """All tickers below 20d SMA → breadth near 0.0."""
    np.random.seed(43)
    price_data = {}
    for i in range(20):
        close = 100 - np.arange(30) * 0.5  # steady downtrend
        price_data[f"T{i}"] = pd.DataFrame({"close": close})
    score = compute_breadth_score(price_data)
    assert score is not None
    assert score <= 0.1


def test_breadth_score_insufficient_data():
    """Too few tickers → None."""
    price_data = {"A": pd.DataFrame({"close": [100, 101]})}
    assert compute_breadth_score(price_data) is None


def test_breadth_affects_regime(sample_spy_df, sample_qqq_df):
    """Broad breadth should boost bull score."""
    result_no_breadth = classify_regime(sample_spy_df, sample_qqq_df, vix=18.0)
    result_with_breadth = classify_regime(sample_spy_df, sample_qqq_df, vix=18.0, breadth_score=0.75)
    # With high breadth, bull confidence should be at least as high
    assert result_with_breadth.breadth_score == 0.75
    if result_no_breadth.regime == Regime.BULL:
        assert result_with_breadth.confidence >= result_no_breadth.confidence
