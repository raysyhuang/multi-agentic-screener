"""Tests for the post-earnings drift (PEAD) signal."""

from __future__ import annotations

import pandas as pd
import pytest

from src.signals.post_earnings_drift import (
    PEADSignal,
    eps_surprise_pct,
    score_post_earnings_drift,
)


def _df(close: float = 100.0, n: int = 30) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n).date,
        "open": [close] * n, "high": [close * 1.01] * n,
        "low": [close * 0.99] * n, "close": [close] * n, "volume": [1_000_000] * n,
    })


def _feat(close: float = 100.0, atr: float = 2.0) -> dict:
    return {"close": close, "atr_14": atr}


def test_eps_surprise_pct():
    assert eps_surprise_pct(2.0, 1.0) == pytest.approx(100.0)
    assert eps_surprise_pct(0.9, 1.0) == pytest.approx(-10.0)
    assert eps_surprise_pct(None, 1.0) is None
    assert eps_surprise_pct(1.0, 0.0) is None  # zero estimate → undefined


def test_no_signal_without_beat():
    # No earnings today.
    assert score_post_earnings_drift("AAA", _df(), _feat(), None) is None
    # A miss.
    assert score_post_earnings_drift("AAA", _df(), _feat(), -5.0) is None
    # A beat below threshold (default 10%).
    assert score_post_earnings_drift("AAA", _df(), _feat(), 5.0) is None


def test_beat_fires_long_with_atr_levels():
    sig = score_post_earnings_drift(
        "AAA", _df(close=100.0), _feat(close=100.0, atr=2.0),
        earnings_surprise_pct=20.0, min_surprise=10.0,
        stop_atr_mult=3.0, target_atr_mult=6.0, holding_period=20,
    )
    assert isinstance(sig, PEADSignal)
    assert sig.direction == "LONG"
    assert sig.entry_price == 100.0
    assert sig.stop_loss == pytest.approx(100.0 - 3.0 * 2.0, abs=0.01)   # 94
    assert sig.target_1 == pytest.approx(100.0 + 6.0 * 2.0, abs=0.01)    # 112
    assert sig.holding_period == 20
    assert sig.components["eps_surprise_pct"] == 20.0


def test_score_scales_with_surprise():
    small = score_post_earnings_drift("AAA", _df(), _feat(), 10.0, min_surprise=10.0)
    big = score_post_earnings_drift("AAA", _df(), _feat(), 50.0, min_surprise=10.0)
    assert big.score > small.score
    assert big.score <= 100.0


def test_threshold_respected():
    assert score_post_earnings_drift("AAA", _df(), _feat(), 9.9, min_surprise=10.0) is None
    assert score_post_earnings_drift("AAA", _df(), _feat(), 10.0, min_surprise=10.0) is not None


def test_rejects_sub_min_price():
    # $3 stock below the default $5 floor → no signal (edge is liquid-only).
    assert score_post_earnings_drift("AAA", _df(close=3.0), _feat(close=3.0, atr=0.1), 20.0) is None
    # Same beat on a $50 name fires.
    assert score_post_earnings_drift("AAA", _df(close=50.0), _feat(close=50.0, atr=1.0), 20.0) is not None


def test_rejects_negative_stop_from_high_atr():
    # 3xATR (3*4=12) exceeds price (10) → would-be negative stop → reject.
    assert score_post_earnings_drift(
        "AAA", _df(close=10.0), _feat(close=10.0, atr=4.0), 20.0,
        stop_atr_mult=3.0, min_price=1.0,
    ) is None


def test_atr_fallback_when_missing():
    # No atr_14 in features → falls back to a fraction of price, still valid levels.
    sig = score_post_earnings_drift("AAA", _df(close=50.0), {"close": 50.0}, 15.0)
    assert sig is not None
    assert sig.stop_loss < 50.0 < sig.target_1
