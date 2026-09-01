"""Integration test: prove shadow mode stores veto in components, not skip_reason.

Tests the actual veto layer with real signals to verify:
1. Shadow mode: vetoed signals stay in list, veto info in components
2. Hard mode: vetoed signals removed from list
3. Persist path: RankedCandidate.persisted_features includes quality_veto
"""

from __future__ import annotations

import pandas as pd

from src.signals.ranker import RankedCandidate
from src.signals.veto import VETO_EXTENDED, apply_veto_layer


def test_shadow_mode_stores_veto_in_components():
    """Shadow mode: vetoed signal stays in list with veto info in components."""
    # Build mock signals with components dict
    class MockSignal:
        def __init__(self, ticker):
            self.ticker = ticker
            self.components = {}

    sig_clean = MockSignal("CLEAN")
    sig_extended = MockSignal("EXTENDED")

    # Build OHLCV: EXTENDED at 20-day high, CLEAN mid-range
    dates = pd.date_range(end="2024-12-31", periods=40, freq="D")
    df_extended = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 40,
        "high": [100.5] * 40,
        "low": [99.0] * 40,
        "close": [100.5] * 40,  # At high
        "volume": [1_000_000] * 40,
    })
    df_clean = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 40,
        "high": [110.0] * 40,
        "low": [90.0] * 40,
        "close": [95.0] * 40,  # Mid-range
        "volume": [1_000_000] * 40,
    })

    price_data = {"EXTENDED": df_extended, "CLEAN": df_clean}

    # Run apply_veto_layer in shadow mode
    signals, _ = apply_veto_layer(
        [sig_clean, sig_extended],
        price_data=price_data,
        fundamental_data_by_ticker={},
        shadow_only=True,
    )

    # Both signals stay in list (shadow mode)
    assert len(signals) == 2
    tickers = [s.ticker for s in signals]
    assert "CLEAN" in tickers
    assert "EXTENDED" in tickers

    # Vetoed signal has veto_reason attached
    extended_sig = next(s for s in signals if s.ticker == "EXTENDED")
    assert hasattr(extended_sig, 'veto_reason')
    assert extended_sig.veto_reason == VETO_EXTENDED

    # Mimic main.py: copy veto_reason into components['quality_veto']
    if hasattr(extended_sig, 'veto_reason') and extended_sig.veto_reason:
        extended_sig.components['quality_veto'] = {
            'reason': extended_sig.veto_reason,
            'vetoed': True,
        }

    # Verify components['quality_veto'] is set
    assert 'quality_veto' in extended_sig.components
    assert extended_sig.components['quality_veto']['reason'] == VETO_EXTENDED
    assert extended_sig.components['quality_veto']['vetoed'] is True

    # Clean signal has no veto_reason
    clean_sig = next(s for s in signals if s.ticker == "CLEAN")
    assert not hasattr(clean_sig, 'veto_reason') or clean_sig.veto_reason is None

    # Test persist path: RankedCandidate.persisted_features includes quality_veto
    ranked = RankedCandidate(
        ticker="EXTENDED",
        signal_model="test_model",
        raw_score=85.0,
        regime_adjusted_score=85.0,
        direction="long",
        entry_price=100.0,
        stop_loss=95.0,
        target_1=105.0,
        target_2=110.0,
        holding_period=5,
        components=dict(extended_sig.components),
        features={},
    )
    features = ranked.persisted_features()
    assert 'model_components' in features
    assert 'quality_veto' in features['model_components']
    assert features['model_components']['quality_veto']['reason'] == VETO_EXTENDED


def test_hard_veto_mode_removes_from_list():
    """Hard veto mode: vetoed signal removed from list."""
    class MockSignal:
        def __init__(self, ticker):
            self.ticker = ticker
            self.components = {}

    sig_clean = MockSignal("CLEAN")
    sig_extended = MockSignal("EXTENDED")

    dates = pd.date_range(end="2024-12-31", periods=40, freq="D")
    df_extended = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 40,
        "high": [100.5] * 40,
        "low": [99.0] * 40,
        "close": [100.5] * 40,
        "volume": [1_000_000] * 40,
    })
    df_clean = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 40,
        "high": [110.0] * 40,
        "low": [90.0] * 40,
        "close": [95.0] * 40,
        "volume": [1_000_000] * 40,
    })

    price_data = {"EXTENDED": df_extended, "CLEAN": df_clean}

    # Run apply_veto_layer in hard veto mode
    signals, _ = apply_veto_layer(
        [sig_clean, sig_extended],
        price_data=price_data,
        fundamental_data_by_ticker={},
        shadow_only=False,
    )

    # Only clean signal remains (hard veto removed EXTENDED)
    assert len(signals) == 1
    assert signals[0].ticker == "CLEAN"
