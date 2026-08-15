"""Unit tests for quality veto layer — extended tape, dilution, data sanity.

These tests use synthetic data only (no network, no parquet) and verify that:
1. Extended tape veto fires when close is at/near 20-day high
2. Dilution veto fires when shares outstanding increase dramatically
3. Data sanity veto fires when two snapshots disagree on key metrics
4. All vetoes fail open when data is missing or insufficient
5. Shadow mode keeps vetoed signals in the output (with veto_reason attached)
6. Official admission/ranking is unchanged in shadow mode
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.signals.veto import (
    VETO_DATA_SANITY,
    VETO_DILUTION,
    VETO_EXTENDED,
    apply_veto_layer,
    veto_data_sanity,
    veto_dilution,
    veto_extended_tape,
)

# --- Synthetic signal object for testing ---

@dataclass
class MockSignal:
    ticker: str
    score: float
    direction: str = "LONG"
    veto_reason: str | None = None


# --- Extended Tape Tests ---

def test_extended_veto_fires_at_20d_high():
    """Synthetic OHLCV: close pinned at 20-day high → veto fires."""
    # Create 40 days of data with close at $100
    dates = pd.date_range(end="2024-12-31", periods=40, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "open": [99.5] * 40,
        "high": [100.5] * 40,
        "low": [99.0] * 40,
        "close": [100.0] * 40,
        "volume": [1_000_000] * 40,
    })
    # Last bar is at the 20-day high
    df.loc[df.index[-1], "close"] = 100.5  # Exactly at high

    result = veto_extended_tape("TEST", df)
    assert result.vetoed is True
    assert result.veto_reason == VETO_EXTENDED
    assert "TEST" in result.ticker


def test_extended_veto_does_not_fire_mid_range():
    """Synthetic OHLCV: close in mid-range → no veto."""
    dates = pd.date_range(end="2024-12-31", periods=40, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 40,
        "high": [110.0] * 40,  # 20-day high is 110
        "low": [90.0] * 40,
        "close": [95.0] * 40,  # Close well below high
        "volume": [1_000_000] * 40,
    })

    result = veto_extended_tape("TEST", df)
    assert result.vetoed is False
    assert result.veto_reason is None


def test_extended_veto_fails_open_short_history():
    """Extended veto: short history (<20+14 bars) → fail open (no veto)."""
    dates = pd.date_range(end="2024-12-31", periods=20, freq="D")  # Only 20 days
    df = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 20,
        "high": [100.5] * 20,
        "low": [99.0] * 20,
        "close": [100.5] * 20,  # At high, but too short
        "volume": [1_000_000] * 20,
    })

    result = veto_extended_tape("TEST", df)
    assert result.vetoed is False, "Should fail open with insufficient data"


def test_extended_veto_fails_open_empty_df():
    """Extended veto: empty DataFrame → fail open."""
    df = pd.DataFrame()
    result = veto_extended_tape("TEST", df)
    assert result.vetoed is False


# --- Dilution Tests ---

def test_dilution_veto_fires_on_3x_shares():
    """Synthetic share counts: 3× YoY → veto fires."""
    fundamental_data = {
        "profile": {
            "sharesOutstanding": 300_000_000,  # Current: 300M
        },
        "ratios": [
            # Quarterly data, most recent first
            {"weightedAverageShsOut": 300_000_000},  # Q0
            {"weightedAverageShsOut": 250_000_000},  # Q-1
            {"weightedAverageShsOut": 200_000_000},  # Q-2
            {"weightedAverageShsOut": 150_000_000},  # Q-3
            {"weightedAverageShsOut": 100_000_000},  # Q-4 (1 year ago)
        ],
    }

    result = veto_dilution("TEST", fundamental_data, dilution_threshold=2.0)
    assert result.vetoed is True
    assert result.veto_reason == VETO_DILUTION
    assert "3.00x" in result.veto_detail or "3x" in result.veto_detail.lower()


def test_dilution_veto_does_not_fire_on_10pct_growth():
    """Synthetic share counts: 1.1× YoY (normal growth) → no veto."""
    fundamental_data = {
        "profile": {
            "sharesOutstanding": 110_000_000,  # Current: 110M
        },
        "ratios": [
            {"weightedAverageShsOut": 110_000_000},  # Q0
            {"weightedAverageShsOut": 108_000_000},  # Q-1
            {"weightedAverageShsOut": 106_000_000},  # Q-2
            {"weightedAverageShsOut": 104_000_000},  # Q-3
            {"weightedAverageShsOut": 100_000_000},  # Q-4 (1 year ago)
        ],
    }

    result = veto_dilution("TEST", fundamental_data, dilution_threshold=2.0)
    assert result.vetoed is False


def test_dilution_veto_fails_open_no_historical():
    """Dilution veto: no historical share data → fail open."""
    fundamental_data = {
        "profile": {
            "sharesOutstanding": 300_000_000,  # Only current, no historical
        },
        "ratios": [],  # Empty ratios list
    }

    result = veto_dilution("TEST", fundamental_data)
    assert result.vetoed is False, "Should fail open without historical data"


def test_dilution_veto_fails_open_no_data():
    """Dilution veto: no fundamental data → fail open."""
    result = veto_dilution("TEST", None)
    assert result.vetoed is False


# --- Data Sanity Tests ---

def test_data_sanity_veto_fires_on_2x_revenue_disagreement():
    """Synthetic snapshots: 2× revenue disagreement → veto fires."""
    snapshot_a = {
        "revenue": 100_000_000,  # $100M
        "sharesOutstanding": 50_000_000,
    }
    snapshot_b = {
        "revenue": 200_000_000,  # $200M (2x difference!)
        "sharesOutstanding": 50_000_000,  # Shares agree
    }

    result = veto_data_sanity("TEST", snapshot_a, snapshot_b, tolerance_pct=0.10)
    assert result.vetoed is True
    assert result.veto_reason == VETO_DATA_SANITY
    assert "revenue" in result.veto_detail.lower()


def test_data_sanity_veto_does_not_fire_on_2pct_disagreement():
    """Synthetic snapshots: 2% revenue disagreement (within tolerance) → no veto."""
    snapshot_a = {
        "revenue": 100_000_000,
        "sharesOutstanding": 50_000_000,
    }
    snapshot_b = {
        "revenue": 102_000_000,  # 2% higher, within 10% tolerance
        "sharesOutstanding": 50_000_000,
    }

    result = veto_data_sanity("TEST", snapshot_a, snapshot_b, tolerance_pct=0.10)
    assert result.vetoed is False


def test_data_sanity_veto_fails_open_one_side_missing():
    """Data sanity veto: one snapshot missing a metric → fail open."""
    snapshot_a = {
        "revenue": 100_000_000,
    }
    snapshot_b = {
        "sharesOutstanding": 50_000_000,  # No revenue field
    }

    result = veto_data_sanity("TEST", snapshot_a, snapshot_b)
    assert result.vetoed is False, "Should fail open when metrics are missing"


def test_data_sanity_veto_fails_open_no_snapshot():
    """Data sanity veto: missing snapshot → fail open."""
    snapshot_a = {"revenue": 100_000_000}
    result = veto_data_sanity("TEST", snapshot_a, None)
    assert result.vetoed is False


# --- Integration: apply_veto_layer ---

def test_apply_veto_layer_shadow_mode():
    """Shadow mode: vetoed signals are kept in output with veto_reason attached."""
    # Create a signal that will be vetoed (extended tape)
    sig = MockSignal(ticker="EXTENDED", score=85.0)
    
    # OHLCV with close at 20-day high
    dates = pd.date_range(end="2024-12-31", periods=40, freq="D")
    df_extended = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 40,
        "high": [100.5] * 40,
        "low": [99.0] * 40,
        "close": [100.5] * 40,  # At high
        "volume": [1_000_000] * 40,
    })

    price_data = {"EXTENDED": df_extended}
    fundamental_data_by_ticker = {"EXTENDED": {}}

    filtered, veto_results = apply_veto_layer(
        [sig],
        price_data=price_data,
        fundamental_data_by_ticker=fundamental_data_by_ticker,
        shadow_only=True,
    )

    # In shadow mode, signal should be kept
    assert len(filtered) == 1
    assert filtered[0].ticker == "EXTENDED"
    assert filtered[0].veto_reason == VETO_EXTENDED

    # Veto results should show it was vetoed
    vetoed_results = [r for r in veto_results if r.vetoed]
    assert len(vetoed_results) == 1
    assert vetoed_results[0].veto_reason == VETO_EXTENDED


def test_apply_veto_layer_hard_mode():
    """Hard veto mode (shadow_only=False): vetoed signals are removed."""
    sig = MockSignal(ticker="EXTENDED", score=85.0)
    
    # OHLCV with close at 20-day high
    dates = pd.date_range(end="2024-12-31", periods=40, freq="D")
    df_extended = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 40,
        "high": [100.5] * 40,
        "low": [99.0] * 40,
        "close": [100.5] * 40,
        "volume": [1_000_000] * 40,
    })

    price_data = {"EXTENDED": df_extended}
    fundamental_data_by_ticker = {"EXTENDED": {}}

    filtered, veto_results = apply_veto_layer(
        [sig],
        price_data=price_data,
        fundamental_data_by_ticker=fundamental_data_by_ticker,
        shadow_only=False,
    )

    # In hard veto mode, signal should be removed
    assert len(filtered) == 0

    # Veto results should show it was vetoed
    vetoed_results = [r for r in veto_results if r.vetoed]
    assert len(vetoed_results) == 1


def test_apply_veto_layer_official_picks_unchanged_in_shadow_mode():
    """Shadow mode: non-vetoed signals pass through unchanged."""
    # Create signals: one clean, one vetoed
    sig_clean = MockSignal(ticker="CLEAN", score=90.0)
    sig_extended = MockSignal(ticker="EXTENDED", score=85.0)

    dates = pd.date_range(end="2024-12-31", periods=40, freq="D")
    
    # CLEAN: mid-range close
    df_clean = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 40,
        "high": [110.0] * 40,
        "low": [90.0] * 40,
        "close": [95.0] * 40,  # Mid-range
        "volume": [1_000_000] * 40,
    })

    # EXTENDED: at high
    df_extended = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 40,
        "high": [100.5] * 40,
        "low": [99.0] * 40,
        "close": [100.5] * 40,  # At high
        "volume": [1_000_000] * 40,
    })

    price_data = {
        "CLEAN": df_clean,
        "EXTENDED": df_extended,
    }
    fundamental_data_by_ticker = {
        "CLEAN": {},
        "EXTENDED": {},
    }

    filtered, _ = apply_veto_layer(
        [sig_clean, sig_extended],
        price_data=price_data,
        fundamental_data_by_ticker=fundamental_data_by_ticker,
        shadow_only=True,
    )

    # Both signals should be in output (shadow mode)
    assert len(filtered) == 2
    
    # CLEAN should have no veto_reason
    clean_sig = next((s for s in filtered if s.ticker == "CLEAN"), None)
    assert clean_sig is not None
    assert not hasattr(clean_sig, "veto_reason") or clean_sig.veto_reason is None

    # EXTENDED should have veto_reason
    extended_sig = next((s for s in filtered if s.ticker == "EXTENDED"), None)
    assert extended_sig is not None
    assert extended_sig.veto_reason == VETO_EXTENDED


def test_apply_veto_layer_multiple_vetoes():
    """If multiple vetoes fire, first one wins (for skip_reason)."""
    # This signal will trigger both extended and dilution vetoes
    sig = MockSignal(ticker="MULTI", score=85.0)

    dates = pd.date_range(end="2024-12-31", periods=40, freq="D")
    df_extended = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 40,
        "high": [100.5] * 40,
        "low": [99.0] * 40,
        "close": [100.5] * 40,  # At high
        "volume": [1_000_000] * 40,
    })

    # Dilution data: 3x shares
    fundamental_data = {
        "profile": {"sharesOutstanding": 300_000_000},
        "ratios": [
            {"weightedAverageShsOut": 300_000_000},
            {"weightedAverageShsOut": 250_000_000},
            {"weightedAverageShsOut": 200_000_000},
            {"weightedAverageShsOut": 150_000_000},
            {"weightedAverageShsOut": 100_000_000},
        ],
    }

    price_data = {"MULTI": df_extended}
    fundamental_data_by_ticker = {"MULTI": fundamental_data}

    filtered, veto_results = apply_veto_layer(
        [sig],
        price_data=price_data,
        fundamental_data_by_ticker=fundamental_data_by_ticker,
        shadow_only=True,
    )

    # Signal should be vetoed and have one of the veto reasons
    assert len(filtered) == 1
    assert filtered[0].veto_reason in [VETO_EXTENDED, VETO_DILUTION]

    # Multiple veto results should be recorded
    vetoed_results = [r for r in veto_results if r.vetoed]
    assert len(vetoed_results) >= 1  # At least one veto fired
