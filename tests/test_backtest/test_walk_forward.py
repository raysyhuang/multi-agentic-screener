"""Tests for walk-forward backtesting engine."""

from datetime import date

import pandas as pd

from src.backtest.walk_forward import run_walk_forward


def test_walk_forward_basic(sample_ohlcv):
    """Test walk-forward on a simple signal."""
    signals_df = pd.DataFrame([{
        "date": sample_ohlcv["date"].iloc[20],  # signal on day 20
        "ticker": "TEST",
        "signal_model": "breakout",
        "direction": "LONG",
        "entry_price": float(sample_ohlcv["close"].iloc[20]),
        "stop_loss": float(sample_ohlcv["close"].iloc[20]) * 0.95,
        "target_1": float(sample_ohlcv["close"].iloc[20]) * 1.10,
    }])

    result = run_walk_forward(
        signals_df,
        price_data={"TEST": sample_ohlcv},
        holding_periods=[5, 10],
    )

    assert result.total_trades > 0
    assert len(result.trades) > 0
    for trade in result.trades:
        assert trade.exit_reason in ("target", "stop", "expiry")
        assert trade.entry_date > signals_df["date"].iloc[0]  # no look-ahead


def test_walk_forward_no_data():
    signals_df = pd.DataFrame([{
        "date": date(2025, 3, 1),
        "ticker": "MISSING",
        "signal_model": "breakout",
        "direction": "LONG",
        "entry_price": 100,
        "stop_loss": 95,
        "target_1": 110,
    }])

    result = run_walk_forward(signals_df, price_data={}, holding_periods=[5])
    assert result.total_trades == 0


def test_walk_forward_entry_is_t_plus_1(sample_ohlcv):
    """Verify no look-ahead: entry must be after signal date."""
    signal_date = sample_ohlcv["date"].iloc[30]
    signals_df = pd.DataFrame([{
        "date": signal_date,
        "ticker": "TEST",
        "signal_model": "breakout",
        "direction": "LONG",
        "entry_price": float(sample_ohlcv["close"].iloc[30]),
        "stop_loss": float(sample_ohlcv["close"].iloc[30]) * 0.90,
        "target_1": float(sample_ohlcv["close"].iloc[30]) * 1.15,
    }])

    result = run_walk_forward(
        signals_df, price_data={"TEST": sample_ohlcv}, holding_periods=[10],
    )

    for trade in result.trades:
        assert trade.entry_date > signal_date, "Entry must be after signal date (T+1)"
