"""Tests for Telegram alert formatting."""

from src.output.telegram import format_daily_alert, format_outcome_alert


def test_format_daily_alert_with_picks():
    picks = [
        {
            "ticker": "AAPL",
            "direction": "LONG",
            "entry_price": 195.50,
            "stop_loss": 190.00,
            "target_1": 210.00,
            "confidence": 78,
            "signal_model": "breakout",
            "thesis": "Strong momentum breakout with volume confirmation.",
            "holding_period": 10,
        },
    ]

    msg = format_daily_alert(picks, "bull", "2025-03-15")
    assert "AAPL" in msg
    assert "LONG" in msg
    assert "$195.50" in msg
    assert "breakout" in msg
    assert "78" in msg
    assert "BULL" in msg
    assert "2025-03-15" in msg


def test_format_daily_alert_no_picks():
    msg = format_daily_alert([], "choppy", "2025-03-15")
    assert "No high-conviction picks" in msg
    assert "CHOPPY" in msg


def test_format_daily_alert_risk_reward():
    picks = [
        {
            "ticker": "MSFT",
            "direction": "LONG",
            "entry_price": 400.0,
            "stop_loss": 390.0,
            "target_1": 420.0,
            "confidence": 65,
            "signal_model": "catalyst",
            "thesis": "",
            "holding_period": 15,
        },
    ]

    msg = format_daily_alert(picks, "bull", "2025-03-15")
    assert "R:R:" in msg
    assert "2.0:1" in msg  # (420-400)/(400-390) = 2.0


def test_format_outcome_alert():
    outcomes = [
        {"ticker": "AAPL", "pnl_pct": 3.5, "exit_reason": "target"},
        {"ticker": "MSFT", "pnl_pct": -1.2, "exit_reason": "stop"},
        {"ticker": "NVDA", "pnl_pct": 0.8, "exit_reason": "open"},
    ]

    msg = format_outcome_alert(outcomes)
    assert "AAPL" in msg
    assert "+3.50%" in msg
    assert "MSFT" in msg
    assert "-1.20%" in msg
    assert "still open" in msg


def test_format_outcome_alert_empty():
    assert format_outcome_alert([]) == ""
