"""Tests for Telegram alert formatting."""

import re

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
    assert "\u25b2" in msg  # direction arrow for LONG
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
    assert "R:R" in msg
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
    assert "(open)" in msg


def test_format_outcome_alert_empty():
    assert format_outcome_alert([]) == ""


def test_format_daily_alert_validation_failed():
    """When validation fails, alert should show failed checks."""
    msg = format_daily_alert(
        picks=[],
        regime="bull",
        run_date="2025-03-15",
        validation_failed=True,
        failed_checks=["slippage_sensitivity_check", "regime_survival_check"],
        key_risks=["Slippage sensitivity too high (0.65)"],
    )
    assert "Validation FAILED" in msg
    assert "slippage_sensitivity_check" in msg
    assert "regime_survival_check" in msg
    assert "Slippage sensitivity" in msg
    assert "All picks blocked" in msg


def test_format_daily_alert_with_fragility_warnings():
    """Alerts with picks should include fragility warnings if present."""
    picks = [
        {
            "ticker": "AAPL", "direction": "LONG", "entry_price": 195.0,
            "stop_loss": 190.0, "target_1": 210.0, "confidence": 70,
            "signal_model": "breakout", "thesis": "Test", "holding_period": 10,
        },
    ]
    msg = format_daily_alert(
        picks, "bull", "2025-03-15",
        key_risks=["Small sample size (15 trades)"],
    )
    assert "AAPL" in msg
    assert "Risks" in msg
    assert "Small sample size" in msg


def test_format_daily_alert_escapes_html():
    """HTML special characters in dynamic fields must be escaped."""
    picks = [
        {
            "ticker": "A<B>C",
            "direction": "LONG",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "target_1": 110.0,
            "confidence": 70,
            "signal_model": "breakout",
            "thesis": 'Price > 50 & volume < 1M "test"',
            "holding_period": 10,
        },
    ]
    msg = format_daily_alert(picks, "bull", "2025-03-15")
    assert "A&lt;B&gt;C" in msg
    assert "&amp;" in msg
    # No raw < or > outside of known HTML tags
    stripped = re.sub(r"</?(?:b|i|code)>", "", msg)
    assert "<" not in stripped
    assert ">" not in stripped


def test_format_outcome_alert_escapes_html():
    """Outcome alerts must escape ticker and exit reason."""
    outcomes = [
        {"ticker": "X&Y", "pnl_pct": 1.0, "exit_reason": "target<1>"},
    ]
    msg = format_outcome_alert(outcomes)
    assert "X&amp;Y" in msg
    assert "target&lt;1&gt;" in msg
