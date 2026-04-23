"""Tests for Telegram alert formatting."""

from contextlib import asynccontextmanager
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.output.telegram as telegram
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


_SAMPLE_SCORECARD = {
    "mean_reversion": {"trades": 17, "win_rate": 0.53, "avg_pnl": 0.28, "profit_factor": 1.2, "open_positions": 5},
    "sniper": {"trades": 0, "status": "waiting for bull/choppy regime", "open_positions": 0},
}


def test_scorecard_shown_on_validation_failure():
    """Model scorecard must appear even when validation gate blocks all picks."""
    msg = format_daily_alert(
        picks=[],
        regime="bear",
        run_date="2025-03-15",
        validation_failed=True,
        failed_checks=["slippage_sensitivity_check"],
        model_scorecard=_SAMPLE_SCORECARD,
    )
    assert "Validation FAILED" in msg
    assert "Model Scorecard (30d)" in msg
    assert "mean_reversion" in msg
    assert "17 trades" in msg
    assert "sniper" in msg


def test_scorecard_shown_on_no_picks():
    """Model scorecard must appear when there are zero picks."""
    msg = format_daily_alert(
        picks=[],
        regime="choppy",
        run_date="2025-03-15",
        model_scorecard=_SAMPLE_SCORECARD,
    )
    assert "No high-conviction picks" in msg
    assert "Model Scorecard (30d)" in msg
    assert "mean_reversion" in msg


def test_scorecard_shown_on_normal_picks():
    """Model scorecard must appear on normal alerts with picks."""
    picks = [
        {
            "ticker": "AAPL", "direction": "LONG", "entry_price": 195.0,
            "stop_loss": 190.0, "target_1": 210.0, "confidence": 70,
            "signal_model": "mean_reversion", "thesis": "Test", "holding_period": 3,
        },
    ]
    msg = format_daily_alert(
        picks=picks,
        regime="bull",
        run_date="2025-03-15",
        model_scorecard=_SAMPLE_SCORECARD,
    )
    assert "AAPL" in msg
    assert "Model Scorecard (30d)" in msg
    assert "17 trades" in msg
    assert "sniper" in msg


def test_format_outcome_alert_escapes_html():
    """Outcome alerts must escape ticker and exit reason."""
    outcomes = [
        {"ticker": "X&Y", "pnl_pct": 1.0, "exit_reason": "target<1>"},
    ]
    msg = format_outcome_alert(outcomes)
    assert "X&amp;Y" in msg
    assert "target&lt;1&gt;" in msg


@pytest.mark.asyncio
async def test_get_model_scorecard_filters_by_execution_mode(monkeypatch):
    """Scorecard should exclude stale rows from other execution modes."""
    from src.db import session as db_session

    captured: dict[str, list[str]] = {"joins": [], "wheres": []}
    execute_calls = 0

    closed_result = MagicMock()
    closed_result.all.return_value = [
        SimpleNamespace(
            signal_model="mean_reversion",
            trades=4,
            wins=3,
            avg_pnl=1.4049,
        ),
    ]

    open_result = MagicMock()
    open_result.all.return_value = [
        SimpleNamespace(signal_model="mean_reversion", open_count=2),
    ]

    class _FakeSession:
        async def execute(self, statement):
            nonlocal execute_calls
            execute_calls += 1
            captured["joins"].append(str(statement).lower())
            captured["wheres"].append(str(statement.whereclause).lower())
            return closed_result if execute_calls == 1 else open_result

    @asynccontextmanager
    async def _fake_get_session():
        yield _FakeSession()

    monkeypatch.setattr(db_session, "get_session", _fake_get_session)

    scorecard = await telegram.get_model_scorecard(days=30, execution_mode="quant_only")

    assert "breakout" not in scorecard
    assert scorecard["mean_reversion"] == {
        "trades": 4,
        "win_rate": 0.75,
        "avg_pnl": 1.4049,
        "open_positions": 2,
    }
    assert scorecard["sniper"]["status"] == "waiting for bull/choppy regime"
    assert any("daily_runs" in join for join in captured["joins"])
    assert any("execution_mode" in where for where in captured["wheres"])


def test_format_daily_alert_validation_failed_shows_execution_mode():
    """Validation-failed alerts should still show the run mode."""
    msg = format_daily_alert(
        picks=[],
        regime="bear",
        run_date="2026-03-10",
        validation_failed=True,
        failed_checks=["slippage_sensitivity_check"],
        execution_mode="quant_only",
        model_scorecard=_SAMPLE_SCORECARD,
    )

    assert "Mode: QUANT_ONLY" in msg


def test_alert_prefix_defaults_to_mas(monkeypatch):
    monkeypatch.setattr(
        telegram,
        "get_settings",
        lambda: SimpleNamespace(telegram_alert_prefix="MAS", telegram_source_id="mas"),
    )
    msg = format_daily_alert([], "bull", "2026-04-23")
    assert "[MAS]" in msg
    assert "[IBKR]" not in msg


def test_alert_prefix_respects_ibkr_override(monkeypatch):
    monkeypatch.setattr(
        telegram,
        "get_settings",
        lambda: SimpleNamespace(telegram_alert_prefix="IBKR", telegram_source_id="ibkr"),
    )
    msg = format_daily_alert([], "bull", "2026-04-23")
    assert "[IBKR]" in msg
    assert "[MAS]" not in msg
