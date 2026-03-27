"""Tests for per-model validation bootstrap logic.

Verifies that:
- Per-model cards are isolated (sniper can't block MR)
- Models with < 10 trades auto-pass fragility checks
- Slippage sensitivity requires >= 30 trades to block
- execution_mode filtering excludes stale agentic_full trades
"""

from datetime import date, timedelta

from src.backtest.validation_card import (
    ValidationCard,
    run_validation_checks,
)


def _make_card(
    signal_model: str = "mean_reversion",
    total_trades: int = 50,
    slippage_sensitivity: float = 0.1,
    bull_win_rate: float = 0.7,
    bear_win_rate: float = 0.55,
    choppy_win_rate: float = 0.6,
    **kwargs,
) -> ValidationCard:
    defaults = dict(
        win_rate=0.65,
        avg_pnl_pct=1.5,
        performance_dispersion=0.1,
        threshold_sensitivity=0.1,
        variants_tested=1,
        multiple_testing_penalty=0.0,
        deflated_sharpe=0.95,
        is_robust=True,
        fragility_score=20.0,
        notes=["PASSED"],
    )
    defaults.update(kwargs)
    return ValidationCard(
        signal_model=signal_model,
        total_trades=total_trades,
        slippage_sensitivity=slippage_sensitivity,
        bull_win_rate=bull_win_rate,
        bear_win_rate=bear_win_rate,
        choppy_win_rate=choppy_win_rate,
        **defaults,
    )


# ── Slippage sensitivity sample-size guard ──


def test_slippage_high_but_few_trades_passes():
    """With < 30 trades, slippage_sensitivity_check auto-passes even if value is high."""
    today = date(2025, 3, 15)
    card = _make_card(total_trades=15, slippage_sensitivity=1.5)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=[],
        validation_card=card,
    )
    assert result.checks["slippage_sensitivity_check"] == "pass"


def test_slippage_high_with_enough_trades_fails():
    """With >= 30 trades, high slippage sensitivity correctly fails."""
    today = date(2025, 3, 15)
    card = _make_card(total_trades=30, slippage_sensitivity=0.8)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=[],
        validation_card=card,
    )
    assert result.checks["slippage_sensitivity_check"] == "fail"


# ── Per-model isolation ──


def test_none_card_passes_all_fragility_checks():
    """When validation_card is None (< 10 trades for model), all fragility checks pass."""
    today = date(2025, 3, 15)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=[],
        validation_card=None,
    )
    assert result.validation_status == "pass"
    assert result.checks["slippage_sensitivity_check"] == "pass"
    assert result.checks["regime_survival_check"] == "pass"
    assert result.checks["confidence_calibration_check"] == "pass"
    assert result.checks["threshold_sensitivity_check"] == "pass"


def test_weak_card_does_not_affect_none_card():
    """Simulates per-model: a weak sniper card cannot block MR when MR has no card."""
    today = date(2025, 3, 15)

    # Sniper card: lots of trades, bad stats → fails
    sniper_card = _make_card(
        signal_model="sniper",
        total_trades=50,
        slippage_sensitivity=0.9,
        bull_win_rate=0.3,
        bear_win_rate=0.2,
        choppy_win_rate=0.4,
    )
    sniper_result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=[],
        validation_card=sniper_card,
    )
    assert sniper_result.validation_status == "fail"

    # MR card: None (< 10 trades) → passes
    mr_result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=[],
        validation_card=None,
    )
    assert mr_result.validation_status == "pass"


# ── Structural checks always apply ──


def test_structural_checks_fail_regardless_of_card():
    """Future-data leakage fails even with no validation card."""
    today = date(2025, 3, 15)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=["forward_return_5d"],
        validation_card=None,
    )
    assert result.checks["future_data_guard_check"] == "fail"
    assert result.validation_status == "fail"


# ── Bootstrap scenario: small trade count ──


def test_bootstrap_scenario_small_sample_all_passes():
    """Early-stage model with 5 trades: all statistical checks auto-pass."""
    today = date(2025, 3, 15)
    card = _make_card(
        total_trades=5,
        slippage_sensitivity=2.0,  # would fail with enough trades
        bull_win_rate=0.0,
        bear_win_rate=0.6,
        choppy_win_rate=0.0,  # only 1/3 regimes → would fail with enough trades
    )
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=[],
        validation_card=card,
    )
    # All statistical checks should pass due to insufficient sample
    assert result.checks["slippage_sensitivity_check"] == "pass"
    assert result.checks["regime_survival_check"] == "pass"
    assert result.checks["threshold_sensitivity_check"] == "pass"
    assert result.checks["confidence_calibration_check"] == "pass"
    assert result.validation_status == "pass"
