"""Tests for validation gate enforcement (NoSilentPass rule).

Tests the 6 validation checks from docs/validation_contract.md:
1. timestamp_integrity_check
2. next_bar_execution_check
3. future_data_guard_check
4. slippage_sensitivity_check
5. threshold_sensitivity_check
6. confidence_calibration_check
"""

from datetime import date, timedelta

from src.backtest.validation_card import (
    ValidationCard,
    run_validation_checks,
)


def _make_card(
    win_rate: float = 0.65,
    slippage_sensitivity: float = 0.1,
    threshold_sensitivity: float = 0.1,
    total_trades: int = 50,
    fragility_score: float = 20.0,
) -> ValidationCard:
    return ValidationCard(
        signal_model="breakout",
        total_trades=total_trades,
        win_rate=win_rate,
        avg_pnl_pct=1.5,
        performance_dispersion=0.1,
        slippage_sensitivity=slippage_sensitivity,
        threshold_sensitivity=threshold_sensitivity,
        variants_tested=1,
        multiple_testing_penalty=0.0,
        bull_win_rate=0.7,
        bear_win_rate=0.5,
        choppy_win_rate=0.55,
        deflated_sharpe=0.95,
        is_robust=True,
        fragility_score=fragility_score,
        notes=["PASSED"],
    )


# ── All checks pass ──


def test_all_checks_pass():
    today = date(2025, 3, 15)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=["rsi_14", "atr_pct", "rvol_20d"],
        validation_card=_make_card(),
    )
    assert result.validation_status == "pass"
    assert all(v == "pass" for v in result.checks.values())
    assert len(result.key_risks) == 0


# ── Check 1: timestamp_integrity ──


def test_timestamp_integrity_fails_on_future_signal():
    today = date(2025, 3, 15)
    future = date(2025, 3, 20)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[future],
        execution_dates=[future + timedelta(days=1)],
        feature_columns=[],
    )
    assert result.checks["timestamp_integrity_check"] == "fail"
    assert result.validation_status == "fail"
    assert any("look-ahead" in r.lower() for r in result.key_risks)


# ── Check 2: next_bar_execution ──


def test_next_bar_execution_fails_on_same_day():
    today = date(2025, 3, 15)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today],  # same day = same-bar fill
        feature_columns=[],
    )
    assert result.checks["next_bar_execution_check"] == "fail"
    assert result.validation_status == "fail"
    assert any("Same-bar fill" in r for r in result.key_risks)


# ── Check 3: future_data_guard ──


def test_future_data_guard_detects_leaking_columns():
    today = date(2025, 3, 15)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=["rsi_14", "forward_return_5d", "future_price"],
    )
    assert result.checks["future_data_guard_check"] == "fail"
    assert result.validation_status == "fail"
    assert len(result.leakage_checks.future_data_columns_found) == 2


def test_future_data_guard_passes_clean_columns():
    today = date(2025, 3, 15)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=["rsi_14", "atr_pct", "rvol_20d", "sma_20"],
    )
    assert result.checks["future_data_guard_check"] == "pass"


# ── Check 4: slippage_sensitivity ──


def test_slippage_sensitivity_fails_when_high():
    today = date(2025, 3, 15)
    card = _make_card(slippage_sensitivity=0.7)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=[],
        validation_card=card,
    )
    assert result.checks["slippage_sensitivity_check"] == "fail"
    assert result.validation_status == "fail"


def test_slippage_sensitivity_passes_no_card():
    """Without a validation card, slippage check passes by default."""
    today = date(2025, 3, 15)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=[],
        validation_card=None,
    )
    assert result.checks["slippage_sensitivity_check"] == "pass"


# ── Check 5: threshold_sensitivity ──


def test_threshold_sensitivity_fails_when_high():
    today = date(2025, 3, 15)
    card = _make_card(threshold_sensitivity=0.5)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=[],
        validation_card=card,
    )
    assert result.checks["threshold_sensitivity_check"] == "fail"


# ── Check 6: confidence_calibration ──


def test_confidence_calibration_fails_low_win_rate():
    today = date(2025, 3, 15)
    card = _make_card(win_rate=0.35, total_trades=50)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=[],
        validation_card=card,
    )
    assert result.checks["confidence_calibration_check"] == "fail"


def test_confidence_calibration_passes_insufficient_data():
    """With < 30 trades, calibration check passes (not enough data)."""
    today = date(2025, 3, 15)
    card = _make_card(win_rate=0.30, total_trades=10)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=[],
        validation_card=card,
    )
    assert result.checks["confidence_calibration_check"] == "pass"


# ── NoSilentPass aggregate ──


def test_single_failure_blocks_all_picks():
    """Any single failed check means validation_status = fail."""
    today = date(2025, 3, 15)
    # Only the future_data_guard fails
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=["rsi_14", "forward_return"],
        validation_card=_make_card(),
    )
    assert result.validation_status == "fail"
    # Only 1 check should fail
    failed = [k for k, v in result.checks.items() if v == "fail"]
    assert len(failed) == 1
    assert "future_data_guard_check" in failed


def test_empty_signals_all_pass():
    """With no signals (empty lists), all checks pass."""
    today = date(2025, 3, 15)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[],
        execution_dates=[],
        feature_columns=[],
    )
    assert result.validation_status == "pass"


def test_fragility_score_normalized():
    """Fragility score from card (0-100) should be normalized to 0-1."""
    today = date(2025, 3, 15)
    card = _make_card(fragility_score=45.0)
    result = run_validation_checks(
        run_date=today,
        signal_dates=[today],
        execution_dates=[today + timedelta(days=1)],
        feature_columns=[],
        validation_card=card,
    )
    assert result.fragility_score == 0.45
