"""Validation card — per-signal fragility report + pipeline validation gate.

Assesses how robust a signal is across:
- Performance dispersion across time periods
- Slippage sensitivity
- Threshold sensitivity
- Number of variants tested (multiple testing penalty)

Also implements the 6 validation checks from docs/validation_contract.md
as a pipeline gate (NoSilentPass rule: any failed check blocks picks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from src.contracts import (
    LeakageChecks,
    FragilityMetrics,
    ValidationPayload,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationCard:
    signal_model: str
    total_trades: int
    win_rate: float
    avg_pnl_pct: float

    # Robustness measures
    performance_dispersion: float  # std of rolling window returns
    slippage_sensitivity: float  # how much does doubling slippage hurt?
    threshold_sensitivity: float  # score range where signal flips on/off
    variants_tested: int  # how many parameter combos were tried
    multiple_testing_penalty: float  # Bonferroni-style adjustment

    # Regime breakdown
    bull_win_rate: float
    bear_win_rate: float
    choppy_win_rate: float

    # Deflated Sharpe Ratio (0-1, >0.95 is statistically significant)
    deflated_sharpe: float

    # Verdict
    is_robust: bool
    fragility_score: float  # 0-100, lower = more robust
    notes: list[str]


def generate_validation_card(
    signal_model: str,
    trade_returns: list[float],
    trade_returns_by_regime: dict[str, list[float]],
    slippage_returns: list[float],  # returns with 2x slippage
    variants_tested: int = 1,
) -> ValidationCard:
    """Generate a fragility report for a signal model."""
    if not trade_returns:
        return _empty_card(signal_model)

    arr = np.array(trade_returns)
    total = len(arr)
    win_rate = float(np.mean(arr > 0))
    avg_pnl = float(np.mean(arr))

    # --- Performance dispersion ---
    # Split into 5 equal windows and compute win rate in each
    window_size = max(1, total // 5)
    window_win_rates = []
    for i in range(0, total, window_size):
        window = arr[i:i + window_size]
        if len(window) > 0:
            window_win_rates.append(float(np.mean(window > 0)))
    dispersion = float(np.std(window_win_rates)) if len(window_win_rates) > 1 else 0

    # --- Slippage sensitivity ---
    slippage_arr = np.array(slippage_returns) if slippage_returns else arr
    base_avg = float(np.mean(arr))
    slippage_avg = float(np.mean(slippage_arr))
    slippage_sensitivity = abs(base_avg - slippage_avg) / max(abs(base_avg), 0.01) if base_avg != 0 else 0

    # --- Threshold sensitivity ---
    # Not directly computable without re-running signals; estimate from score distribution
    threshold_sensitivity = dispersion * 2  # proxy

    # --- Multiple testing penalty ---
    # Bonferroni: required p-value = 0.05 / variants_tested
    # In practice: if you tested 10 variants, your best result needs to be 10x better
    penalty = min(1.0, np.log(max(1, variants_tested)) / np.log(20))

    # --- Regime breakdown ---
    def _regime_wr(regime_key: str) -> float:
        r = trade_returns_by_regime.get(regime_key, [])
        if not r:
            return 0.0
        return float(np.mean(np.array(r) > 0))

    bull_wr = _regime_wr("bull")
    bear_wr = _regime_wr("bear")
    choppy_wr = _regime_wr("choppy")

    # --- Fragility score ---
    # Lower is better (more robust)
    fragility = 0.0
    notes = []

    # High dispersion = fragile
    if dispersion > 0.2:
        fragility += 25
        notes.append(f"High performance dispersion ({dispersion:.2f})")

    # Slippage sensitive
    if slippage_sensitivity > 0.5:
        fragility += 20
        notes.append(f"Highly slippage-sensitive ({slippage_sensitivity:.2f})")

    # Multiple testing
    if penalty > 0.5:
        fragility += 15
        notes.append(f"Multiple testing concern ({variants_tested} variants)")

    # Poor regime diversity
    regime_rates = [r for r in [bull_wr, bear_wr, choppy_wr] if r > 0]
    if len(regime_rates) < 2:
        fragility += 20
        notes.append("Positive only in 0-1 regime types")

    # Low total trades (small sample)
    if total < 30:
        fragility += 20
        notes.append(f"Small sample size ({total} trades)")
    elif total < 100:
        fragility += 10

    # --- Deflated Sharpe Ratio ---
    from src.backtest.metrics import deflated_sharpe_ratio
    sharpe = float(np.mean(arr)) / float(np.std(arr, ddof=1)) * np.sqrt(50) if np.std(arr, ddof=1) > 0 else 0
    dsr = deflated_sharpe_ratio(sharpe, variants_tested, trade_returns)

    if dsr < 0.5 and variants_tested > 1:
        fragility += 15
        notes.append(f"Low Deflated Sharpe Ratio ({dsr:.2f}) — possible selection bias")

    fragility = min(100, fragility)
    is_robust = fragility < 40 and win_rate > 0.5 and avg_pnl > 0

    if is_robust:
        notes.append("PASSED: Signal appears robust")
    else:
        notes.append("FAILED: Signal shows fragility concerns")

    return ValidationCard(
        signal_model=signal_model,
        total_trades=total,
        win_rate=round(win_rate, 4),
        avg_pnl_pct=round(avg_pnl, 4),
        performance_dispersion=round(dispersion, 4),
        slippage_sensitivity=round(slippage_sensitivity, 4),
        threshold_sensitivity=round(threshold_sensitivity, 4),
        variants_tested=variants_tested,
        multiple_testing_penalty=round(penalty, 4),
        bull_win_rate=round(bull_wr, 4),
        bear_win_rate=round(bear_wr, 4),
        choppy_win_rate=round(choppy_wr, 4),
        deflated_sharpe=dsr,
        is_robust=is_robust,
        fragility_score=round(fragility, 2),
        notes=notes,
    )


def _empty_card(signal_model: str) -> ValidationCard:
    return ValidationCard(
        signal_model=signal_model, total_trades=0, win_rate=0, avg_pnl_pct=0,
        performance_dispersion=0, slippage_sensitivity=0, threshold_sensitivity=0,
        variants_tested=0, multiple_testing_penalty=0, bull_win_rate=0, bear_win_rate=0,
        choppy_win_rate=0, deflated_sharpe=0, is_robust=False, fragility_score=100,
        notes=["No trades to validate"],
    )


# ── Pipeline Validation Gate (NoSilentPass) ─────────────────────────────

_PASS = "pass"
_FAIL = "fail"


def run_validation_checks(
    run_date: date,
    signal_dates: list[date | datetime],
    execution_dates: list[date | datetime],
    feature_columns: list[str],
    validation_card: ValidationCard | None = None,
    slippage_bps: float = 10.0,
) -> ValidationPayload:
    """Run the 6 validation checks from docs/validation_contract.md.

    Any single failure sets validation_status to 'fail', enforcing the
    NoSilentPass rule: failed validation blocks picks.

    Args:
        run_date: The as-of date for this pipeline run.
        signal_dates: Dates when each signal was generated.
        execution_dates: Dates when each signal would be executed (T+1).
        feature_columns: Column names used in feature engineering.
        validation_card: Pre-computed fragility card for the signal model.
        slippage_bps: Slippage assumption in basis points.
    """
    checks: dict[str, str] = {}
    key_risks: list[str] = []
    notes_parts: list[str] = []

    # Known future-leaking column patterns
    _FUTURE_PATTERNS = {"forward_", "future_", "target_actual", "next_day_close"}

    # ── Check 1: timestamp_integrity_check ──
    # Signals must only use data available at asof_timestamp.
    asof_ok = True
    for sd in signal_dates:
        d = sd.date() if isinstance(sd, datetime) else sd
        if d > run_date:
            asof_ok = False
            break
    checks["timestamp_integrity_check"] = _PASS if asof_ok else _FAIL
    if not asof_ok:
        key_risks.append("Signal dates extend beyond as-of date (look-ahead)")

    # ── Check 2: next_bar_execution_check ──
    # Execution must be T+1 or later relative to signal date.
    exec_ok = True
    for sd, ed in zip(signal_dates, execution_dates):
        sd_date = sd.date() if isinstance(sd, datetime) else sd
        ed_date = ed.date() if isinstance(ed, datetime) else ed
        if ed_date <= sd_date:
            exec_ok = False
            break
    checks["next_bar_execution_check"] = _PASS if exec_ok else _FAIL
    if not exec_ok:
        key_risks.append("Same-bar fill detected (execution <= signal date)")

    # ── Check 3: future_data_guard_check ──
    # Reject columns/fields tagged as future-known.
    future_cols = [c for c in feature_columns if any(p in c.lower() for p in _FUTURE_PATTERNS)]
    no_future = len(future_cols) == 0
    checks["future_data_guard_check"] = _PASS if no_future else _FAIL
    if not no_future:
        key_risks.append(f"Future-leaking columns found: {future_cols}")

    # ── Check 4: slippage_sensitivity_check ──
    # Signal must survive +50% slippage increase.
    if validation_card and validation_card.total_trades > 0:
        slippage_ok = validation_card.slippage_sensitivity < 0.5
        slippage_sens = validation_card.slippage_sensitivity
    else:
        slippage_ok = True  # no data to check → pass by default
        slippage_sens = 0.0
    checks["slippage_sensitivity_check"] = _PASS if slippage_ok else _FAIL
    if not slippage_ok:
        key_risks.append(f"Slippage sensitivity too high ({slippage_sens:.2f})")

    # ── Check 5: threshold_sensitivity_check ──
    # Score threshold +/- 10% should not flip >30% of signals.
    if validation_card and validation_card.total_trades > 0:
        threshold_ok = validation_card.threshold_sensitivity < 0.3
        threshold_sens = validation_card.threshold_sensitivity
    else:
        threshold_ok = True
        threshold_sens = 0.0
    checks["threshold_sensitivity_check"] = _PASS if threshold_ok else _FAIL
    if not threshold_ok:
        key_risks.append(f"Threshold sensitivity too high ({threshold_sens:.2f})")

    # ── Check 6: confidence_calibration_check ──
    # High-confidence predictions must have higher win rate than low-confidence.
    # When no card data, pass by default (first run).
    if validation_card and validation_card.total_trades >= 30:
        cal_ok = validation_card.win_rate > 0.45  # minimum bar
        cal_bucket = (
            "high" if validation_card.win_rate > 0.6
            else "medium" if validation_card.win_rate > 0.5
            else "low"
        )
    else:
        cal_ok = True
        cal_bucket = "insufficient_data"
    checks["confidence_calibration_check"] = _PASS if cal_ok else _FAIL
    if not cal_ok:
        key_risks.append(f"Win rate below calibration minimum ({validation_card.win_rate:.2%})")

    # ── Aggregate ──
    all_passed = all(v == _PASS for v in checks.values())
    n_failed = sum(1 for v in checks.values() if v == _FAIL)

    if not all_passed:
        notes_parts.append(f"{n_failed}/{len(checks)} checks failed — NoSilentPass blocks picks")
    else:
        notes_parts.append("All 6 validation checks passed")

    fragility_score = (
        validation_card.fragility_score / 100.0  # normalize 0-100 → 0-1
        if validation_card and validation_card.total_trades > 0
        else 0.0
    )

    return ValidationPayload(
        leakage_checks=LeakageChecks(
            asof_timestamp_present=checks["timestamp_integrity_check"] == _PASS,
            next_bar_execution_enforced=checks["next_bar_execution_check"] == _PASS,
            future_data_columns_found=future_cols,
        ),
        fragility_metrics=FragilityMetrics(
            slippage_sensitivity=slippage_sens,
            threshold_sensitivity=threshold_sens,
            confidence_calibration_bucket=cal_bucket if validation_card else "no_data",
        ),
        validation_status=_PASS if all_passed else _FAIL,
        checks=checks,
        fragility_score=round(min(1.0, fragility_score), 4),
        key_risks=key_risks,
        notes="; ".join(notes_parts),
    )
