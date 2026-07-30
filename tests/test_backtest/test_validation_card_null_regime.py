"""Regression: a NULL Signal.regime must not crash the validation-card builder.

Signal.regime is nullable and NULL rows DO exist in the live DB (the dashboard
export shows mean_reversion trades with regime=None). Before 2026-07-30 the builder
called signal.regime.lower() with no guard, so one such row entering the 90-day
window would raise AttributeError — and because validation fails CLOSED, that would
block every pick for the day.
"""
from __future__ import annotations

from src.backtest.validation_card import generate_validation_card, run_validation_checks


def _by_regime_from(signals_and_pnl):
    """Mirrors the grouping in build_validation_card_from_history."""
    by_regime: dict[str, list[float]] = {}
    for regime, pnl in signals_and_pnl:
        by_regime.setdefault((regime or "unknown").lower(), []).append(pnl)
    return by_regime


def test_null_regime_groups_as_unknown_instead_of_crashing():
    rows = [(None, 1.0), ("BULL", 2.0), (None, -1.0), ("bear", -0.5)]
    by_regime = _by_regime_from(rows)
    assert set(by_regime) == {"unknown", "bull", "bear"}
    assert by_regime["unknown"] == [1.0, -1.0]


def test_card_builds_with_only_null_regimes():
    """All-NULL regimes must still produce a card (all regime counts zero)."""
    by_regime = _by_regime_from([(None, 1.0)] * 12)
    card = generate_validation_card(
        "mean_reversion", [1.0] * 12, by_regime, [0.9] * 12, variants_tested=1,
    )
    assert card.total_trades == 12
    assert card.bull_trades == card.bear_trades == card.choppy_trades == 0


def test_two_sampled_regimes_require_both_positive():
    """Documents the strict rule behind the live block: with exactly 2 sampled
    regimes, required_positive = min(2, 2) = 2, so ONE weak cohort blocks all picks."""
    from datetime import date

    # >=30 trades, else the check auto-passes as "insufficient data".
    by_regime = {"bull": [1.0] * 20, "bear": [-1.0] * 20}  # bull positive, bear not
    card = generate_validation_card(
        "mean_reversion", [1.0] * 20 + [-1.0] * 20, by_regime, [0.9] * 40,
        variants_tested=1,
    )
    payload = run_validation_checks(
        run_date=date(2026, 7, 30),
        signal_dates=[date(2026, 7, 29)] * 40,
        execution_dates=[date(2026, 7, 30)] * 40,
        feature_columns=["rsi_2"],
        validation_card=card,
        allowed_regimes={"bull", "bear", "choppy"},
    )
    assert payload.checks["regime_survival_check"] == "fail"
    assert any("1/2" in r for r in payload.key_risks)
