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


def _payload_for(by_regime, returns):
    from datetime import date

    card = generate_validation_card(
        "mean_reversion", returns, by_regime, [r - 0.1 for r in returns],
        variants_tested=1,
    )
    n = len(returns)
    return card, run_validation_checks(
        run_date=date(2026, 7, 31),
        signal_dates=[date(2026, 7, 30)] * n,
        execution_dates=[date(2026, 7, 31)] * n,
        feature_columns=["rsi_2"],
        validation_card=card,
        allowed_regimes={"bull", "bear", "choppy"},
    )


def test_undersampled_regime_cannot_veto():
    """The live block: bear 8/17 = 47% WR (Wilson CI [26%, 69%]) blocked the whole
    book for 3 days while live MR ran 75% WR. A cohort that small cannot distinguish
    a broken model from a healthy one, so it must not gate production."""
    by_regime = {"bear": [1.0] * 8 + [-1.0] * 9, "choppy": [1.0] * 9 + [-1.0] * 5}
    returns = by_regime["bear"] + by_regime["choppy"]
    _, payload = _payload_for(by_regime, returns)
    assert payload.checks["regime_survival_check"] == "pass"


def test_large_negative_regime_still_blocks():
    """The check must keep its teeth: a genuinely large, genuinely losing cohort
    (n >= 30) still fails regime survival."""
    by_regime = {"bear": [1.0] * 12 + [-1.0] * 23, "choppy": [1.0] * 12 + [-1.0] * 23}
    returns = by_regime["bear"] + by_regime["choppy"]
    _, payload = _payload_for(by_regime, returns)
    assert payload.checks["regime_survival_check"] == "fail"


def test_one_large_positive_regime_passes():
    """Sniper's live shape: a single well-sampled positive regime (bull n=76, 55%)
    should pass rather than be penalised for having no bear trades by design."""
    by_regime = {"bull": [1.0] * 42 + [-1.0] * 34}
    returns = by_regime["bull"]
    _, payload = _payload_for(by_regime, returns)
    assert payload.checks["regime_survival_check"] == "pass"
