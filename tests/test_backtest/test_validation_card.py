"""Tests for validation card generation."""

from datetime import date

from src.backtest.validation_card import generate_validation_card


def test_robust_signal():
    returns = [2.0, -1.0, 3.0, 1.5, -0.5, 2.5, 1.0, -1.5, 3.0, 2.0,
               1.5, -0.5, 2.0, 3.0, -1.0, 1.5, 2.0, -0.5, 1.0, 2.5,
               -1.0, 3.0, 1.5, -0.5, 2.0, 1.0, -1.5, 2.5, 3.0, 1.0]
    by_regime = {"bull": returns[:15], "bear": returns[15:20], "choppy": returns[20:]}
    slippage_returns = [r - 0.2 for r in returns]

    card = generate_validation_card("breakout", returns, by_regime, slippage_returns, variants_tested=1)

    assert card.total_trades == 30
    assert card.win_rate > 0.5
    assert card.fragility_score < 60


def test_fragile_signal_small_sample():
    returns = [2.0, -1.0, 3.0]
    card = generate_validation_card("test", returns, {}, [], variants_tested=10)

    assert card.fragility_score > 40
    assert not card.is_robust
    assert any("Small sample" in n for n in card.notes)


def test_empty_signal():
    card = generate_validation_card("empty", [], {}, [])
    assert card.total_trades == 0
    assert card.fragility_score == 100
    assert not card.is_robust


# ---------------------------------------------------------------------------
# Check 4 — slippage gate must be DIRECTIONAL (regression for the sign-blind
# ratio that passed losers and failed thin winners)
# ---------------------------------------------------------------------------

def _card_from(returns, model="mean_reversion"):
    """Build a card the way live does: stressed returns = 10bp haircut."""
    from src.backtest.validation_card import generate_validation_card
    return generate_validation_card(
        signal_model=model,
        trade_returns=returns,
        trade_returns_by_regime={"bull": returns},
        slippage_returns=[r - 0.10 for r in returns],
        variants_tested=1,
    )


def _slippage_check(card):
    from src.backtest.validation_card import run_validation_checks
    payload = run_validation_checks(
        signal_dates=[date(2026, 8, 1)] * card.total_trades,
        execution_dates=[date(2026, 8, 2)] * card.total_trades,
        run_date=date(2026, 8, 1),
        feature_columns=["rsi_2"],
        validation_card=card,
    )
    return payload.checks.get("slippage_sensitivity_check")


def test_slippage_check_fails_a_losing_stream():
    """A stream losing 1%/trade must FAIL. The old ratio scored it 0.10 → pass."""
    card = _card_from([-1.0] * 40)
    assert card.slippage_sensitivity < 0.5      # the old statistic said "robust"
    assert _slippage_check(card) == "fail"


def test_slippage_check_passes_a_thin_but_real_winner():
    """+0.15%/trade survives a 10bp haircut and must PASS.

    The old ratio scored 0.67 (>0.5) and blocked it — the false-positive class
    that caused the 2026-07 outages.
    """
    card = _card_from([0.15] * 40)
    assert card.slippage_sensitivity > 0.5      # old statistic said "fragile"
    assert _slippage_check(card) == "pass"


def test_slippage_check_fails_when_costs_erase_the_edge():
    """+0.05%/trade does NOT survive a 10bp haircut → fail."""
    assert _slippage_check(_card_from([0.05] * 40)) == "fail"


def test_slippage_check_auto_passes_under_30_trades():
    assert _slippage_check(_card_from([-1.0] * 20)) == "pass"
