"""Tests for validation card generation."""

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
