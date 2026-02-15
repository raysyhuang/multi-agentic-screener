"""Tests for portfolio construction and position sizing."""

from src.portfolio.construct import (
    build_trade_plan,
    PortfolioConfig,
    SizingMethod,
)


def _make_candidate(ticker, confidence=70, entry=100, stop=95, target=110, atr_pct=5.0):
    return {
        "ticker": ticker,
        "direction": "LONG",
        "entry_price": entry,
        "stop_loss": stop,
        "target_1": target,
        "confidence": confidence,
        "signal_model": "breakout",
        "holding_period": 10,
        "atr_pct": atr_pct,
        "avg_daily_volume": 1_000_000,
    }


def test_basic_trade_plan():
    candidates = [_make_candidate("AAPL"), _make_candidate("MSFT", confidence=65)]
    plans = build_trade_plan(candidates)
    assert len(plans) == 2
    assert plans[0].ticker == "AAPL"  # Higher confidence first
    assert all(p.weight_pct > 0 for p in plans)
    assert all(p.shares > 0 for p in plans)


def test_empty_candidates():
    plans = build_trade_plan([])
    assert plans == []


def test_confidence_filter():
    config = PortfolioConfig(min_confidence=60)
    candidates = [
        _make_candidate("AAPL", confidence=70),
        _make_candidate("LOW", confidence=30),
    ]
    plans = build_trade_plan(candidates, config=config)
    assert len(plans) == 1
    assert plans[0].ticker == "AAPL"


def test_max_positions():
    config = PortfolioConfig(max_positions=2)
    candidates = [_make_candidate(f"T{i}", confidence=60 + i) for i in range(5)]
    plans = build_trade_plan(candidates, config=config)
    assert len(plans) == 2


def test_bear_regime_reduces_weight():
    candidates = [_make_candidate("AAPL", confidence=80)]
    bull_plans = build_trade_plan(candidates, regime="bull")
    bear_plans = build_trade_plan(candidates, regime="bear")
    # Bear regime exposure multiplier is 0.5 vs bull's 1.0
    assert bear_plans[0].weight_pct < bull_plans[0].weight_pct


def test_kelly_sizing():
    config = PortfolioConfig(sizing_method=SizingMethod.KELLY)
    candidates = [_make_candidate("AAPL", confidence=80, entry=100, stop=95, target=115)]
    plans = build_trade_plan(candidates, config=config)
    assert len(plans) == 1
    assert plans[0].weight_pct > 0


def test_equal_sizing():
    config = PortfolioConfig(sizing_method=SizingMethod.EQUAL)
    candidates = [_make_candidate("AAPL"), _make_candidate("MSFT")]
    plans = build_trade_plan(candidates, config=config)
    assert len(plans) == 2
    # Equal weight: 50% each (before regime/quality adjustments)
    assert abs(plans[0].weight_pct - plans[1].weight_pct) < 5  # Roughly equal


def test_volatility_sizing_high_vol_smaller():
    """Higher ATR% should produce smaller position sizes."""
    candidates_low_vol = [_make_candidate("LOW", atr_pct=3.0)]
    candidates_high_vol = [_make_candidate("HIGH", atr_pct=10.0)]
    config = PortfolioConfig(sizing_method=SizingMethod.VOLATILITY)

    low_plans = build_trade_plan(candidates_low_vol, config=config)
    high_plans = build_trade_plan(candidates_high_vol, config=config)

    # Both should succeed
    assert len(low_plans) == 1
    assert len(high_plans) == 1
    # Low vol → larger position (capped at max)
    assert low_plans[0].weight_pct >= high_plans[0].weight_pct


def test_reward_risk_ratio():
    candidates = [_make_candidate("AAPL", entry=100, stop=95, target=115)]
    plans = build_trade_plan(candidates)
    assert len(plans) == 1
    # R:R = (115-100)/(100-95) = 3.0
    assert plans[0].reward_risk_ratio == 3.0


def test_position_clamped_to_bounds():
    config = PortfolioConfig(min_position_pct=5.0, max_position_pct=20.0)
    candidates = [_make_candidate("AAPL", atr_pct=0.1)]  # Tiny ATR → huge raw weight
    plans = build_trade_plan(candidates, config=config)
    assert plans[0].weight_pct <= 20.0
