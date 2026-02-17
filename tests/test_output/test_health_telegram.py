"""Tests for health alert Telegram formatting."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from src.contracts import HealthComponent, HealthState, PositionHealthCard
from src.output.telegram import format_health_alert, format_near_miss_resolution_alert


def _make_component(name: str, score: float, weight: float = 0.20) -> HealthComponent:
    return HealthComponent(
        name=name,
        score=score,
        weight=weight,
        weighted_score=round(score * weight, 2),
        details={},
    )


def _make_card(
    ticker: str = "AAPL",
    state: HealthState = HealthState.WATCH,
    previous_state: HealthState = HealthState.ON_TRACK,
    score: float = 55.0,
    pnl: float = 2.5,
    days: int = 5,
    expected: int = 10,
    invalidation_reason: str | None = None,
    weakest_name: str = "volume",
    weakest_score: float = 20.0,
    score_velocity: float | None = None,
) -> PositionHealthCard:
    return PositionHealthCard(
        trend_health=_make_component("trend", 70.0, 0.30),
        momentum_health=_make_component("momentum", 60.0, 0.25),
        volume_confirmation=_make_component(weakest_name, weakest_score, 0.15),
        risk_integrity=_make_component("risk", 65.0, 0.20),
        regime_alignment=_make_component("regime", 80.0, 0.10),
        promising_score=score,
        state=state,
        previous_state=previous_state,
        state_changed=True,
        score_velocity=score_velocity,
        hard_invalidation=invalidation_reason is not None,
        invalidation_reason=invalidation_reason,
        days_held=days,
        expected_hold_days=expected,
        pnl_pct=pnl,
        mfe_pct=5.0,
        mae_pct=-1.0,
        current_price=110.0,
        atr_14=2.0,
        atr_stop_distance=1.5,
        signal_id=1,
        ticker=ticker,
        signal_model="breakout",
        as_of_date=date.today(),
    )


class TestFormatHealthAlert:
    def test_state_change_formatting(self):
        card = _make_card()
        msg = format_health_alert([card])
        assert "AAPL" in msg
        # State transition shown as emojis: on_track=✅, watch=⚠️
        assert "\u2705" in msg  # on_track emoji
        assert "\u26a0" in msg  # watch emoji
        assert "55/100" in msg
        assert "+2.50%" in msg
        assert "Day 5/10" in msg

    def test_empty_input_returns_empty(self):
        assert format_health_alert([]) == ""

    def test_invalidation_reason_displayed(self):
        card = _make_card(
            state=HealthState.EXIT,
            invalidation_reason="breakout_range_failure",
        )
        msg = format_health_alert([card])
        assert "breakout_range_failure" in msg
        assert "Invalidation" in msg

    def test_weakest_component_shown(self):
        card = _make_card(weakest_name="volume", weakest_score=15.0)
        msg = format_health_alert([card])
        assert "volume" in msg
        assert "15/100" in msg

    def test_multiple_cards(self):
        cards = [
            _make_card(ticker="AAPL"),
            _make_card(ticker="MSFT", pnl=-3.0),
        ]
        msg = format_health_alert(cards)
        assert "AAPL" in msg
        assert "MSFT" in msg
        assert "-3.00%" in msg

    def test_velocity_displayed_when_present(self):
        card = _make_card(score_velocity=-7.5)
        msg = format_health_alert([card])
        assert "Velocity:" in msg
        assert "-7.5" in msg

    def test_velocity_not_displayed_when_none(self):
        card = _make_card(score_velocity=None)
        msg = format_health_alert([card])
        assert "Velocity:" not in msg


class TestFormatNearMissResolution:
    def test_basic_formatting(self):
        resolved = [
            {"ticker": "AAPL", "counterfactual_return": 5.2, "exit_reason": "target"},
            {"ticker": "MSFT", "counterfactual_return": -3.1, "exit_reason": "stop"},
        ]
        msg = format_near_miss_resolution_alert(resolved)
        assert "AAPL" in msg
        assert "MSFT" in msg
        assert "+5.20%" in msg
        assert "-3.10%" in msg
        assert "target" in msg
        assert "stop" in msg

    def test_empty_returns_empty(self):
        assert format_near_miss_resolution_alert([]) == ""

    def test_win_rate_commentary_winners(self):
        # All winners → should note filtering profitable trades
        resolved = [
            {"ticker": "AAPL", "counterfactual_return": 5.0, "exit_reason": "target"},
            {"ticker": "MSFT", "counterfactual_return": 3.0, "exit_reason": "target"},
        ]
        msg = format_near_miss_resolution_alert(resolved)
        assert "blocked profitable" in msg

    def test_win_rate_commentary_losers(self):
        # All losers → should note correct filtering
        resolved = [
            {"ticker": "AAPL", "counterfactual_return": -5.0, "exit_reason": "stop"},
            {"ticker": "MSFT", "counterfactual_return": -3.0, "exit_reason": "stop"},
        ]
        msg = format_near_miss_resolution_alert(resolved)
        assert "correctly blocked" in msg

    def test_shows_count_and_stats(self):
        resolved = [
            {"ticker": "AAPL", "counterfactual_return": 5.0, "exit_reason": "target"},
        ]
        msg = format_near_miss_resolution_alert(resolved)
        assert "Resolved: <b>1</b>" in msg
        assert "WR:" in msg
        assert "<b>100%</b>" in msg
