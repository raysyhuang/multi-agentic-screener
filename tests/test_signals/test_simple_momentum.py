from datetime import date, timedelta

import pandas as pd
import pytest

from src.backtest.simple_momentum import build_monthly_rebalances
from src.signals.simple_momentum import (
    SimpleMomentumRule,
    build_momentum_panel,
    monthly_signal_dates,
    select_simple_momentum,
)


def _prices(daily_gain: float, sessions: int = 280) -> pd.DataFrame:
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(sessions)]
    close = [100.0 * ((1.0 + daily_gain) ** i) for i in range(sessions)]
    return pd.DataFrame({"date": dates, "open": close, "close": close})


def test_ranks_only_explicit_pit_members_by_12_1_momentum():
    prices = {
        "FAST": _prices(0.003),
        "MID": _prices(0.002),
        "SLOW": _prices(0.001),
    }
    rule = SimpleMomentumRule(top_n=2)
    panel = build_momentum_panel(prices, rule)
    signal_date = prices["FAST"].iloc[-1]["date"]

    picks = select_simple_momentum(panel, signal_date, {"FAST", "MID", "SLOW"}, rule)

    assert [pick.ticker for pick in picks] == ["FAST", "MID"]
    assert [pick.rank for pick in picks] == [1, 2]


def test_missing_membership_fails_closed():
    prices = {"FAST": _prices(0.003)}
    panel = build_momentum_panel(prices)
    signal_date = prices["FAST"].iloc[-1]["date"]

    assert select_simple_momentum(panel, signal_date, set()) == []


def test_equal_scores_use_ticker_as_deterministic_tie_breaker():
    prices = {"BBB": _prices(0.002), "AAA": _prices(0.002)}
    rule = SimpleMomentumRule(top_n=2)
    panel = build_momentum_panel(prices, rule)
    signal_date = prices["AAA"].iloc[-1]["date"]

    picks = select_simple_momentum(panel, signal_date, set(prices), rule)

    assert [pick.ticker for pick in picks] == ["AAA", "BBB"]


def test_monthly_schedule_uses_last_observed_session():
    calendar = [date(2025, 1, 30), date(2025, 1, 31), date(2025, 2, 3), date(2025, 2, 28)]
    assert monthly_signal_dates(calendar, calendar[0], calendar[-1]) == [
        date(2025, 1, 31), date(2025, 2, 28),
    ]


def test_rebalance_executes_on_next_market_session_and_equal_weights():
    prices = {"FAST": _prices(0.003), "MID": _prices(0.002)}
    calendar = list(prices["FAST"]["date"])
    # Keep one session after the signal window so execution can be scheduled.
    signal_date = monthly_signal_dates(calendar, calendar[-35], calendar[-2])[-1]
    rule = SimpleMomentumRule(top_n=2)
    eligible = {signal_date: set(prices)}

    rows = build_monthly_rebalances(
        prices, calendar, eligible, start=signal_date, end=signal_date, rule=rule,
    )

    assert len(rows) == 1
    assert rows[0].execution_date > rows[0].signal_date
    assert rows[0].tickers == ("FAST", "MID")
    assert rows[0].target_weight == 0.5


def test_rule_rejects_invalid_hidden_variants():
    with pytest.raises(ValueError, match="formation_sessions"):
        SimpleMomentumRule(formation_sessions=20, skip_recent_sessions=21)
