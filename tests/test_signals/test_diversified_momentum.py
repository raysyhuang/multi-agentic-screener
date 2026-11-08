from datetime import date, timedelta
import math

import pandas as pd

from src.backtest.diversified_momentum import build_diversified_momentum_rebalances
from src.signals.diversified_momentum import (
    DiversifiedMomentumRule,
    build_diversified_momentum_panel,
    scheduled_signal_dates,
    select_diversified_momentum,
)


def _prices(returns: list[float]) -> pd.DataFrame:
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(len(returns) + 1)]
    closes = [100.0]
    for value in returns:
        closes.append(closes[-1] * (1.0 + value))
    return pd.DataFrame({"date": dates, "close": closes})


def _trending_prices(mean_log_return: float, noise: float, sessions: int = 279) -> pd.DataFrame:
    log_returns = [mean_log_return + (noise if i % 2 else -noise) for i in range(sessions)]
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(sessions + 1)]
    closes = [100.0]
    for value in log_returns:
        closes.append(closes[-1] * math.exp(value))
    return pd.DataFrame({"date": dates, "close": closes})


def test_volatility_penalty_prefers_smoother_stock_with_same_raw_gain():
    smooth_prices = _trending_prices(0.001, 0.0001)
    bumpy_prices = _trending_prices(0.001, 0.01)
    panel = build_diversified_momentum_panel({"SMOOTH": smooth_prices, "BUMPY": bumpy_prices})
    signal_date = smooth_prices.iloc[-1]["date"]
    rule = DiversifiedMomentumRule(selected_fraction=0.5)

    picks = select_diversified_momentum(panel, signal_date, {"SMOOTH", "BUMPY"}, rule)

    assert [pick.ticker for pick in picks] == ["SMOOTH"]


def test_selects_rounded_top_quintile_and_fails_closed_on_membership():
    price_data = {
        f"S{i}": _trending_prices(0.0005 + i * 0.0001, 0.00005)
        for i in range(10)
    }
    panel = build_diversified_momentum_panel(price_data)
    signal_date = next(iter(price_data.values())).iloc[-1]["date"]

    picks = select_diversified_momentum(panel, signal_date, set(price_data))

    assert len(picks) == 2
    assert select_diversified_momentum(panel, signal_date, set()) == []


def test_schedule_has_only_february_and_august_month_ends():
    calendar = [
        date(2025, 2, 27), date(2025, 2, 28), date(2025, 3, 3),
        date(2025, 8, 28), date(2025, 8, 29), date(2025, 9, 2),
    ]
    assert scheduled_signal_dates(calendar, calendar[0], calendar[-1]) == [
        date(2025, 2, 28), date(2025, 8, 29),
    ]


def test_rebalance_executes_next_session_at_equal_weights():
    prices = {
        f"S{i}": _trending_prices(0.001 + i * 0.0001, 0.00005)
        for i in range(10)
    }
    calendar = list(next(iter(prices.values()))["date"])
    rule = DiversifiedMomentumRule(signal_months=(9, 12))
    signal_date = scheduled_signal_dates(calendar, calendar[-40], calendar[-2], rule)[-1]
    rows = build_diversified_momentum_rebalances(
        prices,
        calendar,
        {signal_date: set(prices)},
        start=signal_date,
        end=signal_date,
        rule=rule,
    )
    assert len(rows) == 1
    assert rows[0].execution_date > signal_date
    assert len(rows[0].tickers) == 2
    assert rows[0].target_weight == 0.5
