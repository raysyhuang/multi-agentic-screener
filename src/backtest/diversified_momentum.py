"""Target generation for the frozen diversified-momentum research rule."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.signals.diversified_momentum import (
    DiversifiedMomentumRule,
    build_diversified_momentum_panel,
    scheduled_signal_dates,
    select_diversified_momentum,
)


@dataclass(frozen=True)
class DiversifiedMomentumRebalance:
    signal_date: date
    execution_date: date
    tickers: tuple[str, ...]
    target_weight: float


def build_diversified_momentum_rebalances(
    price_data: dict[str, pd.DataFrame],
    market_calendar: list[date],
    eligible_by_date: dict[date, set[str]],
    *,
    start: date,
    end: date,
    rule: DiversifiedMomentumRule = DiversifiedMomentumRule(),
) -> list[DiversifiedMomentumRebalance]:
    """Create two scheduled, next-session, equal-weight rebalances per year."""
    if start > end:
        raise ValueError("start must be on or before end")
    calendar = sorted(set(market_calendar))
    next_session = {calendar[i]: calendar[i + 1] for i in range(len(calendar) - 1)}
    panel = build_diversified_momentum_panel(price_data, rule)
    rows: list[DiversifiedMomentumRebalance] = []
    for signal_date in scheduled_signal_dates(calendar, start, end, rule):
        execution_date = next_session.get(signal_date)
        eligible = eligible_by_date.get(signal_date)
        if execution_date is None or eligible is None:
            continue
        picks = select_diversified_momentum(panel, signal_date, eligible, rule)
        if not picks:
            continue
        tickers = tuple(pick.ticker for pick in picks)
        rows.append(DiversifiedMomentumRebalance(
            signal_date=signal_date,
            execution_date=execution_date,
            tickers=tickers,
            target_weight=1.0 / len(tickers),
        ))
    return rows
