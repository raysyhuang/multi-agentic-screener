"""Leakage-resistant order generation for the MAS v2 momentum baseline.

Portfolio accounting remains in the common account engine.  This adapter owns
only the strategy decisions: month-end signal, next-session execution, and the
equal-weight target list.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd

from src.signals.simple_momentum import (
    SimpleMomentumRule,
    build_momentum_panel,
    monthly_signal_dates,
    select_simple_momentum,
)


@dataclass(frozen=True)
class MomentumRebalance:
    signal_date: date
    execution_date: date
    tickers: tuple[str, ...]
    target_weight: float
    ranks: tuple[int, ...]
    momentum_12_1: tuple[float, ...]


def build_monthly_rebalances(
    price_data: dict[str, pd.DataFrame],
    market_calendar: list[date],
    eligible_by_date: dict[date, set[str]],
    *,
    start: date,
    end: date,
    rule: SimpleMomentumRule = SimpleMomentumRule(),
) -> list[MomentumRebalance]:
    """Build auditable target portfolios without silently inventing PIT data."""
    if start > end:
        raise ValueError("start must be on or before end")
    calendar = sorted(set(market_calendar))
    next_session = {calendar[i]: calendar[i + 1] for i in range(len(calendar) - 1)}
    panel = build_momentum_panel(price_data, rule)
    out: list[MomentumRebalance] = []
    for signal_date in monthly_signal_dates(calendar, start, end):
        execution_date = next_session.get(signal_date)
        eligible = eligible_by_date.get(signal_date)
        if execution_date is None or eligible is None:
            continue
        picks = select_simple_momentum(panel, signal_date, eligible, rule)
        if not picks:
            continue
        out.append(MomentumRebalance(
            signal_date=signal_date,
            execution_date=execution_date,
            tickers=tuple(p.ticker for p in picks),
            target_weight=1.0 / len(picks),
            ranks=tuple(p.rank for p in picks),
            momentum_12_1=tuple(p.momentum_12_1 for p in picks),
        ))
    return out


def serialise_rebalances(
    rebalances: list[MomentumRebalance],
    rule: SimpleMomentumRule,
) -> dict:
    """Small deterministic artifact for review before any return is calculated."""
    return {
        "strategy": "mas_v2_simple_momentum",
        "rule": asdict(rule),
        "rebalances": [asdict(row) for row in rebalances],
    }
