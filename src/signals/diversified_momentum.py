"""A small, diversified momentum rule inspired by public factor indices.

This is not an SPMO clone.  It keeps only the parts that are both economically
clear and reproducible with point-in-time data: 12-1 momentum, a volatility
penalty, broad selection, equal weights, and two scheduled reviews per year.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import pandas as pd

from src.signals.simple_momentum import _normalise_closes


@dataclass(frozen=True)
class DiversifiedMomentumRule:
    formation_sessions: int = 252
    skip_recent_sessions: int = 21
    selected_fraction: float = 0.20
    min_price: float = 5.0
    signal_months: tuple[int, int] = (2, 8)
    cost_bps_per_side: float = 25.0

    def __post_init__(self) -> None:
        if self.formation_sessions <= self.skip_recent_sessions + 1:
            raise ValueError("formation_sessions must exceed the skip window")
        if not 0.0 < self.selected_fraction <= 1.0:
            raise ValueError("selected_fraction must be in (0, 1]")
        if self.min_price <= 0:
            raise ValueError("min_price must be positive")
        if len(set(self.signal_months)) != 2 or any(
            month < 1 or month > 12 for month in self.signal_months
        ):
            raise ValueError("signal_months must contain two distinct calendar months")
        if self.cost_bps_per_side < 0:
            raise ValueError("cost cannot be negative")


@dataclass(frozen=True)
class DiversifiedMomentumPick:
    ticker: str
    signal_date: date
    momentum_12_1: float
    daily_volatility: float
    risk_adjusted_momentum: float
    close: float
    rank: int


def build_diversified_momentum_panel(
    price_data: dict[str, pd.DataFrame],
    rule: DiversifiedMomentumRule = DiversifiedMomentumRule(),
) -> pd.DataFrame:
    """Calculate the signal using only prices observable at each close."""
    rows: list[pd.DataFrame] = []
    lookback = rule.formation_sessions - rule.skip_recent_sessions
    for ticker in sorted(price_data):
        df = _normalise_closes(price_data[ticker])
        close = df["close"]
        returns = close.pct_change(fill_method=None)
        volatility = returns.shift(rule.skip_recent_sessions).rolling(lookback).std(ddof=1)
        momentum = (
            close.shift(rule.skip_recent_sessions)
            / close.shift(rule.formation_sessions)
            - 1.0
        )
        rows.append(pd.DataFrame({
            "date": df["date"],
            "ticker": ticker,
            "close": close,
            "momentum_12_1": momentum,
            "daily_volatility": volatility,
            "risk_adjusted_momentum": momentum / volatility,
        }))
    columns = [
        "date", "ticker", "close", "momentum_12_1",
        "daily_volatility", "risk_adjusted_momentum",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)


def select_diversified_momentum(
    panel: pd.DataFrame,
    signal_date: date,
    eligible_tickers: set[str],
    rule: DiversifiedMomentumRule = DiversifiedMomentumRule(),
) -> list[DiversifiedMomentumPick]:
    """Select the top quintile of the explicit point-in-time universe."""
    if not eligible_tickers:
        return []
    day = panel.loc[
        panel["date"].eq(signal_date)
        & panel["ticker"].isin(eligible_tickers)
        & panel["close"].ge(rule.min_price)
    ].dropna(subset=["momentum_12_1", "daily_volatility", "risk_adjusted_momentum"])
    day = day.loc[
        day["daily_volatility"].gt(0.0)
        & day["risk_adjusted_momentum"].map(math.isfinite)
    ]
    count = int(math.floor(len(day) * rule.selected_fraction + 0.5))
    if count == 0:
        return []
    ranked = day.sort_values(
        ["risk_adjusted_momentum", "ticker"],
        ascending=[False, True],
        kind="mergesort",
    ).head(count)
    return [
        DiversifiedMomentumPick(
            ticker=str(row.ticker),
            signal_date=signal_date,
            momentum_12_1=float(row.momentum_12_1),
            daily_volatility=float(row.daily_volatility),
            risk_adjusted_momentum=float(row.risk_adjusted_momentum),
            close=float(row.close),
            rank=rank,
        )
        for rank, row in enumerate(ranked.itertuples(index=False), start=1)
    ]


def scheduled_signal_dates(
    calendar: list[date],
    start: date,
    end: date,
    rule: DiversifiedMomentumRule = DiversifiedMomentumRule(),
) -> list[date]:
    """Last market session of each configured signal month."""
    last_by_month: dict[tuple[int, int], date] = {}
    for session in sorted(set(calendar)):
        if start <= session <= end and session.month in rule.signal_months:
            last_by_month[(session.year, session.month)] = session
    return list(last_by_month.values())
