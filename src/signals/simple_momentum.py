"""MAS v2: one plain monthly momentum rule.

This module intentionally does not import the legacy ranker.  A momentum rank
must not be mixed with MR, sniper, confluence, regime multipliers, or a generic
0-100 score whose meaning changes from model to model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class SimpleMomentumRule:
    """Frozen v1 parameters, each with a direct plain-language meaning."""

    formation_sessions: int = 252
    skip_recent_sessions: int = 21
    top_n: int = 20
    min_price: float = 5.0
    cost_bps_per_side: float = 10.0

    def __post_init__(self) -> None:
        if self.formation_sessions <= self.skip_recent_sessions:
            raise ValueError("formation_sessions must exceed skip_recent_sessions")
        if self.top_n < 1:
            raise ValueError("top_n must be positive")
        if self.min_price <= 0:
            raise ValueError("min_price must be positive")
        if self.cost_bps_per_side < 0:
            raise ValueError("cost cannot be negative")


@dataclass(frozen=True)
class SimpleMomentumPick:
    ticker: str
    signal_date: date
    momentum_12_1: float
    close: float
    rank: int


def _normalise_closes(df: pd.DataFrame) -> pd.DataFrame:
    missing = {"date", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"price data missing columns: {sorted(missing)}")
    out = df.loc[:, ["date", "close"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["close"] = out["close"].astype(float)
    return (
        out.drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_momentum_panel(
    price_data: dict[str, pd.DataFrame],
    rule: SimpleMomentumRule = SimpleMomentumRule(),
) -> pd.DataFrame:
    """Calculate 12-1 momentum using only information known at each close."""
    rows: list[pd.DataFrame] = []
    for ticker in sorted(price_data):
        df = _normalise_closes(price_data[ticker])
        close = df["close"]
        panel = pd.DataFrame({
            "date": df["date"],
            "ticker": ticker,
            "close": close,
            "history_sessions": pd.Series(range(1, len(df) + 1), dtype="int64"),
            "momentum_12_1": (
                close.shift(rule.skip_recent_sessions)
                / close.shift(rule.formation_sessions)
                - 1.0
            ),
        })
        rows.append(panel)
    if not rows:
        return pd.DataFrame(
            columns=["date", "ticker", "close", "history_sessions", "momentum_12_1"]
        )
    return pd.concat(rows, ignore_index=True)


def select_simple_momentum(
    panel: pd.DataFrame,
    signal_date: date,
    eligible_tickers: set[str],
    rule: SimpleMomentumRule = SimpleMomentumRule(),
) -> list[SimpleMomentumPick]:
    """Return the top-N 12-1 winners from an explicit PIT universe.

    Missing membership never falls back to today's ticker list.  Missing price
    history is excluded, never imputed.  Equal scores break by ticker ascending.
    """
    if not eligible_tickers:
        return []
    day = panel.loc[
        panel["date"].eq(signal_date)
        & panel["ticker"].isin(eligible_tickers)
        & panel["close"].ge(rule.min_price)
    ].dropna(subset=["momentum_12_1"])
    ranked = day.sort_values(
        ["momentum_12_1", "ticker"],
        ascending=[False, True],
        kind="mergesort",
    ).head(rule.top_n)
    return [
        SimpleMomentumPick(
            ticker=str(row.ticker),
            signal_date=signal_date,
            momentum_12_1=float(row.momentum_12_1),
            close=float(row.close),
            rank=rank,
        )
        for rank, row in enumerate(ranked.itertuples(index=False), start=1)
    ]


def monthly_signal_dates(calendar: list[date], start: date, end: date) -> list[date]:
    """Last observed market session of each completed month in the window."""
    dates = sorted({d for d in calendar if start <= d <= end})
    by_month: dict[tuple[int, int], date] = {}
    for session in dates:
        by_month[(session.year, session.month)] = session
    return list(by_month.values())
