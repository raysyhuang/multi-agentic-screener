"""Walk-forward backtesting engine.

Tests signals across 5d/10d/15d holding periods on historical data.
No look-ahead: signals fire on day T close, execute at T+1 open.
Realistic costs: configurable slippage + commissions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    ticker: str
    signal_model: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    exit_reason: str  # target / stop / expiry
    holding_days: int
    pnl_pct: float
    pnl_after_costs: float
    max_favorable_excursion: float  # best unrealized P&L during trade
    max_adverse_excursion: float  # worst unrealized P&L during trade


@dataclass
class WalkForwardResult:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_pnl_pct: float
    avg_pnl_after_costs: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_holding_days: float
    trades: list[TradeResult]
    by_holding_period: dict[int, dict]


def run_walk_forward(
    signals_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    holding_periods: list[int] | None = None,
) -> WalkForwardResult:
    """Run walk-forward backtest on historical signals.

    Args:
        signals_df: DataFrame with columns: date, ticker, signal_model, direction,
                    entry_price (close on signal day), stop_loss, target_1
        price_data: dict of ticker -> OHLCV DataFrame
        holding_periods: list of holding periods to test (default: [5, 10, 15])

    Returns:
        WalkForwardResult with trade-level detail and aggregate metrics
    """
    settings = get_settings()
    if holding_periods is None:
        holding_periods = settings.holding_periods

    all_trades: list[TradeResult] = []
    by_period: dict[int, list[TradeResult]] = {p: [] for p in holding_periods}

    for _, signal in signals_df.iterrows():
        ticker = signal["ticker"]
        if ticker not in price_data or price_data[ticker].empty:
            continue

        df = price_data[ticker].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date if not isinstance(df["date"].iloc[0], date) else df["date"]

        signal_date = signal["date"] if isinstance(signal["date"], date) else signal["date"].date()

        for period in holding_periods:
            trade = _simulate_trade(
                df=df,
                signal_date=signal_date,
                direction=signal.get("direction", "LONG"),
                stop_loss=signal["stop_loss"],
                target=signal["target_1"],
                max_holding_days=period,
                slippage_pct=settings.slippage_pct,
                commission=settings.commission_per_trade,
                entry_price_hint=signal["entry_price"],
            )
            if trade:
                trade_result = TradeResult(
                    ticker=ticker,
                    signal_model=signal.get("signal_model", "unknown"),
                    **trade,
                )
                all_trades.append(trade_result)
                by_period[period].append(trade_result)

    # Compute aggregate metrics
    if not all_trades:
        return WalkForwardResult(
            total_trades=0, wins=0, losses=0, win_rate=0, avg_pnl_pct=0,
            avg_pnl_after_costs=0, total_return_pct=0, max_drawdown_pct=0,
            sharpe_ratio=0, sortino_ratio=0, profit_factor=0, avg_holding_days=0,
            trades=[], by_holding_period={},
        )

    pnls = [t.pnl_after_costs for t in all_trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)

    # Period-level summaries
    period_summaries = {}
    for period, trades in by_period.items():
        if trades:
            p = [t.pnl_after_costs for t in trades]
            period_summaries[period] = {
                "trades": len(trades),
                "win_rate": sum(1 for x in p if x > 0) / len(p),
                "avg_pnl_pct": np.mean(p),
                "sharpe": _compute_sharpe(p),
            }
        else:
            period_summaries[period] = {"trades": 0, "win_rate": 0, "avg_pnl_pct": 0, "sharpe": 0}

    return WalkForwardResult(
        total_trades=len(all_trades),
        wins=wins,
        losses=losses,
        win_rate=wins / len(all_trades),
        avg_pnl_pct=np.mean([t.pnl_pct for t in all_trades]),
        avg_pnl_after_costs=np.mean(pnls),
        total_return_pct=sum(pnls),
        max_drawdown_pct=_compute_max_drawdown(pnls),
        sharpe_ratio=_compute_sharpe(pnls),
        sortino_ratio=_compute_sortino(pnls),
        profit_factor=_compute_profit_factor(pnls),
        avg_holding_days=np.mean([t.holding_days for t in all_trades]),
        trades=all_trades,
        by_holding_period=period_summaries,
    )


def _simulate_trade(
    df: pd.DataFrame,
    signal_date: date,
    direction: str,
    stop_loss: float,
    target: float,
    max_holding_days: int,
    slippage_pct: float,
    commission: float,
    entry_price_hint: float,
) -> dict | None:
    """Simulate a single trade.

    Entry: T+1 open (first trading day after signal_date).
    Exit: first of target hit / stop hit / max holding period.
    """
    # Find T+1 (first trading day after signal date)
    future = df[df["date"] > signal_date].sort_values("date")
    if future.empty:
        return None

    # Entry at T+1 open with slippage
    entry_row = future.iloc[0]
    entry_price = float(entry_row["open"])
    if direction == "LONG":
        entry_price *= (1 + slippage_pct)
    else:
        entry_price *= (1 - slippage_pct)

    entry_date = entry_row["date"]

    # Walk through subsequent days
    holding_window = future.iloc[:max_holding_days + 1]
    if len(holding_window) < 2:
        return None

    max_favorable = 0.0
    max_adverse = 0.0
    exit_price = None
    exit_date = None
    exit_reason = None

    for i in range(1, len(holding_window)):
        row = holding_window.iloc[i]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if direction == "LONG":
            # Check stop loss (hit during the day)
            if low <= stop_loss:
                exit_price = stop_loss * (1 - slippage_pct)
                exit_date = row["date"]
                exit_reason = "stop"
                break

            # Check target (hit during the day)
            if high >= target:
                exit_price = target * (1 - slippage_pct)
                exit_date = row["date"]
                exit_reason = "target"
                break

            # Track excursions
            day_best = (high - entry_price) / entry_price * 100
            day_worst = (low - entry_price) / entry_price * 100
            max_favorable = max(max_favorable, day_best)
            max_adverse = min(max_adverse, day_worst)

    # If no exit triggered, exit at close of last day
    if exit_price is None:
        last_row = holding_window.iloc[-1]
        exit_price = float(last_row["close"]) * (1 - slippage_pct)
        exit_date = last_row["date"]
        exit_reason = "expiry"

    # Calculate P&L
    if direction == "LONG":
        pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - exit_price) / entry_price * 100

    # Commission impact (as % of trade value, assuming $10K position)
    commission_pct = (commission * 2) / (entry_price * 100) * 100  # round trip
    pnl_after_costs = pnl_pct - commission_pct

    holding_days = (exit_date - entry_date).days if isinstance(exit_date, date) and isinstance(entry_date, date) else max_holding_days

    return {
        "entry_date": entry_date,
        "entry_price": round(entry_price, 2),
        "exit_date": exit_date,
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "holding_days": holding_days,
        "pnl_pct": round(pnl_pct, 4),
        "pnl_after_costs": round(pnl_after_costs, 4),
        "max_favorable_excursion": round(max_favorable, 4),
        "max_adverse_excursion": round(max_adverse, 4),
    }


def _compute_sharpe(returns: list[float], risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    excess = arr - risk_free_rate
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def _compute_sortino(returns: list[float], risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    excess = arr - risk_free_rate
    downside = arr[arr < 0]
    if len(downside) == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(252))


def _compute_profit_factor(returns: list[float]) -> float:
    gains = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _compute_max_drawdown(returns: list[float]) -> float:
    if not returns:
        return 0.0
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
