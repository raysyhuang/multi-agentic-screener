"""Backtest performance metrics — Sharpe, Sortino, max drawdown, profit factor, win rate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PerformanceMetrics:
    total_trades: int
    win_rate: float
    avg_return_pct: float
    median_return_pct: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    expectancy: float  # avg win * win_rate - avg loss * loss_rate
    payoff_ratio: float  # avg win / avg loss


def compute_metrics(returns: list[float]) -> PerformanceMetrics:
    """Compute comprehensive performance metrics from a list of trade returns (%)."""
    if not returns:
        return _empty_metrics()

    arr = np.array(returns)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    total = len(arr)
    win_rate = len(wins) / total if total > 0 else 0

    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
    loss_rate = 1 - win_rate

    # Expectancy
    expectancy = avg_win * win_rate + avg_loss * loss_rate  # avg_loss is negative

    # Payoff ratio
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Profit factor
    total_gains = float(np.sum(wins)) if len(wins) > 0 else 0
    total_losses = abs(float(np.sum(losses))) if len(losses) > 0 else 0
    profit_factor = total_gains / total_losses if total_losses > 0 else (float("inf") if total_gains > 0 else 0)

    # Drawdown
    cumulative = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    # Sharpe (annualized, assuming ~50 trades/year for short-term system)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1
    sharpe = float(np.mean(arr)) / std * np.sqrt(50) if std > 0 else 0

    # Sortino
    downside = arr[arr < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1
    sortino = float(np.mean(arr)) / downside_std * np.sqrt(50) if downside_std > 0 else 0

    # Calmar
    calmar = float(np.sum(arr)) / max_dd if max_dd > 0 else 0

    # Consecutive streaks
    max_con_wins, max_con_losses = _consecutive_streaks(returns)

    return PerformanceMetrics(
        total_trades=total,
        win_rate=round(win_rate, 4),
        avg_return_pct=round(float(np.mean(arr)), 4),
        median_return_pct=round(float(np.median(arr)), 4),
        total_return_pct=round(float(np.sum(arr)), 4),
        sharpe_ratio=round(float(sharpe), 4),
        sortino_ratio=round(float(sortino), 4),
        calmar_ratio=round(float(calmar), 4),
        max_drawdown_pct=round(max_dd, 4),
        profit_factor=round(profit_factor, 4),
        avg_win_pct=round(avg_win, 4),
        avg_loss_pct=round(avg_loss, 4),
        max_consecutive_wins=max_con_wins,
        max_consecutive_losses=max_con_losses,
        expectancy=round(expectancy, 4),
        payoff_ratio=round(payoff, 4),
    )


def _consecutive_streaks(returns: list[float]) -> tuple[int, int]:
    max_wins = max_losses = 0
    current_wins = current_losses = 0

    for r in returns:
        if r > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)

    return max_wins, max_losses


def _empty_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_trades=0, win_rate=0, avg_return_pct=0, median_return_pct=0,
        total_return_pct=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
        max_drawdown_pct=0, profit_factor=0, avg_win_pct=0, avg_loss_pct=0,
        max_consecutive_wins=0, max_consecutive_losses=0, expectancy=0, payoff_ratio=0,
    )
