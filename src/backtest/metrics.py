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


def deflated_sharpe_ratio(
    observed_sharpe: float,
    num_trials: int,
    returns: list[float],
    annualization_factor: float = 50.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (Bailey & López de Prado).

    Penalizes the observed Sharpe ratio for the number of strategy variants
    tested (selection bias). Returns the probability that the observed Sharpe
    exceeds zero after correcting for multiple testing.

    Args:
        observed_sharpe: The Sharpe ratio of the selected strategy.
        num_trials: Number of strategy variants tested (parameter combos).
        returns: Trade-level returns used to estimate skewness/kurtosis.
        annualization_factor: Trade frequency per year (default 50 for short-term).

    Returns:
        DSR as a probability (0-1). Values > 0.95 suggest the Sharpe is real.
    """
    from scipy import stats

    if num_trials <= 1 or len(returns) < 10 or observed_sharpe <= 0:
        return 0.0

    arr = np.array(returns)
    n = len(arr)

    # Skewness and excess kurtosis of returns
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))  # excess kurtosis

    # Expected maximum Sharpe under the null (all trials are noise)
    # E[max(Z)] ≈ (1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N*e))
    # Simplified: E[max] ≈ √(2 * log(N)) - (log(π) + log(log(N))) / (2 * √(2 * log(N)))
    log_n = np.log(num_trials)
    if log_n <= 0:
        return 0.0

    e_max_z = np.sqrt(2 * log_n) - (np.log(np.pi) + np.log(log_n)) / (2 * np.sqrt(2 * log_n))

    # Variance of the Sharpe ratio estimator (Lo, 2002)
    # Var(SR) ≈ (1 + 0.5 * SR² - skew * SR + (kurt/4) * SR²) / (n - 1)
    sr = observed_sharpe / np.sqrt(annualization_factor)  # de-annualize
    var_sr = (1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / max(1, n - 1)
    std_sr = np.sqrt(max(var_sr, 1e-10))

    # DSR = P(SR > 0 | observed, trials) = Φ((SR - E[max]) / std(SR))
    dsr = float(stats.norm.cdf((sr - e_max_z) / std_sr))

    return round(max(0.0, min(1.0, dsr)), 4)


def _empty_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_trades=0, win_rate=0, avg_return_pct=0, median_return_pct=0,
        total_return_pct=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
        max_drawdown_pct=0, profit_factor=0, avg_win_pct=0, avg_loss_pct=0,
        max_consecutive_wins=0, max_consecutive_losses=0, expectancy=0, payoff_ratio=0,
    )
