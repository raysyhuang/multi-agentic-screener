"""Tests for backtest metrics."""

from src.backtest.metrics import compute_metrics


def test_compute_metrics_all_wins():
    returns = [2.0, 3.0, 1.5, 4.0, 2.5]
    m = compute_metrics(returns)
    assert m.win_rate == 1.0
    assert m.avg_return_pct > 0
    assert m.max_consecutive_wins == 5
    assert m.max_consecutive_losses == 0
    assert m.profit_factor == float("inf")


def test_compute_metrics_mixed():
    returns = [5.0, -2.0, 3.0, -1.0, 4.0, -3.0, 2.0]
    m = compute_metrics(returns)
    assert 0 < m.win_rate < 1
    assert m.profit_factor > 0
    assert m.max_drawdown_pct > 0
    assert m.expectancy != 0


def test_compute_metrics_all_losses():
    returns = [-1.0, -2.0, -3.0]
    m = compute_metrics(returns)
    assert m.win_rate == 0.0
    assert m.avg_return_pct < 0
    assert m.max_consecutive_losses == 3


def test_compute_metrics_empty():
    m = compute_metrics([])
    assert m.total_trades == 0
    assert m.win_rate == 0


def test_sharpe_and_sortino_positive_for_good_returns():
    returns = [1.0, 2.0, 1.5, 3.0, 0.5, 2.0, 1.0]
    m = compute_metrics(returns)
    assert m.sharpe_ratio > 0
    assert m.sortino_ratio > 0
