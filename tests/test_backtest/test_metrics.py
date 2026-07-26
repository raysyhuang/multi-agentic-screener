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


def test_deflated_sharpe_discriminates_real_edge_from_noise():
    """DSR must be high for an obviously-real edge and low for noise.

    Regression for a dimensional bug: the DSR compared a per-trade Sharpe (~0.3)
    directly to an expected-max z-score (~1.8) before dividing by the Sharpe SE,
    which forced DSR≈0 for EVERY per-trade strategy — a t-stat-17 edge scored 0.00.
    """
    import numpy as np

    from src.backtest.metrics import deflated_sharpe_ratio

    rng = np.random.default_rng(0)
    # Obviously-real edge: mean +0.8%, sd 1% → per-trade SR ≈ 0.8, t-stat ≈ 17.
    real = (0.8 + rng.standard_normal(500) * 1.0).tolist()
    per_trade_sr = float(np.mean(real) / np.std(real, ddof=1))
    ann = per_trade_sr * np.sqrt(50)
    dsr_real = deflated_sharpe_ratio(ann, num_trials=6, returns=real)
    assert dsr_real > 0.95, f"real edge should score high, got {dsr_real}"

    # Pure noise: mean 0 → Sharpe ~0 → DSR should be small.
    noise = (rng.standard_normal(500) * 1.0).tolist()
    ann_noise = float(np.mean(noise) / np.std(noise, ddof=1)) * np.sqrt(50)
    dsr_noise = deflated_sharpe_ratio(ann_noise, num_trials=6, returns=noise)
    assert dsr_noise < 0.5, f"noise should score low, got {dsr_noise}"


def test_deflated_sharpe_penalizes_more_trials():
    """More variants searched → same edge is deflated toward a lower DSR."""
    import numpy as np

    from src.backtest.metrics import deflated_sharpe_ratio

    rng = np.random.default_rng(1)
    # A marginal edge so the multiple-testing penalty is visible (not saturated).
    r = (0.15 + rng.standard_normal(400) * 1.0).tolist()
    ann = float(np.mean(r) / np.std(r, ddof=1)) * np.sqrt(50)
    dsr_few = deflated_sharpe_ratio(ann, num_trials=2, returns=r)
    dsr_many = deflated_sharpe_ratio(ann, num_trials=100, returns=r)
    assert dsr_few >= dsr_many
