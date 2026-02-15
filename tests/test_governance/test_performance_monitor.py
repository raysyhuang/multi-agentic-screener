"""Tests for decay detection and rolling performance metrics."""

from src.governance.performance_monitor import (
    compute_rolling_metrics,
    check_decay,
    DecayThresholds,
    RollingMetrics,
)


def _make_trades(pnls, maes=None, mfes=None):
    """Helper to build trade dicts."""
    trades = []
    for i, pnl in enumerate(pnls):
        trades.append({
            "pnl_pct": pnl,
            "max_adverse": maes[i] if maes else -abs(pnl) * 0.5,
            "max_favorable": mfes[i] if mfes else abs(pnl) * 1.5,
        })
    return trades


def test_rolling_metrics_winning():
    trades = _make_trades([2.0, 3.0, 1.5, -0.5, 2.5])
    m = compute_rolling_metrics(trades)
    assert m.total_trades == 5
    assert m.hit_rate == 0.8  # 4 wins / 5 total
    assert m.expectancy > 0


def test_rolling_metrics_losing():
    trades = _make_trades([-1.0, -2.0, -1.5, -0.5])
    m = compute_rolling_metrics(trades)
    assert m.total_trades == 4
    assert m.hit_rate == 0.0
    assert m.expectancy < 0


def test_rolling_metrics_empty():
    m = compute_rolling_metrics([])
    assert m.total_trades == 0
    assert m.hit_rate == 0.0


def test_rolling_metrics_with_window():
    trades = _make_trades([5.0, 5.0, 5.0, -10.0, -10.0])
    m_all = compute_rolling_metrics(trades)
    m_recent = compute_rolling_metrics(trades, window=2)
    assert m_recent.total_trades == 2
    assert m_recent.hit_rate == 0.0  # last 2 are losses


def test_no_decay_when_healthy():
    baseline = RollingMetrics(hit_rate=0.6, avg_mae_pct=2.0, expectancy=1.5, total_trades=50)
    live = RollingMetrics(hit_rate=0.55, avg_mae_pct=2.2, expectancy=1.2, total_trades=20)
    result = check_decay(live, baseline)
    assert result.is_decaying is False
    assert len(result.triggers) == 0


def test_decay_on_hit_rate_collapse():
    baseline = RollingMetrics(hit_rate=0.6, avg_mae_pct=2.0, expectancy=1.5, total_trades=50)
    live = RollingMetrics(hit_rate=0.2, avg_mae_pct=2.0, expectancy=0.1, total_trades=20)
    result = check_decay(live, baseline)
    assert result.is_decaying is True
    assert any("hit_rate_decay" in t for t in result.triggers)


def test_decay_on_mae_expansion():
    baseline = RollingMetrics(hit_rate=0.6, avg_mae_pct=2.0, expectancy=1.5, total_trades=50)
    live = RollingMetrics(hit_rate=0.55, avg_mae_pct=4.0, expectancy=1.0, total_trades=20)
    result = check_decay(live, baseline)
    assert result.is_decaying is True
    assert any("mae_expansion" in t for t in result.triggers)


def test_decay_on_negative_expectancy():
    baseline = RollingMetrics(hit_rate=0.6, avg_mae_pct=2.0, expectancy=1.5, total_trades=50)
    live = RollingMetrics(hit_rate=0.45, avg_mae_pct=2.0, expectancy=-0.5, total_trades=20)
    result = check_decay(live, baseline)
    assert result.is_decaying is True
    assert any("negative_expectancy" in t for t in result.triggers)


def test_no_decay_insufficient_data():
    baseline = RollingMetrics(hit_rate=0.6, avg_mae_pct=2.0, expectancy=1.5, total_trades=50)
    live = RollingMetrics(hit_rate=0.1, avg_mae_pct=5.0, expectancy=-2.0, total_trades=5)
    result = check_decay(live, baseline)
    assert result.is_decaying is False  # < min_trades_for_check


def test_custom_thresholds():
    baseline = RollingMetrics(hit_rate=0.6, avg_mae_pct=2.0, expectancy=1.5, total_trades=50)
    live = RollingMetrics(hit_rate=0.35, avg_mae_pct=2.5, expectancy=0.3, total_trades=15)
    # Strict thresholds
    strict = DecayThresholds(hit_rate_ratio=0.8, mae_multiplier=1.2, min_expectancy=0.5)
    result = check_decay(live, baseline, thresholds=strict)
    assert result.is_decaying is True
