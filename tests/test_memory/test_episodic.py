"""Tests for episodic memory — uses dataclass construction (no DB needed)."""

from src.memory.episodic import TickerHistory, ModelPerformance, EpisodicContext


def test_ticker_history_defaults():
    th = TickerHistory(ticker="AAPL")
    assert th.times_signaled == 0
    assert th.win_rate is None
    assert th.recent_outcomes == []


def test_ticker_history_with_data():
    th = TickerHistory(
        ticker="AAPL",
        times_signaled=5,
        times_approved=3,
        times_vetoed=2,
        win_count=2,
        loss_count=1,
        win_rate=66.7,
        avg_pnl_pct=2.5,
        recent_outcomes=[
            {"pnl_pct": 5.0, "exit_reason": "target"},
            {"pnl_pct": -2.0, "exit_reason": "stop"},
        ],
    )
    assert th.times_signaled == 5
    assert th.win_rate == 66.7
    assert len(th.recent_outcomes) == 2


def test_model_performance_defaults():
    mp = ModelPerformance(signal_model="breakout", regime="bull")
    assert mp.total_signals == 0
    assert mp.win_rate is None


def test_model_performance_with_data():
    mp = ModelPerformance(
        signal_model="breakout",
        regime="bull",
        total_signals=20,
        win_count=12,
        loss_count=8,
        win_rate=60.0,
        avg_pnl_pct=1.8,
        avg_max_adverse=-3.5,
    )
    assert mp.win_rate == 60.0
    assert mp.avg_max_adverse == -3.5


def test_episodic_context_combines():
    ctx = EpisodicContext(
        ticker_history=TickerHistory(ticker="AAPL", times_signaled=3),
        model_performance=ModelPerformance(signal_model="breakout", regime="bull", total_signals=10),
    )
    assert ctx.ticker_history.ticker == "AAPL"
    assert ctx.model_performance.total_signals == 10
