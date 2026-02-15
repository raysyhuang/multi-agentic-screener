"""Tests for signal ranking, confluence detection, and cooldown."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.features.regime import Regime
from src.signals.breakout import BreakoutSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.ranker import (
    rank_candidates,
    deduplicate_signals,
    filter_correlated_picks,
    detect_confluence,
    apply_confluence_bonus,
    apply_cooldown,
    RankedCandidate,
)


def _make_breakout(ticker: str, score: float) -> BreakoutSignal:
    return BreakoutSignal(
        ticker=ticker, score=score, direction="LONG",
        entry_price=100, stop_loss=95, target_1=110, target_2=115,
        holding_period=10, components={},
    )


def _make_mean_rev(ticker: str, score: float) -> MeanReversionSignal:
    return MeanReversionSignal(
        ticker=ticker, score=score, direction="LONG",
        entry_price=50, stop_loss=48, target_1=53, target_2=55,
        holding_period=5, components={},
    )


def test_rank_respects_regime_gate():
    signals = [
        _make_breakout("AAPL", 80),
        _make_mean_rev("MSFT", 75),
    ]
    # In BEAR regime, breakout should be filtered out
    result = rank_candidates(signals, Regime.BEAR, {}, top_n=10)
    tickers = [r.ticker for r in result]
    assert "AAPL" not in tickers  # breakout blocked in bear
    assert "MSFT" in tickers


def test_rank_sorts_by_adjusted_score():
    signals = [
        _make_breakout("LOW", 55),
        _make_breakout("HIGH", 90),
        _make_breakout("MID", 70),
    ]
    result = rank_candidates(signals, Regime.BULL, {}, top_n=10)
    assert result[0].ticker == "HIGH"
    assert result[-1].ticker == "LOW"


def test_rank_respects_top_n():
    signals = [_make_breakout(f"T{i}", 60 + i) for i in range(20)]
    result = rank_candidates(signals, Regime.BULL, {}, top_n=5)
    assert len(result) == 5


def test_deduplicate_keeps_best():
    signals = [
        _make_breakout("AAPL", 80),
        _make_mean_rev("AAPL", 90),  # same ticker, higher score
    ]
    result = deduplicate_signals(signals)
    assert len(result) == 1
    assert result[0].score == 90


def _make_ranked(ticker: str, score: float) -> RankedCandidate:
    return RankedCandidate(
        ticker=ticker, signal_model="breakout", raw_score=score,
        regime_adjusted_score=score, direction="LONG", entry_price=100,
        stop_loss=95, target_1=110, target_2=None, holding_period=10,
        components={}, features={},
    )


def test_correlation_filter_drops_identical():
    """Two tickers with identical returns should be filtered."""
    np.random.seed(42)
    returns = np.random.randn(30).cumsum() + 100
    price_data = {
        "A": pd.DataFrame({"close": returns}),
        "B": pd.DataFrame({"close": returns}),  # identical
    }
    candidates = [_make_ranked("A", 90), _make_ranked("B", 80)]
    result = filter_correlated_picks(candidates, price_data, max_correlation=0.75)
    assert len(result) == 1
    assert result[0].ticker == "A"


def test_correlation_filter_keeps_uncorrelated():
    """Two uncorrelated tickers should both survive."""
    np.random.seed(42)
    price_data = {
        "A": pd.DataFrame({"close": np.random.randn(30).cumsum() + 100}),
        "B": pd.DataFrame({"close": np.random.randn(30).cumsum() + 200}),
    }
    candidates = [_make_ranked("A", 90), _make_ranked("B", 80)]
    result = filter_correlated_picks(candidates, price_data, max_correlation=0.75)
    assert len(result) == 2


# --- Confluence Detection ---


def test_confluence_single_model():
    """Single model per ticker = no confluence."""
    signals = [_make_breakout("AAPL", 80)]
    result = detect_confluence(signals)
    assert result["AAPL"].confluence_count == 1
    assert result["AAPL"].is_confluence is False
    assert result["AAPL"].confluence_bonus == 0.0


def test_confluence_two_models():
    """Two models flagging same ticker = confluence."""
    signals = [
        _make_breakout("AAPL", 80),
        _make_mean_rev("AAPL", 75),
    ]
    result = detect_confluence(signals)
    assert result["AAPL"].confluence_count == 2
    assert result["AAPL"].is_confluence is True
    assert result["AAPL"].confluence_bonus == 0.10  # 10% per additional model


def test_confluence_bonus_applied():
    """Confluence bonus should increase adjusted scores."""
    signals = [
        _make_breakout("AAPL", 80),
        _make_mean_rev("AAPL", 75),
        _make_breakout("MSFT", 70),
    ]
    confluence = detect_confluence(signals)
    candidates = [
        _make_ranked("AAPL", 80),
        _make_ranked("MSFT", 85),  # Higher base score
    ]
    result = apply_confluence_bonus(candidates, confluence)
    # AAPL gets 10% bonus: 80 * 1.10 = 88 → now beats MSFT's 85
    assert result[0].ticker == "AAPL"
    assert result[0].regime_adjusted_score == 88.0


# --- Signal Cooldown ---


def test_cooldown_no_recent_signals():
    """No recent signals → nothing suppressed."""
    signals = [_make_breakout("AAPL", 80), _make_breakout("MSFT", 70)]
    result = apply_cooldown(signals, recent_signals=[])
    assert len(result) == 2


def test_cooldown_suppresses_recent():
    """Tickers fired within cooldown window should be suppressed."""
    signals = [_make_breakout("AAPL", 80), _make_breakout("MSFT", 70)]
    recent = [
        {"ticker": "AAPL", "run_date": date.today() - timedelta(days=2)},
    ]
    result = apply_cooldown(signals, recent_signals=recent, cooldown_days=5)
    assert len(result) == 1
    assert result[0].ticker == "MSFT"


def test_cooldown_allows_expired():
    """Tickers that fired beyond the cooldown window should pass."""
    signals = [_make_breakout("AAPL", 80)]
    recent = [
        {"ticker": "AAPL", "run_date": date.today() - timedelta(days=10)},
    ]
    result = apply_cooldown(signals, recent_signals=recent, cooldown_days=5)
    assert len(result) == 1
    assert result[0].ticker == "AAPL"


def test_cooldown_string_dates():
    """Cooldown should handle string dates from DB."""
    signals = [_make_breakout("AAPL", 80)]
    recent = [
        {"ticker": "AAPL", "run_date": str(date.today() - timedelta(days=2))},
    ]
    result = apply_cooldown(signals, recent_signals=recent, cooldown_days=5)
    assert len(result) == 0
