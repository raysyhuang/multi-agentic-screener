"""Tests for signal ranking."""

import numpy as np
import pandas as pd

from src.features.regime import Regime
from src.signals.breakout import BreakoutSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.ranker import rank_candidates, deduplicate_signals, filter_correlated_picks, RankedCandidate


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
