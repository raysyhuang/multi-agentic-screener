import pytest

from src.research.momentum_benchmarks import assert_actual_only, benchmark_tickers


def test_decisive_benchmarks_are_frozen():
    assert benchmark_tickers(decisive_only=True) == ("SPY", "SPMO", "MTUM", "VFMO")


def test_rejects_backfilled_index_history_labeled_as_etf_return():
    with pytest.raises(ValueError, match="actual total returns"):
        assert_actual_only({"SPMO": "backfilled_index", "SPY": "actual_etf_total_return"})


def test_accepts_actual_total_return_series():
    assert_actual_only({ticker: "actual_etf_total_return" for ticker in benchmark_tickers()})
