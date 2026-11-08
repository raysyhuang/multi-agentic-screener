"""Frozen benchmark policy for MAS momentum research."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MomentumBenchmark:
    ticker: str
    role: str
    decisive: bool
    universe: str
    method_family: str
    actual_returns_only: bool = True


MOMENTUM_BENCHMARKS = (
    MomentumBenchmark(
        "SPY", "broad US equity opportunity cost", True,
        "US large cap", "market-cap benchmark",
    ),
    MomentumBenchmark(
        "SPMO", "closest investable hurdle for the MAS rule", True,
        "S&P 500", "12-1 risk-adjusted momentum",
    ),
    MomentumBenchmark(
        "MTUM", "alternative broad US momentum implementation", True,
        "US large and mid cap", "multi-horizon risk-adjusted momentum",
    ),
    MomentumBenchmark(
        "VFMO", "low-cost active implementation reference", True,
        "US all cap", "multi-horizon systematic momentum",
    ),
    MomentumBenchmark(
        "PDP", "long-live-history relative-strength reference", False,
        "US large and mid cap", "technical relative strength",
    ),
    MomentumBenchmark(
        "QMOM", "concentrated momentum stress reference", False,
        "US equities", "concentrated quantitative momentum",
    ),
    MomentumBenchmark(
        "IMTM", "out-of-US external-validity reference", False,
        "developed markets ex-US", "international momentum",
    ),
)


def benchmark_tickers(*, decisive_only: bool = False) -> tuple[str, ...]:
    rows = MOMENTUM_BENCHMARKS
    if decisive_only:
        rows = tuple(row for row in rows if row.decisive)
    return tuple(row.ticker for row in rows)


def assert_actual_only(series_kind_by_ticker: dict[str, str]) -> None:
    """Reject vendor backfills presented as investable ETF history."""
    invalid = {
        ticker: kind
        for ticker, kind in series_kind_by_ticker.items()
        if ticker in benchmark_tickers() and kind != "actual_etf_total_return"
    }
    if invalid:
        raise ValueError(f"ETF benchmarks require actual total returns: {invalid}")
