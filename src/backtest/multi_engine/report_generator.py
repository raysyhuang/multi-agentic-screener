"""Report generator for the multi-engine cross-backtest.

Produces a JSON report with:
  - per_engine: summary, by_regime, by_strategy, equity_curve per engine
  - synthesis: equal-weight and credibility-weight results
  - benchmark: SPY buy-and-hold
  - cross_engine: head-to-head, correlation matrix, convergence analysis, stats
  - trades: full trade log with engine attribution
  - credibility_evolution: rolling weight changes (if enabled)

Reuses :func:`src.backtest.metrics.compute_metrics` and
:func:`src.backtest.metrics.deflated_sharpe_ratio` for per-track metrics,
plus comparison functions from ``src.engines.backtest_comparator``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date

import numpy as np

from src.backtest.metrics import compute_metrics, deflated_sharpe_ratio
from src.backtest.multi_engine.portfolio_sim import (
    DailyPickRecord,
    TrackResult,
    SimulatedTrade,
)

logger = logging.getLogger(__name__)


def generate_report(
    track_results: dict[str, TrackResult],
    daily_records: list[DailyPickRecord],
    start_date: date,
    end_date: date,
    engine_names: list[str],
    config: dict,
    elapsed_s: float,
) -> dict:
    """Build the final JSON report from simulation results."""
    report: dict = {
        "run_date": str(date.today()),
        "date_range": {"start": str(start_date), "end": str(end_date)},
        "trading_days": len(daily_records),
        "engines": engine_names,
        "config": config,
        "elapsed_s": round(elapsed_s, 1),
    }

    # Per-engine summaries
    report["per_engine"] = {}
    for engine in engine_names:
        track = track_results.get(engine)
        if track:
            report["per_engine"][engine] = _build_track_summary(
                track, daily_records, engine
            )

    # Synthesis results
    report["synthesis"] = {
        "equal_weight": _build_track_summary(
            track_results.get("eq_synth", TrackResult("eq_synth", [], [], [])),
            daily_records,
            "eq_synth",
        ),
        "credibility_weight": _build_track_summary(
            track_results.get("cred_synth", TrackResult("cred_synth", [], [], [])),
            daily_records,
            "cred_synth",
        ),
    }

    # Regime-gated synthesis track
    if "regime_gated" in track_results:
        report["synthesis"]["regime_gated"] = _build_track_summary(
            track_results["regime_gated"],
            daily_records,
            "regime_gated",
        )

    # Confidence-sized synthesis track (if present)
    if "sized_synth" in track_results:
        report["synthesis"]["sized_synth"] = _build_track_summary(
            track_results["sized_synth"],
            daily_records,
            "sized_synth",
        )

    # Benchmark
    spy_track = track_results.get("spy_benchmark", TrackResult("spy_benchmark", [], [], []))
    report["benchmark"] = {
        "spy_buy_and_hold": {
            "total_return_pct": round(sum(spy_track.pnl_series), 2),
            "equity_curve": spy_track.equity_curve[-100:],
        },
    }

    # Cross-engine comparison
    report["cross_engine"] = _build_cross_engine(track_results, engine_names)

    # Trade log (capped for file size)
    all_trades: list[dict] = []
    for track in track_results.values():
        for t in track.trades:
            all_trades.append(t.to_dict())
    report["total_trades_all_tracks"] = len(all_trades)
    report["trades"] = all_trades[:5000]

    # Regime distribution
    report["regime_distribution"] = _regime_distribution(daily_records)

    return report


def _build_track_summary(
    track: TrackResult,
    daily_records: list[DailyPickRecord],
    track_name: str,
) -> dict:
    """Compute summary metrics for a single simulation track."""
    pnls = [t.pnl_after_costs for t in track.trades]
    metrics = compute_metrics(pnls)

    # DSR with num_trials=5 (we test 5 strategy variants)
    dsr = 0.0
    if pnls and metrics.sharpe_ratio > 0:
        try:
            dsr = deflated_sharpe_ratio(
                observed_sharpe=metrics.sharpe_ratio,
                num_trials=5,
                returns=pnls,
            )
        except Exception:
            dsr = 0.0

    # By-regime breakdown
    by_regime = _by_regime(track.trades, daily_records)

    # By-strategy breakdown
    by_strategy = _by_strategy(track.trades)

    return {
        "summary": {
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "avg_return_pct": metrics.avg_return_pct,
            "median_return_pct": metrics.median_return_pct,
            "total_return_pct": metrics.total_return_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "profit_factor": metrics.profit_factor,
            "expectancy": metrics.expectancy,
            "payoff_ratio": metrics.payoff_ratio,
            "max_consecutive_wins": metrics.max_consecutive_wins,
            "max_consecutive_losses": metrics.max_consecutive_losses,
            "deflated_sharpe_ratio": dsr,
        },
        "by_regime": by_regime,
        "by_strategy": by_strategy,
        "equity_curve": track.equity_curve[-200:],
    }


def _by_regime(
    trades: list[SimulatedTrade],
    daily_records: list[DailyPickRecord],
) -> dict:
    """Break down trade performance by market regime on signal date."""
    # Build date → regime mapping
    date_regime: dict[date, str] = {}
    for rec in daily_records:
        date_regime[rec.screen_date] = rec.regime

    by_reg: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        regime = date_regime.get(t.signal_date, "unknown")
        by_reg[regime].append(t.pnl_after_costs)

    result = {}
    for regime, pnls in by_reg.items():
        m = compute_metrics(pnls)
        result[regime] = {
            "trades": m.total_trades,
            "win_rate": m.win_rate,
            "avg_return_pct": m.avg_return_pct,
            "sharpe": m.sharpe_ratio,
            "total_return_pct": m.total_return_pct,
        }
    return result


def _by_strategy(trades: list[SimulatedTrade]) -> dict:
    """Break down trade performance by strategy type."""
    by_strat: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by_strat[t.strategy].append(t.pnl_after_costs)

    result = {}
    for strat, pnls in by_strat.items():
        m = compute_metrics(pnls)
        result[strat] = {
            "trades": m.total_trades,
            "win_rate": m.win_rate,
            "avg_return_pct": m.avg_return_pct,
            "sharpe": m.sharpe_ratio,
            "total_return_pct": m.total_return_pct,
        }
    return result


def _build_cross_engine(
    track_results: dict[str, TrackResult],
    engine_names: list[str],
) -> dict:
    """Build cross-engine comparison section."""
    # Head-to-head table
    head_to_head = []
    synth_tracks = ["eq_synth", "cred_synth", "regime_gated"]
    if "sized_synth" in track_results:
        synth_tracks.append("sized_synth")
    for name in engine_names + synth_tracks:
        track = track_results.get(name)
        if not track or not track.trades:
            continue
        pnls = [t.pnl_after_costs for t in track.trades]
        m = compute_metrics(pnls)
        head_to_head.append({
            "engine": name,
            "total_trades": m.total_trades,
            "win_rate": m.win_rate,
            "avg_return_pct": m.avg_return_pct,
            "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio,
            "max_drawdown_pct": m.max_drawdown_pct,
            "profit_factor": m.profit_factor,
            "calmar": m.calmar_ratio,
            "expectancy": m.expectancy,
        })

    # Sort by Sharpe descending
    head_to_head.sort(key=lambda r: r.get("sharpe", 0), reverse=True)

    # Correlation matrix (ticker overlap)
    correlation = _ticker_overlap(track_results, engine_names)

    # Convergence analysis
    convergence = _convergence_analysis(track_results, engine_names)

    # Statistical tests (Welch's t-test between engine pairs)
    stat_tests = _statistical_tests(track_results, engine_names)

    return {
        "head_to_head": head_to_head,
        "correlation_matrix": correlation,
        "convergence_analysis": convergence,
        "statistical_tests": stat_tests,
    }


def _ticker_overlap(
    track_results: dict[str, TrackResult], engine_names: list[str]
) -> dict:
    """Compute ticker overlap between engine pairs."""
    engine_tickers: dict[str, set[str]] = {}
    for name in engine_names:
        track = track_results.get(name)
        if track:
            engine_tickers[name] = {t.ticker for t in track.trades}
        else:
            engine_tickers[name] = set()

    matrix = {}
    names = list(engine_tickers.keys())
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            set_a = engine_tickers[a]
            set_b = engine_tickers[b]
            union = set_a | set_b
            inter = set_a & set_b
            overlap = len(inter) / len(union) * 100 if union else 0
            matrix[f"{a} x {b}"] = {
                "overlap_pct": round(overlap, 1),
                "shared_tickers": len(inter),
                "total_unique": len(union),
            }
    return matrix


def _convergence_analysis(
    track_results: dict[str, TrackResult], engine_names: list[str]
) -> dict:
    """Analyze how convergence (multi-engine agreement) affects returns."""
    synth_track = track_results.get("eq_synth")
    if not synth_track or not synth_track.trades:
        return {"note": "No synthesis trades for convergence analysis"}

    # Group synthesis trades by engine_count (embedded in engine_name as comma-sep)
    by_count: dict[int, list[float]] = defaultdict(list)
    for t in synth_track.trades:
        count = len(t.engine_name.split(","))
        by_count[count].append(t.pnl_after_costs)

    result = {}
    for count, pnls in sorted(by_count.items()):
        m = compute_metrics(pnls)
        result[f"{count}_engines"] = {
            "trades": m.total_trades,
            "win_rate": m.win_rate,
            "avg_return_pct": m.avg_return_pct,
            "sharpe": m.sharpe_ratio,
        }
    return result


def _statistical_tests(
    track_results: dict[str, TrackResult], engine_names: list[str]
) -> list[dict]:
    """Welch's t-test between engine pairs."""
    try:
        from scipy import stats
    except ImportError:
        return [{"note": "scipy not available for statistical tests"}]

    results = []
    all_names = engine_names + ["eq_synth", "cred_synth", "regime_gated"]

    for i, a in enumerate(all_names):
        for j, b in enumerate(all_names):
            if i >= j:
                continue
            track_a = track_results.get(a)
            track_b = track_results.get(b)
            if not track_a or not track_b:
                continue

            pnls_a = [t.pnl_after_costs for t in track_a.trades]
            pnls_b = [t.pnl_after_costs for t in track_b.trades]

            if len(pnls_a) < 10 or len(pnls_b) < 10:
                results.append({
                    "pair": f"{a} vs {b}",
                    "note": "Insufficient trades",
                    "n_a": len(pnls_a),
                    "n_b": len(pnls_b),
                })
                continue

            t_stat, p_value = stats.ttest_ind(pnls_a, pnls_b, equal_var=False)
            mean_a = float(np.mean(pnls_a))
            mean_b = float(np.mean(pnls_b))

            results.append({
                "pair": f"{a} vs {b}",
                "mean_a": round(mean_a, 4),
                "mean_b": round(mean_b, 4),
                "difference": round(mean_a - mean_b, 4),
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_value), 4),
                "significant_at_05": bool(p_value < 0.05),
                "better_engine": a if mean_a > mean_b else b,
                "n_a": len(pnls_a),
                "n_b": len(pnls_b),
            })

    return results


def _regime_distribution(daily_records: list[DailyPickRecord]) -> dict:
    """Count trading days per regime."""
    counts: dict[str, int] = defaultdict(int)
    for rec in daily_records:
        counts[rec.regime] += 1
    total = len(daily_records)
    return {
        regime: {"days": count, "pct": round(count / total * 100, 1) if total else 0}
        for regime, count in sorted(counts.items())
    }
