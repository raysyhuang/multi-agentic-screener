"""Cross-engine backtest comparator.

Reads standardized JSON output from all four engines and produces:
1. Head-to-head table: Win rate, Sharpe, max DD, profit factor per engine
2. Regime breakdown: Which engine performs best in Bull/Bear/Mixed
3. Correlation matrix: Overlap % of picks between engine pairs
4. Blend simulation: Equal-weight and credibility-weight portfolios
5. Statistical tests: Paired t-test on per-trade returns
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Engine names and default report directories
ENGINE_CONFIGS = {
    "multi-agentic-screener": {
        "search_dirs": [Path("backtest_results")],
        "pattern": "mas_replay_*.json",
    },
    "koocore-d": {
        "search_dirs": [
            Path("../KooCore-D/backtest_results"),
            Path("/Users/rayhuang/Documents/Python Project/KooCore-D/backtest_results"),
        ],
        "pattern": "koocore_backtest_*.json",
    },
    "gemini-stst": {
        "search_dirs": [
            Path("../Gemini STST/backtest_results"),
            Path("/Users/rayhuang/Documents/Python Project/Gemini STST/backtest_results"),
        ],
        "pattern": "gemini_backtest_*.json",
    },
    "top3-7d": {
        "search_dirs": [
            Path("../Top3-7D Engine/backtest_results"),
            Path("/Users/rayhuang/Documents/Python Project/Top3-7D Engine/backtest_results"),
        ],
        "pattern": "top3_backtest_*.json",
    },
}


def load_engine_report(engine: str) -> dict | None:
    """Load the most recent backtest report for an engine."""
    config = ENGINE_CONFIGS.get(engine)
    if not config:
        return None

    for search_dir in config["search_dirs"]:
        if not search_dir.exists():
            continue
        files = sorted(search_dir.glob(config["pattern"]), reverse=True)
        if files:
            try:
                report = json.loads(files[0].read_text())
                logger.info("Loaded %s report from %s", engine, files[0])
                return report
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read %s: %s", files[0], e)

    logger.info("No backtest report found for %s", engine)
    return None


def load_all_reports(report_paths: dict[str, str] | None = None) -> dict[str, dict]:
    """Load backtest reports from all engines.

    Args:
        report_paths: Optional explicit paths {engine_name: file_path}.
                     If None, auto-discovers from default directories.
    """
    reports = {}

    if report_paths:
        for engine, path in report_paths.items():
            try:
                reports[engine] = json.loads(Path(path).read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load %s from %s: %s", engine, path, e)
    else:
        for engine in ENGINE_CONFIGS:
            report = load_engine_report(engine)
            if report:
                reports[engine] = report

    return reports


def compare_engines(reports: dict[str, dict]) -> dict[str, Any]:
    """Run full cross-engine comparison.

    Returns a dict with:
    - head_to_head: Summary table comparing key metrics
    - regime_breakdown: Per-regime performance by engine
    - correlation_matrix: Ticker overlap between engine pairs
    - blend_simulation: Combined portfolio simulations
    - statistical_tests: Pairwise significance tests
    """
    if not reports:
        return {"error": "No engine reports loaded", "engines": []}

    result = {
        "run_date": str(date.today()),
        "engines": list(reports.keys()),
        "engine_count": len(reports),
        "head_to_head": _head_to_head(reports),
        "regime_breakdown": _regime_breakdown(reports),
        "correlation_matrix": _correlation_matrix(reports),
        "blend_simulation": _blend_simulation(reports),
        "statistical_tests": _statistical_tests(reports),
    }

    return result


def _head_to_head(reports: dict[str, dict]) -> list[dict]:
    """Build head-to-head comparison table."""
    rows = []
    for engine, report in reports.items():
        summary = report.get("summary", {})
        rows.append({
            "engine": engine,
            "total_trades": summary.get("total_trades", 0),
            "win_rate": summary.get("win_rate", 0),
            "avg_return_pct": summary.get("avg_return_pct", 0),
            "sharpe": summary.get("sharpe", summary.get("sharpe_ratio", 0)),
            "sortino": summary.get("sortino", summary.get("sortino_ratio", 0)),
            "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
            "profit_factor": summary.get("profit_factor", 0),
            "expectancy_pct": summary.get("expectancy_pct", summary.get("expectancy", 0)),
            "calmar": summary.get("calmar", summary.get("calmar_ratio", 0)),
            "avg_hold_days": summary.get("avg_hold_days", summary.get("avg_holding_days", 0)),
            "date_range": report.get("date_range", {}),
        })

    # Sort by Sharpe descending
    rows.sort(key=lambda r: r.get("sharpe", 0), reverse=True)
    return rows


def _regime_breakdown(reports: dict[str, dict]) -> dict[str, list[dict]]:
    """Extract regime-level performance for each engine."""
    regimes = {}  # {regime: [{engine, trades, win_rate, sharpe}]}

    for engine, report in reports.items():
        by_regime = report.get("by_regime", {})
        for regime, stats in by_regime.items():
            regimes.setdefault(regime, []).append({
                "engine": engine,
                "trades": stats.get("trades", 0),
                "win_rate": stats.get("win_rate", 0),
                "sharpe": stats.get("sharpe", 0),
            })

    # Sort each regime by Sharpe
    for regime in regimes:
        regimes[regime].sort(key=lambda r: r.get("sharpe", 0), reverse=True)

    return regimes


def _correlation_matrix(reports: dict[str, dict]) -> dict:
    """Compute ticker overlap between engine pairs.

    Lower correlation = better diversification potential.
    """
    # Extract unique tickers per engine
    engine_tickers: dict[str, set[str]] = {}
    for engine, report in reports.items():
        trades = report.get("trades", [])
        tickers = {t.get("ticker", "") for t in trades if t.get("ticker")}
        engine_tickers[engine] = tickers

    engines = list(engine_tickers.keys())
    matrix = {}

    for i, eng_a in enumerate(engines):
        for j, eng_b in enumerate(engines):
            if i >= j:
                continue

            set_a = engine_tickers[eng_a]
            set_b = engine_tickers[eng_b]
            union = set_a | set_b
            intersection = set_a & set_b

            overlap_pct = len(intersection) / len(union) * 100 if union else 0

            pair_key = f"{eng_a} x {eng_b}"
            matrix[pair_key] = {
                "overlap_pct": round(overlap_pct, 1),
                "shared_tickers": sorted(intersection),
                "total_unique": len(union),
                "engine_a_only": len(set_a - set_b),
                "engine_b_only": len(set_b - set_a),
            }

    return matrix


def _blend_simulation(reports: dict[str, dict]) -> dict:
    """Simulate equal-weight and credibility-weight blended portfolios."""
    if len(reports) < 2:
        return {"note": "Need at least 2 engines for blend simulation"}

    # Collect all trade PnLs by date
    all_trades_by_date: dict[str, list[float]] = {}
    engine_pnls: dict[str, list[float]] = {}

    for engine, report in reports.items():
        trades = report.get("trades", [])
        pnls = []
        for t in trades:
            pnl = t.get("pnl_pct", 0) or 0
            exit_date = t.get("exit_date", t.get("signal_date", ""))
            if exit_date:
                all_trades_by_date.setdefault(str(exit_date), []).append(pnl)
            pnls.append(pnl)
        engine_pnls[engine] = pnls

    # Equal-weight blend: average PnL across engines per date
    sorted_dates = sorted(all_trades_by_date.keys())
    eq_curve = []
    eq_cumulative = 0.0
    eq_returns = []

    for d in sorted_dates:
        avg_pnl = np.mean(all_trades_by_date[d])
        eq_returns.append(avg_pnl)
        eq_cumulative += avg_pnl
        eq_curve.append({
            "date": d,
            "cumulative_pnl_pct": round(eq_cumulative, 2),
        })

    # Equal-weight metrics
    eq_arr = np.array(eq_returns) if eq_returns else np.array([0])
    eq_wins = eq_arr[eq_arr > 0]
    eq_std = float(np.std(eq_arr, ddof=1)) if len(eq_arr) > 1 else 1.0

    # Credibility-weight blend: weight by Sharpe ratio
    sharpes = {}
    for engine, report in reports.items():
        s = report.get("summary", {})
        sharpes[engine] = max(s.get("sharpe", s.get("sharpe_ratio", 0)), 0.01)

    total_sharpe = sum(sharpes.values())
    cred_weights = {e: s / total_sharpe for e, s in sharpes.items()}

    # Weighted PnL
    cred_returns = []
    for engine, pnls in engine_pnls.items():
        w = cred_weights.get(engine, 0)
        for pnl in pnls:
            cred_returns.append(pnl * w)

    cred_arr = np.array(cred_returns) if cred_returns else np.array([0])
    cred_wins = cred_arr[cred_arr > 0]
    cred_std = float(np.std(cred_arr, ddof=1)) if len(cred_arr) > 1 else 1.0

    return {
        "equal_weight": {
            "total_trades": len(eq_returns),
            "win_rate": round(len(eq_wins) / len(eq_arr), 4) if len(eq_arr) > 0 else 0,
            "avg_return_pct": round(float(np.mean(eq_arr)), 2),
            "sharpe": round(float(np.mean(eq_arr)) / eq_std * np.sqrt(50), 2) if eq_std > 0 else 0,
            "total_return_pct": round(float(np.sum(eq_arr)), 2),
            "equity_curve": eq_curve[-50:],  # Last 50 points
        },
        "credibility_weight": {
            "weights": {e: round(w, 4) for e, w in cred_weights.items()},
            "total_trades": len(cred_returns),
            "win_rate": round(len(cred_wins) / len(cred_arr), 4) if len(cred_arr) > 0 else 0,
            "avg_return_pct": round(float(np.mean(cred_arr)), 2),
            "sharpe": round(float(np.mean(cred_arr)) / cred_std * np.sqrt(50), 2) if cred_std > 0 else 0,
            "total_return_pct": round(float(np.sum(cred_arr)), 2),
        },
    }


def _statistical_tests(reports: dict[str, dict]) -> list[dict]:
    """Paired t-test on per-trade returns between engine pairs.

    Tests: Is Engine A significantly better than Engine B?
    """
    from scipy import stats

    engines = list(reports.keys())
    results = []

    for i, eng_a in enumerate(engines):
        for j, eng_b in enumerate(engines):
            if i >= j:
                continue

            pnls_a = [
                t.get("pnl_pct", 0) or 0
                for t in reports[eng_a].get("trades", [])
            ]
            pnls_b = [
                t.get("pnl_pct", 0) or 0
                for t in reports[eng_b].get("trades", [])
            ]

            if len(pnls_a) < 5 or len(pnls_b) < 5:
                results.append({
                    "pair": f"{eng_a} vs {eng_b}",
                    "note": "Insufficient trades for statistical test",
                    "n_a": len(pnls_a),
                    "n_b": len(pnls_b),
                })
                continue

            # Independent samples t-test (Welch's)
            t_stat, p_value = stats.ttest_ind(pnls_a, pnls_b, equal_var=False)

            mean_a = float(np.mean(pnls_a))
            mean_b = float(np.mean(pnls_b))

            results.append({
                "pair": f"{eng_a} vs {eng_b}",
                "mean_a": round(mean_a, 4),
                "mean_b": round(mean_b, 4),
                "difference": round(mean_a - mean_b, 4),
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_value), 4),
                "significant_at_05": p_value < 0.05,
                "significant_at_01": p_value < 0.01,
                "better_engine": eng_a if mean_a > mean_b else eng_b,
                "n_a": len(pnls_a),
                "n_b": len(pnls_b),
            })

    return results


def run_comparison(report_paths: dict[str, str] | None = None) -> dict:
    """Main entry point: load reports and run comparison."""
    reports = load_all_reports(report_paths)

    if not reports:
        logger.warning("No engine reports found — run individual backtests first")
        return {
            "error": "No backtest reports found",
            "hint": "Run each engine's backtest runner first to generate reports",
        }

    logger.info("Loaded reports for %d engines: %s", len(reports), list(reports.keys()))
    return compare_engines(reports)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Accept optional explicit paths as JSON arg
    if len(sys.argv) > 1:
        try:
            paths = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            paths = None
    else:
        paths = None

    result = run_comparison(paths)
    print(json.dumps(result, indent=2, default=str))
