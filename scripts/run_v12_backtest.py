#!/usr/bin/env python
"""Standardized V1.2 backtest runner.

Runs the mean reversion model with V1.2 parameters across 1Y/3Y/5Y horizons
and reports metrics against frozen baselines. Use this for all future backtests
to ensure consistent methodology.

Usage:
    python scripts/run_v12_backtest.py                    # default: 1Y/3Y/5Y
    python scripts/run_v12_backtest.py --years 2          # single horizon
    python scripts/run_v12_backtest.py --compare-baseline # compare vs frozen baseline
    python scripts/run_v12_backtest.py --two-leg          # enable two-leg exits
    python scripts/run_v12_backtest.py --full-v12         # all V1.2 features enabled
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research.signal_backtest import (
    fetch_ohlcv,
    get_sp500_tickers,
    run_model_backtest,
    format_model_report,
    save_trades_csv,
)

# Frozen production parameters (current baseline)
BASELINE_PARAMS = {
    "rsi2_threshold": 10,
    "stop_atr_mult": 0.75,
    "target_atr_mult": 1.5,
    "holding_period": 3,
    "min_score": 40,
    "trail_activate_pct": 0.5,
    "trail_distance_pct": 0.3,
}

# V1.2 parameters: baseline + entry filters (gap, volume slope, ATR floor, choppy gate)
# Partial TP disabled — backtest proved trailing stop alone captures MFE better.
# Entry filters are baked into score_mean_reversion(); no extra params needed here.
V12_PARAMS = {
    **BASELINE_PARAMS,
    # partial_tp_atr_mult intentionally omitted (disabled in production)
}

# Acceptance gates (from plan)
ACCEPTANCE_GATES = {
    "win_rate_delta_min": 0.03,         # >= baseline + 3pp
    "expectancy_min_ratio": 1.0,        # >= baseline
    "profit_factor_tolerance": 0.02,    # >= baseline - 0.02
    "max_drawdown_ratio": 1.10,         # <= baseline × 1.10
    "trade_count_ratio_min": 0.70,      # >= 70% of baseline
}

HORIZONS = [1.0, 3.0, 5.0]
HORIZON_LABELS = {1.0: "1Y", 3.0: "3Y", 5.0: "5Y"}


def _load_baseline(years: float) -> dict | None:
    """Load frozen baseline for a given horizon."""
    label = HORIZON_LABELS.get(years, f"{years}Y")
    path = Path(f"outputs/research/baseline_{label}_2026-04-14.json")
    if path.exists():
        return json.loads(path.read_text())
    return None


def _check_gates(baseline_metrics: dict, test_metrics: dict) -> dict:
    """Check acceptance gates and return pass/fail for each."""
    gates = {}
    bm = baseline_metrics
    tm = test_metrics

    # Win rate >= baseline + 3pp
    wr_delta = tm["win_rate"] - bm["win_rate"]
    gates["win_rate"] = {
        "pass": wr_delta >= ACCEPTANCE_GATES["win_rate_delta_min"],
        "baseline": bm["win_rate"],
        "test": tm["win_rate"],
        "delta": round(wr_delta, 4),
        "gate": f"+{ACCEPTANCE_GATES['win_rate_delta_min']:.0%}",
    }

    # Expectancy >= baseline
    gates["expectancy"] = {
        "pass": tm["expectancy"] >= bm["expectancy"] * ACCEPTANCE_GATES["expectancy_min_ratio"],
        "baseline": bm["expectancy"],
        "test": tm["expectancy"],
    }

    # Profit factor >= baseline - 0.02
    gates["profit_factor"] = {
        "pass": tm["profit_factor"] >= bm["profit_factor"] - ACCEPTANCE_GATES["profit_factor_tolerance"],
        "baseline": bm["profit_factor"],
        "test": tm["profit_factor"],
    }

    # Max drawdown <= baseline × 1.10
    gates["max_drawdown"] = {
        "pass": tm["max_drawdown_pct"] <= bm["max_drawdown_pct"] * ACCEPTANCE_GATES["max_drawdown_ratio"],
        "baseline": bm["max_drawdown_pct"],
        "test": tm["max_drawdown_pct"],
    }

    # Trade count >= 70% of baseline
    gates["trade_count"] = {
        "pass": tm["total_trades"] >= bm["total_trades"] * ACCEPTANCE_GATES["trade_count_ratio_min"],
        "baseline": bm["total_trades"],
        "test": tm["total_trades"],
    }

    return gates


def run_backtest(
    years: float,
    params: dict,
    label: str,
    tickers: list[str] | None = None,
    no_cache: bool = False,
) -> dict:
    """Run a single backtest and return serializable results."""
    if tickers is None:
        tickers = get_sp500_tickers()

    price_data = fetch_ohlcv(tickers, years=years, no_cache=no_cache)
    if not price_data:
        print("No data fetched!")
        sys.exit(1)

    result = run_model_backtest("mean_reversion", price_data, params)
    print(format_model_report(result))

    # Compute expiry MFE>2% rate
    expiry_trades = [t for t in result.trades if t.exit_reason == "expiry"]
    expiry_mfe_gt2 = sum(1 for t in expiry_trades if t.mfe_pct > 2.0)
    expiry_mfe_rate = expiry_mfe_gt2 / len(expiry_trades) if expiry_trades else 0.0

    out = {
        "label": label,
        "years": years,
        "params": params,
        "run_date": str(date.today()),
        "metrics": asdict(result.metrics),
        "by_regime": {k: asdict(v) for k, v in result.by_regime.items()},
        "by_exit_reason": result.by_exit_reason,
        "avg_mfe_pct": result.avg_mfe_pct,
        "avg_mae_pct": result.avg_mae_pct,
        "expiry_mfe_gt_2pct": round(expiry_mfe_rate, 4),
        "total_trades": result.metrics.total_trades,
    }

    # Save trades CSV
    out_dir = Path("outputs/research")
    out_dir.mkdir(parents=True, exist_ok=True)
    horizon = HORIZON_LABELS.get(years, f"{years}Y")
    save_trades_csv(result.trades, out_dir / f"backtest_{label}_{horizon}_trades.csv")

    return out


def main():
    parser = argparse.ArgumentParser(description="Standardized V1.2 backtest")
    parser.add_argument("--years", type=float, help="Single horizon (e.g. 2)")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Compare against frozen baseline and check gates")
    parser.add_argument("--two-leg", action="store_true",
                        help="Enable two-leg exits (partial TP at 1.0x ATR)")
    parser.add_argument("--full-v12", action="store_true",
                        help="All V1.2 features enabled")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force fresh data download")
    args = parser.parse_args()

    # Select params
    if args.full_v12 or args.two_leg:
        params = V12_PARAMS.copy()
        label = "v12_two_leg"
    else:
        params = BASELINE_PARAMS.copy()
        label = "baseline"

    # Select horizons
    horizons = [args.years] if args.years else HORIZONS

    tickers = get_sp500_tickers()
    all_results = {}

    for years in horizons:
        horizon_label = HORIZON_LABELS.get(years, f"{years}Y")
        print(f"\n{'='*60}")
        print(f"  Running {label} @ {horizon_label}")
        print(f"{'='*60}")

        result = run_backtest(years, params, label, tickers, args.no_cache)
        all_results[horizon_label] = result

        # Save result
        out_path = Path(f"outputs/research/{label}_{horizon_label}_{date.today()}.json")
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {out_path}")

        # Compare against baseline
        if args.compare_baseline:
            baseline = _load_baseline(years)
            if baseline:
                gates = _check_gates(baseline["metrics"], result["metrics"])
                print(f"\n  Acceptance Gates ({horizon_label}):")
                all_pass = True
                for gate_name, g in gates.items():
                    status = "PASS" if g["pass"] else "FAIL"
                    if not g["pass"]:
                        all_pass = False
                    print(f"    {gate_name:20s}: {status}  (baseline={g['baseline']}, test={g['test']})")

                if all_pass:
                    print(f"\n  ALL GATES PASSED for {horizon_label}")
                else:
                    print(f"\n  SOME GATES FAILED for {horizon_label}")

                # Expiry MFE comparison
                baseline_mfe = baseline.get("expiry_mfe_gt_2pct", "N/A")
                test_mfe = result.get("expiry_mfe_gt_2pct", "N/A")
                print(f"    Expiry MFE>2%: baseline={baseline_mfe}, test={test_mfe}")
            else:
                print(f"\n  No frozen baseline found for {horizon_label}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {label}")
    print(f"{'='*60}")
    print(f"  {'Horizon':8s} {'Trades':>7s} {'WR':>7s} {'Avg':>8s} {'Sharpe':>7s} {'PF':>6s} {'MaxDD':>7s}")
    for h, r in all_results.items():
        m = r["metrics"]
        print(f"  {h:8s} {m['total_trades']:7d} {m['win_rate']:7.1%} {m['avg_return_pct']:+7.2f}% "
              f"{m['sharpe_ratio']:7.2f} {m['profit_factor']:6.2f} {m['max_drawdown_pct']:6.2f}%")


if __name__ == "__main__":
    main()
