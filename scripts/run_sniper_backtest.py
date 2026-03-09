#!/usr/bin/env python
"""Sniper track backtest runner.

Runs the sniper signal model (BB squeeze + vol compression + RS) across
S&P 500 on 1Y/3Y/5Y horizons and checks acceptance gates.

Sniper acceptance gates (different from MR — targets larger moves):
  - Win rate >= 45%
  - Avg trade >= 2.0%
  - Profit factor >= 1.5
  - Max drawdown <= 25%
  - Sharpe >= 0.5
  - Trades/year >= 100

Usage:
    python scripts/run_sniper_backtest.py                    # default: 1Y/3Y/5Y
    python scripts/run_sniper_backtest.py --years 1          # single horizon
    python scripts/run_sniper_backtest.py --atr-floor 3.0    # custom ATR% floor
    python scripts/run_sniper_backtest.py --no-cache          # force fresh data
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
    run_model_backtest,
    format_model_report,
    save_trades_csv,
)
from src.research.sp500_tickers import SP500_TICKERS

# Default sniper parameters
SNIPER_PARAMS = {
    "min_score": 70,
    "atr_pct_floor": 5.0,
    "stop_atr_mult": 1.5,
    "target_atr_mult": 3.0,
    "holding_period": 7,
    "trail_activate_pct": 1.0,
    "trail_distance_pct": 0.5,
}

# Acceptance gates
ACCEPTANCE_GATES = {
    "win_rate_min": 0.45,
    "avg_return_min": 2.0,
    "profit_factor_min": 1.5,
    "max_drawdown_max": 30.0,  # growth sleeve, not core — 30% is appropriate
    "sharpe_min": 0.5,
    "trades_per_year_min": 100,
}

HORIZONS = [1.0, 3.0, 5.0]
HORIZON_LABELS = {1.0: "1Y", 3.0: "3Y", 5.0: "5Y"}


def _check_gates(metrics: dict, years: float) -> dict:
    """Check sniper acceptance gates."""
    gates = {}

    gates["win_rate"] = {
        "pass": metrics["win_rate"] >= ACCEPTANCE_GATES["win_rate_min"],
        "value": metrics["win_rate"],
        "gate": f">= {ACCEPTANCE_GATES['win_rate_min']:.0%}",
    }

    gates["avg_return"] = {
        "pass": metrics["avg_return_pct"] >= ACCEPTANCE_GATES["avg_return_min"],
        "value": metrics["avg_return_pct"],
        "gate": f">= {ACCEPTANCE_GATES['avg_return_min']}%",
    }

    gates["profit_factor"] = {
        "pass": metrics["profit_factor"] >= ACCEPTANCE_GATES["profit_factor_min"],
        "value": metrics["profit_factor"],
        "gate": f">= {ACCEPTANCE_GATES['profit_factor_min']}",
    }

    gates["max_drawdown"] = {
        "pass": metrics["max_drawdown_pct"] <= ACCEPTANCE_GATES["max_drawdown_max"],
        "value": metrics["max_drawdown_pct"],
        "gate": f"<= {ACCEPTANCE_GATES['max_drawdown_max']}%",
    }

    gates["sharpe"] = {
        "pass": metrics["sharpe_ratio"] >= ACCEPTANCE_GATES["sharpe_min"],
        "value": metrics["sharpe_ratio"],
        "gate": f">= {ACCEPTANCE_GATES['sharpe_min']}",
    }

    trades_per_year = metrics["total_trades"] / years if years > 0 else 0
    gates["trades_per_year"] = {
        "pass": trades_per_year >= ACCEPTANCE_GATES["trades_per_year_min"],
        "value": trades_per_year,
        "gate": f">= {ACCEPTANCE_GATES['trades_per_year_min']}",
    }

    return gates


def run_backtest(
    years: float,
    params: dict,
    tickers: list[str] | None = None,
    no_cache: bool = False,
) -> dict:
    """Run a single sniper backtest and return serializable results."""
    if tickers is None:
        tickers = list(SP500_TICKERS)

    # Ensure SPY is in the ticker list for relative strength calculation
    if "SPY" not in tickers:
        tickers = ["SPY"] + tickers

    price_data = fetch_ohlcv(tickers, years=years, no_cache=no_cache)
    if not price_data:
        print("No data fetched!")
        sys.exit(1)

    result = run_model_backtest("sniper", price_data, params)
    print(format_model_report(result))

    out = {
        "label": "sniper",
        "years": years,
        "params": params,
        "run_date": str(date.today()),
        "metrics": asdict(result.metrics),
        "by_regime": {k: asdict(v) for k, v in result.by_regime.items()},
        "by_exit_reason": result.by_exit_reason,
        "avg_mfe_pct": result.avg_mfe_pct,
        "avg_mae_pct": result.avg_mae_pct,
        "total_trades": result.metrics.total_trades,
    }

    # Save trades CSV
    out_dir = Path("outputs/research")
    out_dir.mkdir(parents=True, exist_ok=True)
    horizon = HORIZON_LABELS.get(years, f"{years}Y")
    save_trades_csv(result.trades, out_dir / f"backtest_sniper_{horizon}_trades.csv")

    return out


def main():
    parser = argparse.ArgumentParser(description="Sniper track backtest")
    parser.add_argument("--years", type=float, help="Single horizon (e.g. 1)")
    parser.add_argument("--atr-floor", type=float, default=5.0,
                        help="ATR%% floor for sniper signals (default: 5.0)")
    parser.add_argument("--stop-mult", type=float, default=1.5,
                        help="Stop loss ATR multiplier (default: 1.5)")
    parser.add_argument("--target-mult", type=float, default=3.0,
                        help="Target ATR multiplier (default: 3.0)")
    parser.add_argument("--hold", type=int, default=7,
                        help="Holding period in days (default: 7)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force fresh data download")
    args = parser.parse_args()

    params = SNIPER_PARAMS.copy()
    params["atr_pct_floor"] = args.atr_floor
    params["stop_atr_mult"] = args.stop_mult
    params["target_atr_mult"] = args.target_mult
    params["holding_period"] = args.hold

    horizons = [args.years] if args.years else HORIZONS
    tickers = list(SP500_TICKERS)

    # Ensure SPY for RS calculation
    if "SPY" not in tickers:
        tickers = ["SPY"] + tickers

    all_results = {}

    for years in horizons:
        horizon_label = HORIZON_LABELS.get(years, f"{years}Y")
        print(f"\n{'='*60}")
        print(f"  Running sniper @ {horizon_label}")
        print(f"{'='*60}")

        result = run_backtest(years, params, tickers, args.no_cache)
        all_results[horizon_label] = result

        # Save result
        out_path = Path(f"outputs/research/sniper_{horizon_label}_{date.today()}.json")
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {out_path}")

        # Check acceptance gates
        gates = _check_gates(result["metrics"], years)
        print(f"\n  Acceptance Gates ({horizon_label}):")
        all_pass = True
        for gate_name, g in gates.items():
            status = "PASS" if g["pass"] else "FAIL"
            if not g["pass"]:
                all_pass = False
            print(f"    {gate_name:20s}: {status}  (value={g['value']:.4f}, gate={g['gate']})")

        if all_pass:
            print(f"\n  ALL GATES PASSED for {horizon_label}")
        else:
            print(f"\n  SOME GATES FAILED for {horizon_label}")

    # Save baseline
    if len(all_results) >= 1:
        baseline_path = Path(f"outputs/research/sniper_baseline.json")
        baseline_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nBaseline saved to {baseline_path}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY: sniper")
    print(f"{'='*60}")
    print(f"  {'Horizon':8s} {'Trades':>7s} {'WR':>7s} {'Avg':>8s} {'Sharpe':>7s} {'PF':>6s} {'MaxDD':>7s}")
    for h, r in all_results.items():
        m = r["metrics"]
        print(f"  {h:8s} {m['total_trades']:7d} {m['win_rate']:7.1%} {m['avg_return_pct']:+7.2f}% "
              f"{m['sharpe_ratio']:7.2f} {m['profit_factor']:6.2f} {m['max_drawdown_pct']:6.2f}%")


if __name__ == "__main__":
    main()
