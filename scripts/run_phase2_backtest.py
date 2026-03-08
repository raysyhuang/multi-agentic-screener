#!/usr/bin/env python
"""Phase 2 Win-Rate Lift backtest runner.

Runs A/B matrix comparing V1.2 champion against Phase 2 candidates:
  1. filters_only (champion baseline)
  2. +confirm (close > open confirmation proxy)
  3. +weekly_gate (close > 150d SMA)
  4. +shock_switch (true range > 3×ATR kill-switch)
  5. +confirm + weekly_gate
  6. +confirm + weekly_gate + shock_switch
  7. +confirm + weekly_gate + shock_switch + weekday filter

Usage:
    python scripts/run_phase2_backtest.py                    # default: 1Y
    python scripts/run_phase2_backtest.py --years 3          # 3Y horizon
    python scripts/run_phase2_backtest.py --years 1 --years 3  # multi-horizon
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
    save_trades_csv,
)

# V1.2 champion params (filters-only, no partial TP)
CHAMPION_PARAMS = {
    "rsi2_threshold": 10,
    "stop_atr_mult": 0.75,
    "target_atr_mult": 1.5,
    "holding_period": 3,
    "min_score": 40,
    "trail_activate_pct": 0.5,
    "trail_distance_pct": 0.3,
}

# Phase 2 candidate configs — each adds one or more features on top of champion
CANDIDATES = {
    "champion": {
        **CHAMPION_PARAMS,
    },
    "confirm_only": {
        **CHAMPION_PARAMS,
        "confirm_entry": True,
        "confirm_mode": "close_gt_open",
    },
    "weekly_gate_only": {
        **CHAMPION_PARAMS,
        # weekly_trend_gate is applied in score_mean_reversion via settings
        # We pass a marker so the runner enables it
        "_weekly_gate": True,
    },
    "shock_only": {
        **CHAMPION_PARAMS,
        "_shock_switch": True,
    },
    "confirm_open_entry": {
        **CHAMPION_PARAMS,
        "confirm_entry": True,
        "confirm_mode": "close_gt_open_open_entry",
    },
    "confirm_weekly": {
        **CHAMPION_PARAMS,
        "confirm_entry": True,
        "confirm_mode": "close_gt_open",
        "_weekly_gate": True,
    },
    "confirm_weekly_shock": {
        **CHAMPION_PARAMS,
        "confirm_entry": True,
        "confirm_mode": "close_gt_open",
        "_weekly_gate": True,
        "_shock_switch": True,
    },
    "full_phase2": {
        **CHAMPION_PARAMS,
        "confirm_entry": True,
        "confirm_mode": "close_gt_open",
        "_weekly_gate": True,
        "_shock_switch": True,
        "blocked_weekdays": {4},  # block Friday signals
    },
    # ── Phase 2B: Adaptive exits + score-tiered stops ──
    # Early exit WITH trailing stop (redundant — trail captures first)
    "early_exit_2.0": {
        **CHAMPION_PARAMS,
        "early_exit_mfe_pct": 2.0,
    },
    # Early exit INSTEAD of trailing stop — simpler exit mechanism
    "early_exit_1.5_no_trail": {
        **CHAMPION_PARAMS,
        "trail_activate_pct": 0.0,
        "trail_distance_pct": 0.0,
        "early_exit_mfe_pct": 1.5,
    },
    "early_exit_2.0_no_trail": {
        **CHAMPION_PARAMS,
        "trail_activate_pct": 0.0,
        "trail_distance_pct": 0.0,
        "early_exit_mfe_pct": 2.0,
    },
    "early_exit_1.0_no_trail": {
        **CHAMPION_PARAMS,
        "trail_activate_pct": 0.0,
        "trail_distance_pct": 0.0,
        "early_exit_mfe_pct": 1.0,
    },
    "score_stops": {
        **CHAMPION_PARAMS,
        # (min_score, stop_atr_mult): high-score → wider stop, low-score → tighter
        "score_stop_tiers": [(80, 1.0), (65, 0.75), (40, 0.5)],
    },
    "score_stops_wide": {
        **CHAMPION_PARAMS,
        "score_stop_tiers": [(80, 1.25), (65, 0.75), (40, 0.5)],
    },
    "early2_score_stops": {
        **CHAMPION_PARAMS,
        "early_exit_mfe_pct": 2.0,
        "score_stop_tiers": [(80, 1.0), (65, 0.75), (40, 0.5)],
    },
    "early1.5_score_stops": {
        **CHAMPION_PARAMS,
        "early_exit_mfe_pct": 1.5,
        "score_stop_tiers": [(80, 1.0), (65, 0.75), (40, 0.5)],
    },
    # Sweep tier boundaries
    "score_stops_v3": {
        **CHAMPION_PARAMS,
        "score_stop_tiers": [(85, 1.25), (70, 0.85), (40, 0.50)],
    },
    "score_stops_v4": {
        **CHAMPION_PARAMS,
        "score_stop_tiers": [(75, 1.0), (60, 0.75), (40, 0.50)],
    },
    "score_stops_v5": {
        **CHAMPION_PARAMS,
        "score_stop_tiers": [(80, 1.25), (60, 0.75), (40, 0.40)],
    },
}

# Acceptance gates (Phase 2 — relaxed WR gate to +2pp)
ACCEPTANCE_GATES = {
    "win_rate_delta_min": 0.02,
    "expectancy_min_ratio": 1.0,
    "profit_factor_tolerance": 0.02,
    "max_drawdown_ratio": 1.10,
    "trade_count_ratio_min": 0.65,
}


def _apply_settings_overrides(params: dict) -> tuple[dict, callable]:
    """Apply settings overrides for features that live in config, not params.

    Returns clean params dict and a cleanup function to restore settings.
    """
    from src.config import get_settings
    settings = get_settings()

    # Save originals
    orig_weekly = settings.weekly_trend_gate_enabled
    orig_shock = settings.shock_killswitch_enabled

    # Apply overrides
    if params.pop("_weekly_gate", False):
        settings.weekly_trend_gate_enabled = True
    else:
        settings.weekly_trend_gate_enabled = False

    if params.pop("_shock_switch", False):
        settings.shock_killswitch_enabled = True
    else:
        settings.shock_killswitch_enabled = False

    def restore():
        settings.weekly_trend_gate_enabled = orig_weekly
        settings.shock_killswitch_enabled = orig_shock

    return params, restore


def run_candidate(name: str, params: dict, price_data: dict, years: float) -> dict:
    """Run a single candidate and return results dict."""
    clean_params, restore = _apply_settings_overrides(params.copy())

    try:
        result = run_model_backtest("mean_reversion", price_data, clean_params)
    finally:
        restore()

    m = result.metrics

    # Expiry MFE>2% rate
    expiry_trades = [t for t in result.trades if t.exit_reason == "expiry"]
    expiry_mfe_gt2 = sum(1 for t in expiry_trades if t.mfe_pct > 2.0)
    expiry_mfe_rate = expiry_mfe_gt2 / len(expiry_trades) if expiry_trades else 0.0

    out = {
        "name": name,
        "years": years,
        "run_date": str(date.today()),
        "metrics": asdict(m),
        "by_exit_reason": result.by_exit_reason,
        "avg_mfe_pct": result.avg_mfe_pct,
        "avg_mae_pct": result.avg_mae_pct,
        "expiry_mfe_gt_2pct": round(expiry_mfe_rate, 4),
    }

    # Save trades
    out_dir = Path("outputs/research/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)
    horizon = f"{years}Y" if years == int(years) else f"{years}Y"
    save_trades_csv(result.trades, out_dir / f"phase2_{name}_{horizon}_trades.csv")

    return out


def check_gates(champion: dict, candidate: dict) -> dict:
    """Check acceptance gates between champion and candidate."""
    cm = champion["metrics"]
    tm = candidate["metrics"]
    gates = {}

    wr_delta = tm["win_rate"] - cm["win_rate"]
    gates["win_rate"] = {
        "pass": wr_delta >= ACCEPTANCE_GATES["win_rate_delta_min"],
        "champion": cm["win_rate"],
        "candidate": tm["win_rate"],
        "delta": round(wr_delta, 4),
    }

    gates["expectancy"] = {
        "pass": tm["expectancy"] >= cm["expectancy"] * ACCEPTANCE_GATES["expectancy_min_ratio"],
        "champion": cm["expectancy"],
        "candidate": tm["expectancy"],
    }

    gates["profit_factor"] = {
        "pass": tm["profit_factor"] >= cm["profit_factor"] - ACCEPTANCE_GATES["profit_factor_tolerance"],
        "champion": cm["profit_factor"],
        "candidate": tm["profit_factor"],
    }

    gates["max_drawdown"] = {
        "pass": tm["max_drawdown_pct"] <= cm["max_drawdown_pct"] * ACCEPTANCE_GATES["max_drawdown_ratio"],
        "champion": cm["max_drawdown_pct"],
        "candidate": tm["max_drawdown_pct"],
    }

    gates["trade_count"] = {
        "pass": tm["total_trades"] >= cm["total_trades"] * ACCEPTANCE_GATES["trade_count_ratio_min"],
        "champion": cm["total_trades"],
        "candidate": tm["total_trades"],
    }

    return gates


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Win-Rate Lift A/B backtest")
    parser.add_argument("--years", type=float, action="append",
                        help="Horizon(s) to test (can specify multiple)")
    parser.add_argument("--candidates", type=str,
                        help="Comma-separated candidate names (default: all)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    horizons = args.years or [1.0]
    candidate_names = (
        [c.strip() for c in args.candidates.split(",")]
        if args.candidates
        else list(CANDIDATES.keys())
    )

    tickers = get_sp500_tickers()
    out_dir = Path("outputs/research/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)

    for years in horizons:
        horizon = f"{int(years)}Y" if years == int(years) else f"{years}Y"
        print(f"\n{'='*70}")
        print(f"  Phase 2 A/B Matrix — {horizon}")
        print(f"{'='*70}")

        price_data = fetch_ohlcv(tickers, years=years, no_cache=args.no_cache)
        if not price_data:
            print("No data!")
            continue

        results = {}
        for name in candidate_names:
            if name not in CANDIDATES:
                print(f"  Unknown candidate: {name}, skipping")
                continue
            print(f"\n  Running: {name}")
            results[name] = run_candidate(name, CANDIDATES[name], price_data, years)

        # Summary table
        print(f"\n{'='*70}")
        print(f"  RESULTS — {horizon}")
        print(f"{'='*70}")
        header = f"  {'Name':25s} {'Trades':>7s} {'WR':>7s} {'Avg':>8s} {'Sharpe':>7s} {'Sortino':>8s} {'PF':>6s} {'MaxDD':>7s}"
        print(header)
        print(f"  {'-'*len(header.strip())}")

        for name, r in results.items():
            m = r["metrics"]
            marker = " <-- champion" if name == "champion" else ""
            print(
                f"  {name:25s} {m['total_trades']:7d} {m['win_rate']:7.1%} "
                f"{m['avg_return_pct']:+7.2f}% {m['sharpe_ratio']:7.2f} "
                f"{m['sortino_ratio']:8.2f} {m['profit_factor']:6.2f} "
                f"{m['max_drawdown_pct']:6.2f}%{marker}"
            )

        # Gate checks vs champion
        if "champion" in results:
            champion = results["champion"]
            print(f"\n  Acceptance Gates vs Champion:")
            for name, r in results.items():
                if name == "champion":
                    continue
                gates = check_gates(champion, r)
                passed = sum(1 for g in gates.values() if g["pass"])
                total = len(gates)
                status = "ALL PASS" if passed == total else f"{passed}/{total}"
                wr_delta = gates["win_rate"]["delta"]
                print(f"    {name:25s}: {status}  (WR delta: {wr_delta:+.1%})")

        # Save results
        results_path = out_dir / f"phase2_results_{horizon}_{date.today()}.json"
        results_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n  Saved to {results_path}")


if __name__ == "__main__":
    main()
