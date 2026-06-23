#!/usr/bin/env python
"""Research-only: evaluate the sniper bear-regime hard block.

Drives the production sniper backtest simulator (run_model_backtest) off an
existing cached OHLCV parquet that already contains the market-reference tickers
(SPY/QQQ/^VIX/^TNX/^IRX) needed to build the point-in-time signal-date regime map.
This avoids a fresh yfinance pull (which is being rate-limited).

It runs the SAME data twice:
  - control:   allow_bear=False  -> reproduces the official baseline (bear bucket empty)
  - treatment: allow_bear=True   -> simulates the bear-regime trades the block suppresses

Only the bear gate differs between the two runs, so the bear bucket in the treatment
run is the clean evidence for whether the hard block is justified.

Usage:
    python scripts/run_sniper_bear_research.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research.signal_backtest import (
    run_model_backtest,
    format_model_report,
    save_trades_csv,
)

# Cache with full market-reference tickers + 3Y window (matches post-regime-fix baseline).
CACHE_FILE = Path("data/cache/ohlcv/ohlcv_07d5de9616a8_2023-04-30_2026-04-29.parquet")

# V3 sniper params — identical to scripts/run_sniper_backtest.py SNIPER_PARAMS.
SNIPER_PARAMS = {
    "min_score": 70,
    "atr_pct_floor": 5.0,
    "stop_atr_mult": 1.5,
    "target_atr_mult": 3.0,
    "holding_period": 7,
    "trail_activate_pct": 1.0,
    "trail_distance_pct": 0.5,
}

OUT_DIR = Path("outputs/research")


def load_price_data(path: Path) -> dict[str, pd.DataFrame]:
    """Reconstruct the price_data dict from a cached parquet (mirrors fetch_ohlcv)."""
    combined = pd.read_parquet(path)
    out: dict[str, pd.DataFrame] = {}
    for ticker, grp in combined.groupby("_ticker"):
        out[ticker] = grp.drop(columns=["_ticker"]).reset_index(drop=True)
    return out


def summarize(result, label: str) -> dict:
    print(f"\n{'='*64}\n  {label}\n{'='*64}")
    print(format_model_report(result))
    return {
        "label": label,
        "metrics": asdict(result.metrics),
        "by_regime": {k: asdict(v) for k, v in result.by_regime.items()},
        "by_exit_reason": result.by_exit_reason,
        "total_trades": result.metrics.total_trades,
    }


def main():
    if not CACHE_FILE.exists():
        print(f"Cache not found: {CACHE_FILE}")
        sys.exit(1)

    price_data = load_price_data(CACHE_FILE)
    print(f"Loaded {len(price_data)} tickers from {CACHE_FILE.name}")
    span = pd.read_parquet(CACHE_FILE, columns=["date"])["date"]
    print(f"Data window: {span.min()} -> {span.max()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for label, allow_bear in [("control_block_on", False), ("treatment_block_off", True)]:
        params = dict(SNIPER_PARAMS, allow_bear=allow_bear)
        result = run_model_backtest("sniper", price_data, params)
        results[label] = summarize(result, f"{label} (allow_bear={allow_bear})")
        save_trades_csv(
            result.trades,
            OUT_DIR / f"backtest_sniper_bear_research_{label}_trades.csv",
        )

    out_path = OUT_DIR / "sniper_bear_research.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved comparison to {out_path}")

    # Focused bear verdict
    print(f"\n{'='*64}\n  BEAR-BUCKET VERDICT (treatment run)\n{'='*64}")
    treat = results["treatment_block_off"]["by_regime"]
    for rname in ["bull", "choppy", "bear"]:
        m = treat.get(rname)
        if not m:
            print(f"  {rname:7s}: (no trades)")
            continue
        print(f"  {rname:7s}: n={m['total_trades']:4d}  WR={m['win_rate']:6.1%}  "
              f"avg={m['avg_return_pct']:+6.2f}%  PF={m['profit_factor']:5.2f}  "
              f"Sharpe={m['sharpe_ratio']:5.2f}  MaxDD={m['max_drawdown_pct']:5.1f}%")


if __name__ == "__main__":
    main()
