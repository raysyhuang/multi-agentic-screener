"""Measure the minute-bar fill upgrade: does resolving same-bar stop-vs-target
ties from Polygon 1-minute data change the backtest vs the conservative
stop-first daily assumption?

Runs a model twice — daily-only, then minute-resolved — and reports the delta.
Sniper is the fill-sensitive one (the 82%->54% collapse was about fills), so
default to it. Minute bars are fetched only for ambiguous bars, disk-cached.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.research.signal_backtest import run_model_backtest

SNIPER = dict(use_spy=True, min_score=70, atr_pct_floor=5.0, stop_atr_mult=1.5,
              target_atr_mult=3.0, holding_period=7, gap_through=True,
              sniper_time_stop_days=1, trail_activate_pct=0.5, trail_distance_pct=0.3)
MR = dict(min_score=60, rsi2_threshold=10, stop_atr_mult=0.75, target_atr_mult=1.5,
          holding_period=3, gap_through=True, trail_activate_pct=0.5, trail_distance_pct=0.3)


def _summ(res, label):
    m = res.metrics
    print(f"  {label:18s} N={m.total_trades:>5} WR={m.win_rate:>6.1%} "
          f"avg={m.avg_return_pct:>+7.3f}% PF={m.profit_factor:>5.2f} "
          f"target_exits={res.by_exit_reason.get('target', 0)}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default="outputs/research/ohlcv_polygon_3y.parquet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="sniper")
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
             for t, g in combined.groupby("_ticker")}
    if args.limit:
        keep = list(price)[: args.limit]
        if "SPY" not in keep and "SPY" in price:
            keep.append("SPY")  # sniper relative-strength needs SPY
        price = {t: price[t] for t in keep}
    print(f"{len(price)} tickers, model={args.model}\n")

    entry = SNIPER if args.model == "sniper" else MR

    print("-- daily-only (conservative stop-first on same-bar ties) --")
    base = _summ(run_model_backtest(args.model, price, {**entry}), "daily-only")

    print("\n-- minute-resolved same-bar ties (Polygon 1-min) --")
    mres = _summ(run_model_backtest(args.model, price, {**entry, "use_minute_resolver": True}),
                 "minute-resolved")

    print(f"\nΔ  WR {mres.win_rate - base.win_rate:+.2%}   "
          f"avg {mres.avg_return_pct - base.avg_return_pct:+.3f}pp   "
          f"PF {mres.profit_factor - base.profit_factor:+.2f}")
    print("(minute resolution can only move exits stop→target, so WR/avg should "
          "rise if same-bar ties are material — the size of the move is the answer.)")


if __name__ == "__main__":
    main()
