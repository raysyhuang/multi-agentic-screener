"""Generate live-faithful mean-reversion trades to a CSV, so the short/credit
filter harness (scripts/sniper_short_credit_filter.py) can test whether a
short-activity AVOID filter improves MR expectancy.

MR buys oversold names (RSI2<=10); some oversold names are heavily-shorted falling
knives that keep falling. The unconditional probes showed heavily-shorted names
drift DOWN monotonically — so a short-based avoid-filter has a plausible home on MR
(where sniper's breakout-fuel logic doesn't apply). Let the conditioned data decide.

Live MR config (per MEMORY): RSI(2)<=10, stop 0.75xATR, target 1.5xATR, hold 3d,
trail 0.5/0.3, gap-through fills. Same unified engine as the truth matrix.

Usage:
  python scripts/gen_mr_trades.py --cache-file outputs/research/ohlcv_polygon_3y.parquet \
      --out outputs/research/mr_trades_polygon.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from src.research.signal_backtest import run_model_backtest

LIVE_MR = {
    "rsi2_threshold": 10.0,
    "min_score": 50.0,
    "stop_atr_mult": 0.75,
    "target_atr_mult": 1.5,
    "holding_period": 3,
    "trail_activate_pct": 0.5,
    "trail_distance_pct": 0.3,
    "gap_through": True,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default="outputs/research/ohlcv_polygon_3y.parquet")
    ap.add_argument("--out", default="outputs/research/mr_trades_polygon.csv")
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}
    print(f"Loaded {len(price_data)} tickers; running live-faithful MR backtest...")

    result = run_model_backtest("mean_reversion", price_data, LIVE_MR)
    trades = result.trades
    m = result.metrics
    print(f"MR trades: {m.total_trades}  WR={m.win_rate:.1%}  "
          f"avg={m.avg_return_pct:+.3f}%  expectancy={m.expectancy:+.3f}%")

    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "entry_date", "exit_date", "pnl_pct", "regime", "score", "exit_reason"])
        for t in trades:
            w.writerow([t.ticker, t.entry_date, t.exit_date, round(t.pnl_pct, 4),
                        t.regime, t.score, t.exit_reason])
    print(f"Wrote {len(trades)} trades -> {out}")


if __name__ == "__main__":
    main()
