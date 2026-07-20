"""Cohort replay — reconcile a frozen live cohort against the unified exit engine.

For the MR reconciliation (and any model): take the EXACT frozen, resolution-
repaired cohort of live picks and replay each through the shared unified exit
engine (simulate_trade, realistic gap-through fills). This is apples-to-apples —
same picks, same entry dates, only the exit model swapped to the corrected engine —
so the backtest-vs-live gap can be decomposed into (a) exit-model bias vs
(b) genuine signal decay, rather than compared against a generic 3Y backtest.

Gate discipline: this consumes a frozen cohort artifact. Treat unresolved live rows
as CENSORED (kept, flagged), never silently dropped — pass them through with a
blank live_pnl_pct and they are reported separately.

Input CSV contract (one row per live pick; * = required):
  ticker*          e.g. AAPL
  signal_date*     YYYY-MM-DD (the as-of/close date; entry is the next open)
  stop_loss*       absolute price
  target_1*        absolute price
  holding_period*  integer days (MR = 3)
  live_pnl_pct     optional — the (repaired) realized live return, for gap decomposition
  live_status      optional — resolved | open | censored (unresolved rows kept, flagged)

OHLCV comes from the cached parquet if the ticker is present, else a Polygon fetch.

Usage:
  python scripts/cohort_replay.py --cohort <cohort.csv> [--cache-file <parquet>] [--cost-bps 5]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_metrics
from src.research.signal_backtest import simulate_trade


def _parse_day(s: str) -> date | None:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _load_cohort(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            sd = _parse_day(r.get("signal_date", ""))
            if not r.get("ticker") or sd is None:
                continue
            rows.append({
                "ticker": r["ticker"].strip().upper(),
                "signal_date": sd,
                "stop_loss": float(r["stop_loss"]),
                "target_1": float(r["target_1"]),
                "holding_period": int(float(r.get("holding_period") or 3)),
                "live_pnl_pct": float(r["live_pnl_pct"]) if r.get("live_pnl_pct") not in (None, "") else None,
                "live_status": (r.get("live_status") or "").strip().lower() or None,
            })
    return rows


async def _ohlcv(ticker: str, cache: dict, poly, start: date, end: date) -> pd.DataFrame | None:
    if ticker in cache:
        return cache[ticker]
    if poly is None:
        return None
    try:
        df = await poly.get_ohlcv(ticker, start, end)
        return df if df is not None and not df.empty else None
    except Exception:
        return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="frozen cohort CSV (see module docstring)")
    ap.add_argument("--cache-file", default="", help="daily OHLCV parquet for lookups")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="per-side cost bps")
    args = ap.parse_args()

    cohort = _load_cohort(args.cohort)
    print(f"Loaded {len(cohort)} cohort rows from {args.cohort}")

    cache: dict[str, pd.DataFrame] = {}
    if args.cache_file:
        combined = pd.read_parquet(args.cache_file)
        cache = {t: g.drop(columns=["_ticker"]).sort_values("date").reset_index(drop=True)
                 for t, g in combined.groupby("_ticker")}
    poly = None
    try:
        from src.data.polygon_client import PolygonClient
        poly = PolygonClient()
    except Exception:
        pass

    cost = args.cost_bps / 10000.0
    replayed, censored, unresolvable = [], [], []
    for row in cohort:
        # Unresolved live rows are censored — kept and flagged, never dropped.
        if row["live_status"] in ("open", "censored") and row["live_pnl_pct"] is None:
            censored.append(row)
        df = await _ohlcv(row["ticker"], cache, poly,
                          row["signal_date"] - timedelta(days=10),
                          row["signal_date"] + timedelta(days=row["holding_period"] + 10))
        if df is None or df.empty:
            unresolvable.append(row)
            continue
        res = simulate_trade(
            df, row["signal_date"], stop_loss=row["stop_loss"], target=row["target_1"],
            max_hold=row["holding_period"], slippage=cost, gap_through=True,
        )
        if res is None:
            unresolvable.append(row)
            continue
        replayed.append({**row, "engine_pnl_pct": res["pnl_pct"], "exit_reason": res["exit_reason"]})

    print(f"\nReplayed {len(replayed)} through the unified engine; "
          f"{len(censored)} censored (unresolved live); {len(unresolvable)} no-OHLCV.\n")

    if replayed:
        eng = np.array([r["engine_pnl_pct"] for r in replayed])
        m = compute_metrics(eng.tolist())
        print("CORRECTED-EXIT backtest on the exact cohort (unified engine, gap-through):")
        print(f"  N={m.total_trades}  WR={m.win_rate:.1%}  avg={m.avg_return_pct:+.2f}%  "
              f"expectancy={m.expectancy:+.3f}%  PF={m.profit_factor:.2f}  Sharpe={m.sharpe_ratio:.2f}")

        paired = [r for r in replayed if r["live_pnl_pct"] is not None]
        if paired:
            live = np.array([r["live_pnl_pct"] for r in paired])
            eng2 = np.array([r["engine_pnl_pct"] for r in paired])
            print(f"\nGap decomposition on {len(paired)} resolved rows (live vs corrected engine):")
            print(f"  live avg  {live.mean():+.3f}%   engine avg {eng2.mean():+.3f}%   "
                  f"mean delta {(live - eng2).mean():+.3f}%")
            print("  If live ~= engine: the doc/backtest baseline was the (optimistic)")
            print("  exit-model bias. If live << engine: genuine signal decay beyond exits.")
        else:
            print("\n(no resolved live_pnl_pct provided — supply it for the gap decomposition)")
    print("\nGate: this is the corrected-exit baseline on the frozen cohort. Compare to the")
    print("repaired 90-day live scorecard; do not treat as a headline until both are frozen.")


if __name__ == "__main__":
    asyncio.run(main())
