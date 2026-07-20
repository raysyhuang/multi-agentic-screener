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


def _classify(r: dict) -> str:
    """Map a cohort row to replay class from the audit fields.

    - 'skipped'  : never filled (e.g. gap_above_limit no-chase). Counted in the
                   all-signal denominator; NOT simulated as a trade.
    - 'censored' : genuine recent-open, unresolved. Counted; NOT simulated.
    - 'resolved' : a real filled+closed trade → replay through the exit engine.
    """
    audit = (r.get("audit_status") or "").strip().lower()
    if audit == "skipped" or (r.get("skip_reason") or "").strip():
        return "skipped"
    if audit in ("recent_open", "open") or (r.get("live_status") or "").strip().lower() == "censored":
        return "censored"
    return "resolved"


def _load_cohort(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            sd = _parse_day(r.get("signal_date", ""))
            if not r.get("ticker") or sd is None:
                continue

            def _f(key):
                v = r.get(key)
                return float(v) if v not in (None, "") else None

            rows.append({
                "ticker": r["ticker"].strip().upper(),
                "signal_date": sd,
                "stop_loss": _f("stop_loss"),
                "target_1": _f("target_1"),
                "holding_period": int(float(r.get("holding_period") or 3)),
                "live_pnl_pct": _f("live_pnl_pct"),
                "signal_source": (r.get("signal_source") or "unknown").strip() or "unknown",
                "skip_reason": (r.get("skip_reason") or "").strip() or None,
                "cls": _classify(r),
            })
    return rows


async def _ohlcv(ticker: str, cache: dict, fetched: dict, poly, start: date, end: date) -> pd.DataFrame | None:
    """Return OHLCV covering [start, end]. Prefer the cache ONLY if it actually
    covers the window (the shipped parquet is stale — ends April — so recent live
    picks must come from a fresh Polygon fetch, memoized per ticker)."""
    c = cache.get(ticker)
    if c is not None and not c.empty and c["date"].min() <= start and c["date"].max() >= end:
        return c
    if ticker in fetched:
        return fetched[ticker]
    if poly is None:
        return None
    try:
        df = await poly.get_ohlcv(ticker, start, end)
        df = df if df is not None and not df.empty else None
    except Exception:
        df = None
    fetched[ticker] = df
    return df


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

    # Fetch each ticker ONCE over the whole cohort span (so multiple picks per
    # ticker at different dates are all covered), memoized in `fetched`.
    span_start = min(r["signal_date"] for r in cohort) - timedelta(days=10)
    span_end = max(r["signal_date"] + timedelta(days=r["holding_period"]) for r in cohort) + timedelta(days=10)

    cost = args.cost_bps / 10000.0
    fetched: dict = {}

    # All-signal denominator (never silently drop): classify every row.
    counts = {"resolved": 0, "skipped": 0, "censored": 0}
    for r in cohort:
        counts[r["cls"]] += 1
    print(f"\nAll-signal denominator ({len(cohort)}): "
          f"{counts['resolved']} resolved, {counts['skipped']} skipped/unfilled "
          f"(gap_above_limit etc.), {counts['censored']} censored (recent-open).")

    # Replay ONLY resolved (filled+closed) rows. Skipped rows were never filled —
    # simulating them as trades would fabricate fills; they stay in the denominator.
    replayed, unresolvable = [], []
    for row in cohort:
        if row["cls"] != "resolved":
            continue
        if row["stop_loss"] is None or row["target_1"] is None:
            unresolvable.append(row)
            continue
        df = await _ohlcv(row["ticker"], cache, fetched, poly, span_start, span_end)
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

    if unresolvable:
        print(f"({len(unresolvable)} resolved rows had no OHLCV/levels — reported, not dropped)")

    # Report split BY signal_source (provenance must be preserved) + a combined line.
    def _report(label: str, rows: list[dict]) -> None:
        if not rows:
            print(f"\n{label}: 0 replayed")
            return
        eng = np.array([r["engine_pnl_pct"] for r in rows])
        m = compute_metrics(eng.tolist())
        line = (f"\n{label}  (n_replayed={m.total_trades})\n"
                f"  corrected-exit engine: WR={m.win_rate:.1%}  avg={m.avg_return_pct:+.3f}%  "
                f"expectancy={m.expectancy:+.3f}%  PF={m.profit_factor:.2f}  Sharpe={m.sharpe_ratio:.2f}")
        paired = [r for r in rows if r["live_pnl_pct"] is not None]
        if paired:
            live = np.array([r["live_pnl_pct"] for r in paired])
            eng2 = np.array([r["engine_pnl_pct"] for r in paired])
            line += (f"\n  live avg {live.mean():+.3f}%  vs  engine avg {eng2.mean():+.3f}%  "
                     f"(mean delta {(live - eng2).mean():+.3f}% over {len(paired)} resolved)")
        print(line)

    sources = sorted({r["signal_source"] for r in replayed})
    print("\n=== CORRECTED-EXIT replay on the exact cohort (unified engine, gap-through) ===")
    for src in sources:
        _report(f"[{src}]", [r for r in replayed if r["signal_source"] == src])
    _report("[ALL sources combined]", replayed)

    print("\nRead: live ≈ engine  → the doc/backtest label was optimistic-exit bias (retire it).")
    print("      live << engine  → genuine signal decay beyond the exit model.")
    print("Gate: compare to the repaired 90-day live scorecard; preserve official vs manual-sleeve")
    print("provenance; treat skipped (unfilled) and censored (open) rows as denominator, not trades.")


if __name__ == "__main__":
    asyncio.run(main())
