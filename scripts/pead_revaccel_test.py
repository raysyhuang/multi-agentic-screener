"""E1 guidance-raise PROXY: does a PEAD beat with ACCELERATING revenue growth drift
more? The literal guidance-raise (forward-estimate up-revision) is paywalled on FMP
Starter ($29) — analyst_estimates/financial-estimates are plan-gated. Revenue
*actuals* ARE on Starter, so we test the achievable fundamental-quality dimension
in the same spirit: a beat where YoY revenue growth is ACCELERATING (this quarter's
YoY growth > the prior quarter's YoY growth) is where managements typically raise
guidance.

This is NOT the revenue-SURPRISE gate already shipped in E1 (actual vs analyst
estimate). It is revenue-GROWTH acceleration (sequential YoY inflection) — a
different, orthogonal signal, fully point-in-time from earnings actuals.

For each beat event we compute, from the ticker's sorted quarterly earnings rows:
  yoy_now  = rev[t]   / rev[t-4] - 1   (this quarter's YoY growth)
  yoy_prev = rev[t-1] / rev[t-5] - 1   (prior quarter's YoY growth)
  accel    = yoy_now - yoy_prev        (> 0 = accelerating)
All look-ahead-safe (only revenue reported on/before the event day is used). Drift
measured through the same unified engine / cost as pead_backtest.

Usage:
  python scripts/pead_revaccel_test.py --cache-file outputs/research/ohlcv_polygon_3y.parquet
"""
from __future__ import annotations

import argparse
import asyncio

import numpy as np
import pandas as pd

from scripts.pead_backtest import _parse_day, _summ, build_events, run_config
from src.data.earnings_cache import get_earnings


def _rev(row) -> float | None:
    v = row.get("revenueActual")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


async def _accel_by_ticker_date(ticker: str) -> dict:
    """Map event date -> revenue-growth acceleration for a ticker.

    Uses only quarterly revenue reported up to and including each row (the row's
    own report date), so it is point-in-time at the announcement."""
    rows = await get_earnings(ticker)
    dated = []
    for r in rows:
        ed = _parse_day(r.get("date", ""))
        rv = _rev(r)
        if ed is not None and rv is not None:
            dated.append((ed, rv))
    dated.sort(key=lambda x: x[0])
    # Deduplicate same-date rows (keep last).
    dedup: dict = {}
    for ed, rv in dated:
        dedup[ed] = rv
    seq = sorted(dedup.items())
    out: dict = {}
    for i in range(len(seq)):
        if i < 5:
            continue  # need t, t-1, t-4, t-5
        ed = seq[i][0]
        rev_t, rev_t1 = seq[i][1], seq[i - 1][1]
        rev_t4, rev_t5 = seq[i - 4][1], seq[i - 5][1]
        if min(rev_t, rev_t1, rev_t4, rev_t5) <= 0:
            continue
        yoy_now = rev_t / rev_t4 - 1
        yoy_prev = rev_t1 / rev_t5 - 1
        out[ed] = yoy_now - yoy_prev
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default="outputs/research/ohlcv_polygon_3y.parquet")
    ap.add_argument("--min-surprise", type=float, default=10.0,
                    help="EPS surprise threshold (10 = the strong-beat cohort E1 uses)")
    ap.add_argument("--stop-atr", type=float, default=3.0)
    ap.add_argument("--target-atr", type=float, default=6.0)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--cost-bps", type=float, default=7.5)
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}
    cost = args.cost_bps / 10000.0

    print(f"Building PEAD beat events (>{args.min_surprise:.0f}% EPS surprise)...")
    events = await build_events(price_data, args.min_surprise)

    # Tag each event with revenue-growth acceleration (point-in-time).
    accel_cache: dict = {}
    tagged = 0
    for ev in events:
        tk = ev["ticker"]
        if tk not in accel_cache:
            accel_cache[tk] = await _accel_by_ticker_date(tk)
        # Match the event's signal_date (T+1 trading day) to the nearest earnings
        # row on/before it: the event's date came from an earnings row, so find the
        # accel keyed at the most recent report date <= signal_date.
        sd = ev["signal_date"]
        cand = [d for d in accel_cache[tk] if d <= sd]
        ev["accel"] = accel_cache[tk][max(cand)] if cand else None
        if ev["accel"] is not None:
            tagged += 1
    print(f"{len(events)} beat events; {tagged} tagged with revenue-accel "
          f"(need 5+ quarters of revenue history)\n")

    accel_pos = [e for e in events if e["accel"] is not None and e["accel"] > 0]
    accel_neg = [e for e in events if e["accel"] is not None and e["accel"] <= 0]

    hdr = (f"{'cohort':<22}{'N':>6}{'WR':>8}{'avg%':>8}{'expect%':>9}{'PF':>7}"
           f"{'Sharpe':>7}{'equity×':>8}{'CAGR%':>8}{'eqDD%':>7}")
    print(f"stop {args.stop_atr}xATR / target {args.target_atr}xATR / hold {args.hold}d / "
          f"cost {args.cost_bps}bp/side")
    print(hdr); print("-" * len(hdr))
    for cohort, label in [(events, "all beats"),
                          (accel_pos, "+ rev accel (>0)"),
                          (accel_neg, "- rev decel (<=0)")]:
        trades, eqt = run_config(cohort, stop_atr=args.stop_atr, target_atr=args.target_atr,
                                 hold=args.hold, cost=cost)
        print(_summ(trades, eqt, label))

    # Sub-period stability of the accelerating cohort (guards bull-window beta).
    trades, eqt = run_config(accel_pos, stop_atr=args.stop_atr, target_atr=args.target_atr,
                             hold=args.hold, cost=cost)
    if trades:
        sd = sorted(t["signal_date"] for t in trades)
        edges = (sd[len(sd) // 3], sd[2 * len(sd) // 3])
        print("\nSub-period stability (+ rev accel):")
        print(hdr); print("-" * len(hdr))
        for k, nm in enumerate(("early third", "mid third", "late third")):
            st, se = [], []
            for t, e in zip(trades, eqt):
                third = 0 if t["signal_date"] <= edges[0] else (1 if t["signal_date"] <= edges[1] else 2)
                if third == k:
                    st.append(t); se.append(e)
            print(_summ(st, se, nm))

    # Reference: mean accel in each cohort (sanity that the split is meaningful).
    if accel_pos or accel_neg:
        pos_m = np.mean([e["accel"] for e in accel_pos]) if accel_pos else float("nan")
        neg_m = np.mean([e["accel"] for e in accel_neg]) if accel_neg else float("nan")
        print(f"\nmean YoY-accel: accelerating cohort {pos_m:+.3f}, "
              f"decelerating cohort {neg_m:+.3f}")


if __name__ == "__main__":
    asyncio.run(main())
