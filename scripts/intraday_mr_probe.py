"""Intraday mean-reversion probe (Candidate 2) — real Polygon minute bars.

Thesis: on liquid names, when intraday price drops meaningfully BELOW the session
VWAP, it tends to revert toward VWAP by the close. The gap-continuation intraday
probe surfaced exactly this shape (open→midday fade→recover), motivating a direct
test.

Diagnostic-first (no trading rule yet): at fixed decision times, bucket every
observation by its deviation from session VWAP and measure the forward return to
the close, versus the unconditional base rate. If the most-below-VWAP buckets show
forward-to-close returns above base rate, intraday MR reverts and is worth building.

Decision times (fixed, few-per-day) avoid the overlap autocorrelation that inflates
significance when sampling every minute.

Usage:
  python scripts/intraday_mr_probe.py --cache-file <daily_parquet> --tickers 20 --days 25
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date

import numpy as np
import pandas as pd

from src.data.intraday_cache import get_intraday_day

# Decision times (exchange time). VWAP is established after the open; we avoid the
# first 90 min of open noise and the last 30 min.
DECISION_TIMES = ["11:00", "12:00", "13:00", "14:00", "15:00"]
CLOSE_TIME = "15:59"
# Deviation buckets (price vs session VWAP, %). Negative = below VWAP.
BUCKETS = [(-100, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 0.0),
           (0.0, 0.5), (0.5, 1.0), (1.0, 100)]


def pick_liquid_tickers(price_data: dict[str, pd.DataFrame], n: int) -> list[str]:
    """Top-n tickers by average daily dollar volume (tighter intraday → cleaner VWAP)."""
    dv = {}
    for t, df in price_data.items():
        if t == "SPY" or df.empty:
            continue
        dv[t] = float((df["close"] * df["volume"]).mean())
    return [t for t, _ in sorted(dv.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def sample_days(price_data: dict[str, pd.DataFrame], n: int) -> list[date]:
    """Evenly sample n trading days across the full window."""
    all_days = sorted({d for df in price_data.values() for d in df["date"].tolist()})
    if len(all_days) <= n:
        return all_days
    step = len(all_days) / n
    return [all_days[int(i * step)] for i in range(n)]


def _bucket(dev: float) -> tuple[float, float] | None:
    for lo, hi in BUCKETS:
        if lo <= dev < hi:
            return (lo, hi)
    return None


def observe_day(bars: pd.DataFrame) -> list[tuple[tuple, float]]:
    """Return (bucket, forward_return_to_close) observations for one ticker-day."""
    if bars.empty:
        return []
    b = bars.copy()
    et = pd.to_datetime(b["datetime"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    b["hm"] = et.dt.strftime("%H:%M")
    rth = b[(b["hm"] >= "09:30") & (b["hm"] <= "16:00")].reset_index(drop=True)
    if len(rth) < 60:
        return []
    # Session VWAP (cumulative), using each minute's own VWAP × volume.
    pv = (rth["vwap"].astype(float) * rth["volume"].astype(float)).cumsum()
    vol = rth["volume"].astype(float).cumsum().replace(0, np.nan)
    rth["session_vwap"] = pv / vol

    close_rows = rth[rth["hm"] <= CLOSE_TIME]
    if close_rows.empty:
        return []
    close_px = float(close_rows.iloc[-1]["close"])

    obs = []
    for t in DECISION_TIMES:
        at = rth[rth["hm"] <= t]
        if at.empty:
            continue
        row = at.iloc[-1]
        px = float(row["close"])
        vwap = float(row["session_vwap"])
        if not np.isfinite(vwap) or vwap <= 0 or px <= 0:
            continue
        dev = (px - vwap) / vwap * 100
        bkt = _bucket(dev)
        if bkt is None:
            continue
        fwd = (close_px - px) / px * 100  # forward return to close
        obs.append((bkt, fwd))
    return obs


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", required=True)
    ap.add_argument("--tickers", type=int, default=20)
    ap.add_argument("--days", type=int, default=25)
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}

    tickers = pick_liquid_tickers(price_data, args.tickers)
    days = sample_days(price_data, args.days)
    print(f"Probing {len(tickers)} liquid tickers × {len(days)} days "
          f"= {len(tickers) * len(days)} ticker-days (cached after first pull)")

    by_bucket: dict[tuple, list[float]] = {b: [] for b in BUCKETS}
    all_fwd: list[float] = []
    failed = 0
    for ticker in tickers:
        for day in days:
            try:
                bars = await get_intraday_day(ticker, day)
            except Exception:
                failed += 1
                continue
            for bkt, fwd in observe_day(bars):
                by_bucket[bkt].append(fwd)
                all_fwd.append(fwd)

    base = float(np.mean(all_fwd)) if all_fwd else 0.0
    if failed:
        print(f"({failed} pulls failed after retries — skipped)")
    print(f"\nForward return to close by VWAP-deviation bucket at decision times")
    print(f"Base rate (all obs): {base:+.3f}%  N={len(all_fwd)}\n")
    print(f"{'VWAP dev bucket':<18}{'fwd→close%':>11}{'edge vs base':>13}{'% up':>8}{'N':>7}")
    print("-" * 57)
    for lo, hi in BUCKETS:
        v = by_bucket[(lo, hi)]
        if not v:
            continue
        a = np.array(v)
        label = f"[{lo:>5}, {hi:>4})%"
        print(f"{label:<18}{a.mean():>11.3f}{(a.mean() - base) * 100:>12.1f}bp"
              f"{(a > 0).mean() * 100:>7.1f}%{len(a):>7}")
    print("\nReversion edge = below-VWAP buckets (negative dev) showing fwd→close "
          "ABOVE base rate.")


if __name__ == "__main__":
    asyncio.run(main())
