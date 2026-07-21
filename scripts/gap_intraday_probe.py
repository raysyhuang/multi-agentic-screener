"""Intraday drift probe for gap-continuation signals (uses real Polygon minute bars).

The daily research found gap-up signals carry a real but tiny +13 bps/3-day drift
that ATR stops destroy. The decisive question for whether intraday execution can
rescue the edge: does that drift accrue INTRADAY on the entry day (so an
opening-range / VWAP intraday rule could capture it), or is the day flat?

This pulls minute bars for the T+1 entry day of a bounded sample of gap signals
(cached per ticker-day) and builds the average intraday return path from the T+1
open, at regular-session checkpoints. Pure diagnostic — no trading rule yet.

Usage:
  python scripts/gap_intraday_probe.py --cache-file <daily_parquet> --sample 120
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date

import numpy as np
import pandas as pd

from src.data.intraday_cache import get_intraday_day
from src.features.technical import compute_all_technical_features

MIN_HISTORY = 60
# Regular-session checkpoints in EXCHANGE time (America/New_York), so DST is
# handled correctly year-round. Polygon minute timestamps are UTC ms; we convert.
CHECKPOINTS = [("open+15m", "09:45"), ("open+30m", "10:00"), ("+1h", "10:30"),
               ("+2h", "11:30"), ("midday", "12:00"), ("+4h", "13:30"),
               ("close-1h", "15:00"), ("close", "15:59")]
RTH_OPEN, RTH_CLOSE = "09:30", "16:00"


def gap_signals(df: pd.DataFrame, gap_min: float = 4.0) -> list[tuple[date, date]]:
    """Return (signal_day T0, entry_day T+1) pairs for gap-up signals."""
    if len(df) < MIN_HISTORY:
        return []
    e = compute_all_technical_features(df).reset_index(drop=True)
    dates = e["date"].tolist()
    out = []
    for i in range(MIN_HISTORY, len(e) - 1):
        row = e.iloc[i]
        gap, rvol, close = row.get("gap_pct"), row.get("rvol"), float(row["close"])
        high, low, sma50 = float(row["high"]), float(row["low"]), row.get("sma_50")
        if gap is None or pd.isna(gap) or float(gap) < gap_min:
            continue
        if rvol is None or pd.isna(rvol) or float(rvol) < 1.5:
            continue
        rng = high - low
        if rng <= 0 or (close - low) / rng < 0.5:
            continue
        if sma50 is None or pd.isna(sma50) or close <= float(sma50):
            continue
        out.append((dates[i], dates[i + 1]))  # (T0 gap day, T+1 entry day)
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", required=True)
    ap.add_argument("--sample", type=int, default=120, help="max signals to probe (bounds API pulls)")
    ap.add_argument("--gap-min", type=float, default=4.0)
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}

    # Collect (ticker, entry_day) signals, then evenly subsample to bound pulls.
    all_sigs = []
    for ticker, df in price_data.items():
        if ticker == "SPY":
            continue
        for _t0, entry_day in gap_signals(df.sort_values("date").reset_index(drop=True), args.gap_min):
            all_sigs.append((ticker, entry_day))
    print(f"{len(all_sigs)} gap signals found; sampling up to {args.sample}")
    if len(all_sigs) > args.sample:
        step = len(all_sigs) / args.sample
        all_sigs = [all_sigs[int(i * step)] for i in range(args.sample)]

    # For each signal pull the T+1 minute bars and record the return path.
    paths: dict[str, list[float]] = {label: [] for label, _ in CHECKPOINTS}
    used = 0
    failed = 0
    for ticker, entry_day in all_sigs:
        try:
            bars = await get_intraday_day(ticker, entry_day)
        except Exception:
            failed += 1
            continue
        if bars.empty:
            continue
        bars = bars.copy()
        # Convert UTC minute timestamps to exchange time (handles DST correctly).
        et = pd.to_datetime(bars["datetime"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        bars["hm"] = et.dt.strftime("%H:%M")
        rth = bars[(bars["hm"] >= RTH_OPEN) & (bars["hm"] <= RTH_CLOSE)]
        if rth.empty:
            continue
        entry = float(rth.iloc[0]["open"])
        if entry <= 0:
            continue
        used += 1
        for label, hm in CHECKPOINTS:
            at = rth[rth["hm"] <= hm]
            if not at.empty:
                px = float(at.iloc[-1]["close"])
                paths[label].append((px - entry) / entry * 100)

    if failed:
        print(f"({failed} pulls failed after retries — skipped)")
    print(f"\nIntraday drift on the T+1 entry day (from RTH open), N={used} signals")
    print(f"{'checkpoint':<12}{'avg %':>9}{'median %':>10}{'% up':>8}{'N':>7}")
    print("-" * 46)
    for label, _ in CHECKPOINTS:
        v = paths[label]
        if not v:
            continue
        a = np.array(v)
        print(f"{label:<12}{a.mean():>9.3f}{np.median(a):>10.3f}{(a > 0).mean() * 100:>7.1f}%{len(a):>7}")


if __name__ == "__main__":
    asyncio.run(main())
