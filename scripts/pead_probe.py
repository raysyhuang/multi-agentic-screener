"""Post-earnings-drift (PEAD) probe — Candidate 3 — daily bars + FMP earnings.

Thesis (the classic PEAD anomaly): after an earnings report, stocks drift in the
direction of the surprise for several weeks. For a long-only book: buy the beats,
they keep rising.

Diagnostic-first (no trading rule yet): for every earnings event in the window,
bucket by EPS surprise and measure the forward drift from the day AFTER the
announcement day (T+1, fully post-reaction) to +5 / +10 / +20 trading days,
versus the unconditional base rate. If beats drift above base and misses below,
PEAD is present and capturable.

Point-in-time safe: entry is the first trading day strictly AFTER the first
trading day on/after the FMP report date, so the announcement-day reaction itself
is never part of the measured drift.

Usage:
  python scripts/pead_probe.py --cache-file <daily_parquet> [--gap-min-surprise 5]
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.data.earnings_cache import get_earnings

HORIZONS = [5, 10, 20]
# Surprise buckets on (epsActual - epsEstimated) / |epsEstimated| * 100.
BUCKETS = [("big miss", -1e9, -25), ("miss", -25, -2), ("inline", -2, 2),
           ("beat", 2, 25), ("big beat", 25, 1e9)]


def _parse_day(s: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:len(fmt) + 2].strip(), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _surprise_pct(row: dict) -> float | None:
    a, e = row.get("epsActual"), row.get("epsEstimated")
    if a is None or e is None:
        return None
    try:
        a, e = float(a), float(e)
    except (TypeError, ValueError):
        return None
    if abs(e) < 1e-6:
        return None
    return (a - e) / abs(e) * 100


def _bucket(sp: float) -> str | None:
    for name, lo, hi in BUCKETS:
        if lo <= sp < hi:
            return name
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", required=True)
    ap.add_argument("--no-cache", action="store_true", help="force refetch earnings")
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).sort_values("date").reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}
    win_start = min(df["date"].min() for df in price_data.values())
    win_end = max(df["date"].max() for df in price_data.values())
    print(f"{len(price_data)} tickers, window {win_start}..{win_end}; pulling earnings (cached)...")

    by_bucket: dict[str, dict[int, list[float]]] = {b[0]: {h: [] for h in HORIZONS} for b in BUCKETS}
    base: dict[int, list[float]] = {h: [] for h in HORIZONS}
    n_events = 0
    n_tickers = 0

    for ticker, df in price_data.items():
        if ticker == "SPY":
            continue
        closes = df["close"].to_numpy(dtype=float)
        opens = df["open"].to_numpy(dtype=float)
        days = df["date"].tolist()
        day_to_idx = {d: i for i, d in enumerate(days)}
        n = len(df)

        # Base rate: forward close-to-close returns from every bar (this ticker).
        for h in HORIZONS:
            if n > h:
                base[h].extend(((closes[h:] - closes[:-h]) / closes[:-h] * 100).tolist())

        rows = await get_earnings(ticker, no_cache=args.no_cache)
        if rows:
            n_tickers += 1
        for row in rows:
            ed = _parse_day(str(row.get("date", "")))
            if ed is None or ed < win_start or ed > win_end:
                continue
            sp = _surprise_pct(row)
            if sp is None:
                continue
            bkt = _bucket(sp)
            if bkt is None:
                continue
            # First trading day on/after the report date = announcement day E'.
            e_idx = next((day_to_idx[d] for d in days if d >= ed), None)
            if e_idx is None:
                continue
            entry_idx = e_idx + 1  # day AFTER the announcement day (post-reaction)
            if entry_idx >= n:
                continue
            entry = opens[entry_idx]  # T+1 open entry
            if entry <= 0:
                continue
            n_events += 1
            for h in HORIZONS:
                if entry_idx + h < n:
                    by_bucket[bkt][h].append((closes[entry_idx + h] - entry) / entry * 100)

    print(f"{n_events} earnings events matched across {n_tickers} tickers\n")
    print("Forward drift from T+1 open (post-announcement) by EPS-surprise bucket")
    base_avg = {h: (float(np.mean(base[h])) if base[h] else 0.0) for h in HORIZONS}
    print(f"base rate  +5d {base_avg[5]:+.2f}%   +10d {base_avg[10]:+.2f}%   +20d {base_avg[20]:+.2f}%\n")
    print(f"{'bucket':<10}" + "".join(f"{f'+{h}d avg':>10}{'edge':>8}" for h in HORIZONS) + f"{'N':>7}")
    print("-" * 70)
    for name, _, _ in BUCKETS:
        cells = ""
        nn = 0
        for h in HORIZONS:
            v = by_bucket[name][h]
            nn = max(nn, len(v))
            if v:
                a = float(np.mean(v))
                cells += f"{a:>10.2f}{(a - base_avg[h]) * 100:>6.0f}bp"
            else:
                cells += f"{'—':>10}{'':>8}"
        print(f"{name:<10}{cells}{nn:>7}")
    print("\nPEAD present if beats drift ABOVE base and misses BELOW, monotonically.")


if __name__ == "__main__":
    asyncio.run(main())
