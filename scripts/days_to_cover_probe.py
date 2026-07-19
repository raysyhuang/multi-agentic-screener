"""Days-to-cover (short-interest squeeze) probe — untried edge #2.

Distinct from daily short volume: short INTEREST is the outstanding short position
(FINRA, bi-monthly); days_to_cover = short_interest / avg_daily_volume. High
days-to-cover = crowded short = classic squeeze fuel. Thesis: crowded shorts drift
UP (squeeze) or DOWN (informed shorts). Let the data decide.

Diagnostic: from each settlement date, bucket by days_to_cover and measure forward
5/10/20-day return vs base rate. Look-ahead-safe. Cached per ticker.

Usage: python scripts/days_to_cover_probe.py --cache-file <daily_parquet>
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.polygon_client import PolygonClient

CACHE = Path("data/cache/short_interest")
HORIZONS = [5, 10, 20]
PUB_LAG_DAYS = 8  # FINRA publishes short interest ~8 business days after settlement
BUCKETS = [("<1", 0, 1), ("1-2", 1, 2), ("2-3", 2, 3), ("3-5", 3, 5),
           ("5-8", 5, 8), (">=8", 8, 1e9)]


async def _dtc_series(client, ticker) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker}.json"
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            return {datetime.strptime(k, "%Y-%m-%d").date(): v for k, v in raw.items()}
        except Exception:
            pass
    try:
        df = await client.get_short_interest(ticker)
    except Exception:
        return {}
    out = {}
    if not df.empty and "days_to_cover" in df.columns:
        for _, r in df.iterrows():
            if pd.notna(r.get("days_to_cover")):
                out[r["settlement_date"]] = float(r["days_to_cover"])
    path.write_text(json.dumps({k.isoformat(): v for k, v in out.items()}))
    return out


def _bucket(x):
    for name, lo, hi in BUCKETS:
        if lo <= x < hi:
            return name
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).sort_values("date").reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}
    tickers = [t for t in price_data if t != "SPY"]
    if args.limit:
        tickers = tickers[: args.limit]
    print(f"{len(tickers)} tickers; pulling short interest (cached)...")

    client = PolygonClient()
    by_bucket = {b[0]: {h: [] for h in HORIZONS} for b in BUCKETS}
    base = {h: [] for h in HORIZONS}
    n_obs = 0
    for ticker in tickers:
        df = price_data[ticker]
        closes = df["close"].to_numpy(dtype=float)
        days = df["date"].tolist()
        idx = {d: i for i, d in enumerate(days)}
        n = len(df)
        for h in HORIZONS:
            if n > h:
                base[h].extend(((closes[h:] - closes[:-h]) / closes[:-h] * 100).tolist())
        for sdate, dtc in (await _dtc_series(client, ticker)).items():
            # FINRA publishes short interest ~8 business days AFTER settlement, so
            # a point-in-time entry is the first trading day on/after settlement,
            # plus the publication lag. Entering at settlement itself is look-ahead.
            i0 = next((j for j, d in enumerate(days) if d >= sdate), None)
            if i0 is None:
                continue
            i = i0 + PUB_LAG_DAYS
            if i >= n:
                continue
            bkt = _bucket(dtc)
            if bkt is None:
                continue
            n_obs += 1
            for h in HORIZONS:
                if i + h < n:
                    by_bucket[bkt][h].append((closes[i + h] - closes[i]) / closes[i] * 100)

    base_avg = {h: (float(np.mean(base[h])) if base[h] else 0.0) for h in HORIZONS}
    print(f"\n{n_obs} settlement observations. Base rate: "
          + "  ".join(f"+{h}d {base_avg[h]:+.2f}%" for h in HORIZONS) + "\n")
    print(f"{'days-to-cover':<14}" + "".join(f"{f'+{h}d edge':>11}" for h in HORIZONS) + f"{'N':>8}")
    print("-" * 57)
    for name, _, _ in BUCKETS:
        cells, nn = "", 0
        for h in HORIZONS:
            v = by_bucket[name][h]
            nn = max(nn, len(v))
            cells += f"{(np.mean(v) - base_avg[h]) * 100:>9.0f}bp" if v else f"{'—':>11}"
        print(f"{name:<14}{cells}{nn:>8}")
    print("\nEdge = forward return vs base by days-to-cover (crowded-short) bucket.")


if __name__ == "__main__":
    asyncio.run(main())
