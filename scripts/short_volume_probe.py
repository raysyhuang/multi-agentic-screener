"""Short-volume-ratio probe (untried edge) — real Polygon daily short volume.

An information edge distinct from price: the fraction of daily volume executed
short (FINRA). Two competing theses — high short volume as bearish pressure
(drift down) vs contrarian squeeze fuel (drift up). Let the data decide.

Diagnostic-first: bucket every (ticker, day) by short_volume_ratio and measure
forward 5/10/20-day returns vs the base rate. Look-ahead-safe: forward return is
strictly after the observation day.

Per-ticker short-volume is cached to disk (data/cache/short_volume) so re-runs
don't re-pull.

Usage:
  python scripts/short_volume_probe.py --cache-file <daily_parquet>
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.polygon_client import PolygonClient

CACHE = Path("data/cache/short_volume")
HORIZONS = [5, 10, 20]
# short_volume_ratio buckets (percent of daily volume executed short).
BUCKETS = [("<35", 0, 35), ("35-45", 35, 45), ("45-50", 45, 50),
           ("50-55", 50, 55), ("55-65", 55, 65), (">=65", 65, 1000)]


async def _short_ratio_series(client, ticker, start, end) -> dict[date, float]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker}.json"
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            return {datetime.strptime(k, "%Y-%m-%d").date(): v for k, v in raw.items()}
        except Exception:
            pass
    try:
        df = await client.get_short_volume(ticker, start, end)
    except Exception:
        return {}
    out = {}
    if not df.empty and "short_volume_ratio" in df.columns:
        for _, r in df.iterrows():
            if pd.notna(r.get("short_volume_ratio")):
                out[r["date"]] = float(r["short_volume_ratio"])
    path.write_text(json.dumps({k.isoformat(): v for k, v in out.items()}))
    return out


def _bucket(x: float):
    for name, lo, hi in BUCKETS:
        if lo <= x < hi:
            return name
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap tickers for a faster run")
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).sort_values("date").reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}
    tickers = [t for t in price_data if t != "SPY"]
    if args.limit:
        tickers = tickers[: args.limit]
    start = min(df["date"].min() for df in price_data.values())
    end = max(df["date"].max() for df in price_data.values())
    print(f"{len(tickers)} tickers, {start}..{end}; pulling short volume (cached)...")

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
        ratios = await _short_ratio_series(client, ticker, start, end)
        for d, ratio in ratios.items():
            i = idx.get(d)
            if i is None:
                continue
            bkt = _bucket(ratio)
            if bkt is None:
                continue
            n_obs += 1
            for h in HORIZONS:
                if i + h < n:
                    by_bucket[bkt][h].append((closes[i + h] - closes[i]) / closes[i] * 100)

    base_avg = {h: (float(np.mean(base[h])) if base[h] else 0.0) for h in HORIZONS}
    print(f"\n{n_obs} observations. Base rate: "
          + "  ".join(f"+{h}d {base_avg[h]:+.2f}%" for h in HORIZONS) + "\n")
    print(f"{'short vol %':<12}" + "".join(f"{f'+{h}d edge':>11}" for h in HORIZONS) + f"{'N':>8}")
    print("-" * 55)
    for name, _, _ in BUCKETS:
        cells, nn = "", 0
        for h in HORIZONS:
            v = by_bucket[name][h]
            nn = max(nn, len(v))
            cells += f"{(np.mean(v) - base_avg[h]) * 100:>9.0f}bp" if v else f"{'—':>11}"
        print(f"{name:<12}{cells}{nn:>8}")
    print("\nEdge = forward return above/below base rate by short-volume bucket.")


if __name__ == "__main__":
    asyncio.run(main())
