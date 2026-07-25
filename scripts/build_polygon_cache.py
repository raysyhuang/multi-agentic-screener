"""Build the Polygon-backed OHLCV universe cache for backtests.

Replaces the free-yfinance data all research ran on with the paid Polygon feed
(adjusted daily bars). Writes a parquet the existing scripts consume via
--cache-file, and prints a quality report (coverage + a spot-check vs yfinance
so we can trust the swap). Minute-bar fill simulation is a separate follow-on.

Usage:
  python -m scripts.build_polygon_cache --years 3
  python -m scripts.build_polygon_cache --limit 30   # quick smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.research.signal_backtest import fetch_ohlcv, fetch_ohlcv_polygon
from src.research.sp500_tickers import SP500_TICKERS

OUT = "outputs/research/ohlcv_polygon_3y.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    tickers = list(SP500_TICKERS)
    if args.limit:
        tickers = tickers[: args.limit]
    if "SPY" not in tickers:
        tickers.append("SPY")

    price = fetch_ohlcv_polygon(tickers, years=args.years, no_cache=False)

    # Write the combined parquet in the scripts' expected shape (_ticker column).
    frames = []
    for t, df in price.items():
        d = df.copy()
        d["_ticker"] = t
        frames.append(d)
    combined = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out, index=False)

    # --- Quality report ---
    bars = {t: len(df) for t, df in price.items()}
    n = len(price)
    med_bars = sorted(bars.values())[n // 2] if n else 0
    thin = [t for t, b in bars.items() if b < med_bars * 0.5]
    print("\n=== Polygon universe cache — quality report ===")
    print(f"tickers: {n}/{len(tickers)} covered   median bars/ticker: {med_bars}")
    print(f"thin (<50% median bars): {len(thin)}{' — ' + ', '.join(thin[:10]) if thin else ''}")
    print(f"written: {args.out} ({Path(args.out).stat().st_size / 1024 / 1024:.1f}MB)")

    # Spot-check vs yfinance on 3 liquid names (last common close should match).
    print("\nspot-check vs yfinance (last close):")
    yf = fetch_ohlcv(["AAPL", "MSFT", "NVDA"], years=args.years, no_cache=False)
    for t in ["AAPL", "MSFT", "NVDA"]:
        p = price.get(t)
        y = yf.get(t)
        if p is not None and y is not None and len(p) and len(y):
            pc, yc = p.iloc[-1]["close"], y.iloc[-1]["close"]
            diff = abs(pc - yc) / yc * 100 if yc else 0
            flag = "OK" if diff < 0.5 else "DIFF"
            print(f"  {t:6s} polygon {pc:8.2f}  yf {yc:8.2f}  ({diff:.2f}% {flag})")


if __name__ == "__main__":
    main()
