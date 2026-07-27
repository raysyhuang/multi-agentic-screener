"""Insider-activity IC study — does insider buying predict forward returns?

WHY: `score_insider_activity` runs on every pipeline run (main.py:884) and
`get_insider_trading` is fetched for up to 150 tickers/run, but the ONLY consumer
is `catalyst.py`, which is DISABLED (main.py:66 "unproven, sparse data"). So the
signal costs ~20% of the 750/day FMP budget and reaches no scorer. This decides
whether to WIRE it (as a gate/filter) or DROP it.

POINT-IN-TIME SAFETY (the trap this study must avoid):
  * The live `score_insider_activity` keys off `transactionDate` and a
    `date.today()` cutoff — fine live (FMP only returns disclosed filings), but
    look-ahead in a backtest: insiders file Form 4 up to 2 business days AFTER the
    trade, so a transactionDate-keyed window can "see" undisclosed trades.
  * This study therefore keys strictly on **filingDate** (public disclosure) and
    re-implements the window as-of each observation date. Deliberately NOT calling
    the live function, which is not as-of aware.

Design: on a sampled date grid, bucket every (ticker, date) by trailing-90d insider
net ratio (and by cluster-buy count) using only filings on/before that date, then
measure forward 5/10/20-day returns vs the unconditional base rate.

Usage:
  python scripts/insider_ic_study.py --cache-file outputs/research/ohlcv_polygon_3y.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fmp_client import FMPClient

CACHE = Path("data/cache/insider")
HORIZONS = [5, 10, 20]
LOOKBACK_DAYS = 90
# Sample every Nth trading day — insider windows are slow-moving (90d), so daily
# sampling would just autocorrelate the same window and inflate significance.
SAMPLE_EVERY = 10

NET_BUCKETS = [
    ("all sells (-1)", -1.01, -0.99),
    ("mostly sells", -0.99, -0.34),
    ("mixed", -0.34, 0.34),
    ("mostly buys", 0.34, 0.99),
    ("all buys (+1)", 0.99, 1.01),
]


def _is_buy(t: str) -> bool:
    return "P" in t or "BUY" in t or "PURCHASE" in t


def _is_sell(t: str) -> bool:
    return "S" in t or "SELL" in t or "SALE" in t


async def _filings(client: FMPClient, ticker: str) -> list[tuple[date, bool, str]]:
    """(filing_date, is_buy, insider_name) — cached per ticker. filingDate ONLY."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker}.json"
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            return [(datetime.strptime(d, "%Y-%m-%d").date(), b, n) for d, b, n in raw]
        except Exception:
            pass
    try:
        rows = await client.get_insider_trading(ticker)
    except Exception:
        return []
    out: list[tuple[date, bool, str]] = []
    for r in rows or []:
        fd = r.get("filingDate") or ""
        if not fd:
            continue  # no disclosure date -> unusable point-in-time
        try:
            d = datetime.strptime(str(fd)[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        ttype = str(r.get("transactionType") or r.get("acquistionOrDisposition") or "").upper()
        name = str(r.get("reportingName") or r.get("insiderName") or "")
        if _is_buy(ttype):
            out.append((d, True, name))
        elif _is_sell(ttype):
            out.append((d, False, name))
    path.write_text(json.dumps([[d.isoformat(), b, n] for d, b, n in out]))
    return out


def _window_stats(filings, asof: date):
    """Trailing-LOOKBACK_DAYS insider stats using only filings disclosed by `asof`."""
    lo = asof - timedelta(days=LOOKBACK_DAYS)
    buys = [f for f in filings if lo <= f[0] <= asof and f[1]]
    sells = [f for f in filings if lo <= f[0] <= asof and not f[1]]
    total = len(buys) + len(sells)
    if total == 0:
        return None
    net = (len(buys) - len(sells)) / total
    distinct_buyers = len({n for _, _, n in buys if n})
    return {"net": net, "n_buys": len(buys), "distinct_buyers": distinct_buyers}


def _fmt(vals, base) -> str:
    if not vals:
        return f"{'—':>11}"
    return f"{(np.mean(vals) - base) * 100:>9.0f}bp"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default="outputs/research/ohlcv_polygon_3y.parquet")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).sort_values("date").reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}
    tickers = [t for t in price_data if t != "SPY"]
    if args.limit:
        tickers = tickers[: args.limit]
    print(f"{len(tickers)} tickers; pulling insider filings (cached)...")

    client = FMPClient()
    by_net = {b[0]: {h: [] for h in HORIZONS} for b in NET_BUCKETS}
    cluster = {h: [] for h in HORIZONS}      # >=3 distinct insiders buying
    single = {h: [] for h in HORIZONS}       # exactly 1 buyer, no sells
    base = {h: [] for h in HORIZONS}
    n_obs = 0
    no_data = 0

    for ti, ticker in enumerate(tickers, 1):
        if ti % 100 == 0:
            print(f"  ...{ti}/{len(tickers)}")
        df = price_data[ticker]
        closes = df["close"].to_numpy(dtype=float)
        days = df["date"].tolist()
        n = len(df)
        for h in HORIZONS:
            if n > h:
                base[h].extend(((closes[h:] - closes[:-h]) / closes[:-h] * 100).tolist())

        filings = await _filings(client, ticker)
        if not filings:
            no_data += 1
            continue

        for i in range(0, n, SAMPLE_EVERY):
            d = days[i]
            d = d.date() if hasattr(d, "date") else d
            st = _window_stats(filings, d)
            if st is None:
                continue
            n_obs += 1
            fwd = {}
            for h in HORIZONS:
                if i + h < n:
                    fwd[h] = (closes[i + h] - closes[i]) / closes[i] * 100
            for name, lo, hi in NET_BUCKETS:
                if lo <= st["net"] < hi:
                    for h, v in fwd.items():
                        by_net[name][h].append(v)
                    break
            if st["distinct_buyers"] >= 3:
                for h, v in fwd.items():
                    cluster[h].append(v)
            elif st["distinct_buyers"] == 1 and st["n_buys"] >= 1:
                for h, v in fwd.items():
                    single[h].append(v)

    base_avg = {h: (float(np.mean(base[h])) if base[h] else 0.0) for h in HORIZONS}
    print(f"\n{n_obs} point-in-time observations ({no_data} tickers had no usable filings).")
    print("Base rate: " + "  ".join(f"+{h}d {base_avg[h]:+.2f}%" for h in HORIZONS) + "\n")

    hdr = f"{'insider net ratio':<20}" + "".join(f"{f'+{h}d edge':>11}" for h in HORIZONS) + f"{'N':>8}"
    print(hdr); print("-" * len(hdr))
    for name, _, _ in NET_BUCKETS:
        cells, nn = "", 0
        for h in HORIZONS:
            v = by_net[name][h]
            nn = max(nn, len(v))
            cells += _fmt(v, base_avg[h])
        print(f"{name:<20}{cells}{nn:>8}")

    print()
    hdr2 = f"{'cluster signal':<20}" + "".join(f"{f'+{h}d edge':>11}" for h in HORIZONS) + f"{'N':>8}"
    print(hdr2); print("-" * len(hdr2))
    for label, dd in ((">=3 distinct buyers", cluster), ("single buyer", single)):
        cells, nn = "", 0
        for h in HORIZONS:
            nn = max(nn, len(dd[h]))
            cells += _fmt(dd[h], base_avg[h])
        print(f"{label:<20}{cells}{nn:>8}")

    print("\nEdge = forward return vs unconditional base rate, in basis points.")
    print("DECISION: a flat/negative table => DROP the insider fetch (reclaims ~20% of")
    print("the FMP daily budget, zero pick impact since catalyst is disabled).")


if __name__ == "__main__":
    asyncio.run(main())
