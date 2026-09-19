"""Build the wide, survivorship-free event-study universe (research funnel, Phase 1a).

Why this exists: every PEAD backtest so far ran on the CURRENT S&P 500 replayed
backwards (503 large caps, survivorship-biased), while live PEAD fires across
<=1000 names down to $300M market cap. Live has been trading a population no
backtest sampled. The halted PIT-universe project is not needed to fix that for
EVENT studies — an event study only has to know whether a name was liquid on
the event's own date, which a name's own price history answers.

Three stages, each cached and resumable:

  candidates  Polygon whole-market daily bars on the first session of each
              quarter in the window, intersected with Polygon's POINT-IN-TIME
              reference listing (`date=`) for types CS/ADRC on that same date.
              A name later delisted is kept, because it is selected on the date
              it was alive. The screen here is deliberately LOOSER than the
              study's (single-day $ volume); the precise no-lookahead screen is
              `src.research.event_study.build_panel`.
  earnings    FMP `/stable/earnings?symbol=` per candidate (deep history, covers
              small caps and delisted names), written to the SAME disk cache
              `src.data.earnings_cache` reads, so existing PEAD scripts see it.
              FMP Starter rejects historical `from` on the whole-market calendar
              (402 beyond ~12 months), so per-ticker is the only route.
  prices      Polygon adjusted daily bars for every candidate + SPY.

Usage:
  python scripts/build_event_universe.py --stage all
  python scripts/build_event_universe.py --stage candidates --years 3.2
  python scripts/build_event_universe.py --stage earnings --fmp-rate 4
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.data.earnings_cache import CACHE_DIR as EARNINGS_CACHE_DIR  # noqa: E402
from src.data.fmp_client import FMPClient  # noqa: E402
from src.data.polygon_client import PolygonClient  # noqa: E402
from src.research.signal_backtest import fetch_ohlcv_polygon, get_last_ohlcv_provenance  # noqa: E402

OUT_DIR = REPO / "outputs" / "research"
CANDIDATES_JSON = OUT_DIR / "event_universe_candidates.json"
EARNINGS_MANIFEST = OUT_DIR / "event_universe_earnings_manifest.json"
PRICES_PARQUET = OUT_DIR / "ohlcv_polygon_wide_3y.parquet"
TYPES = ("CS", "ADRC")


def quarter_starts(start: date, end: date) -> list[date]:
    """First calendar day of every quarter in [start, end], plus `start` itself."""
    out = {start}
    y, q = start.year, (start.month - 1) // 3
    while True:
        d = date(y, 3 * q + 1, 1)
        if d > end:
            break
        if d >= start:
            out.add(d)
        q += 1
        if q == 4:
            y, q = y + 1, 0
    return sorted(out)


def to_dash(sym: str) -> str:
    """Polygon share classes are dot-form (BRK.B); caches and FMP use dash (BRK-B)."""
    return sym.replace(".", "-").upper()


def liquid_on(rows: list[dict], min_price: float, min_dollar_vol: float) -> set[str]:
    out = set()
    for r in rows:
        c, v = r.get("c"), r.get("v")
        px = r.get("vw") or c
        if c is None or v is None or px is None:
            continue
        if c >= min_price and px * v >= min_dollar_vol:
            out.add(r["T"])
    return out


# One query per leading character, run sequentially. Polygon's cursor pagination
# on /v3/reference/tickers silently TRUNCATES (observed 2026-09-19: 1,507 and
# 3,900 names returned for a listing of ~5,500, no error) — the live aggregator
# shards alphabetically for the same reason. Per-letter keeps every query to a
# few hundred rows, well under the page size, so there is nothing to truncate.
_SHARDS = [(None, "A")] + [(chr(c), chr(c + 1)) for c in range(ord("A"), ord("Z"))] + [("Z", None)]


async def typed_as_of(client: PolygonClient, d: date) -> set[str]:
    """Tickers of type CS/ADRC that were ACTIVE ON date `d` (point-in-time)."""
    out: set[str] = set()
    for ty in TYPES:
        for gte, lt in _SHARDS:
            ref = await client.get_all_tickers(ticker_type=ty, as_of=d, max_pages=5,
                                               ticker_gte=gte, ticker_lt=lt)
            out |= {r["ticker"] for r in ref if r.get("ticker")}
    return out


async def stage_candidates(years: float, min_price: float, min_dollar_vol: float) -> dict:
    client = PolygonClient()
    end = date.today()
    start = end - timedelta(days=int(years * 365.25))
    per_date, union = [], set()
    for qd in quarter_starts(start, end):
        d, rows = qd, []
        for _ in range(6):                      # step to the first session on/after qd
            rows = await client.get_grouped_daily(d)
            if rows:
                break
            d += timedelta(days=1)
        if not rows:
            per_date.append({"quarter_start": str(qd), "session": None, "note": "no session found"})
            continue
        liquid = liquid_on(rows, min_price, min_dollar_vol)
        typed = await typed_as_of(client, d)
        both = liquid & typed
        union |= both
        per_date.append({"quarter_start": str(qd), "session": str(d), "n_bars": len(rows),
                         "n_liquid": len(liquid), "n_typed": len(typed), "n_both": len(both)})
        print(f"  {d}: bars={len(rows)} liquid={len(liquid)} typed={len(typed)} both={len(both)} "
              f"union={len(union)}", flush=True)
    tickers = sorted({to_dash(t) for t in union})
    # Completeness tripwire: nearly every current S&P 500 name is a liquid CS and
    # must be in the union. A low figure means the listing truncated again.
    from src.research.sp500_tickers import SP500_TICKERS
    sp = {to_dash(t) for t in SP500_TICKERS}
    sp_cov = len(sp & set(tickers)) / len(sp)
    print(f"  S&P 500 coverage of the candidate union: {sp_cov:.1%}")
    if sp_cov < 0.95:
        raise SystemExit(f"refusing: only {sp_cov:.1%} of the S&P 500 is in the candidate union — "
                         "the reference listing is incomplete")
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": [str(start), str(end)], "types": list(TYPES),
        "screen": {"min_price": min_price, "min_single_day_dollar_vol": min_dollar_vol,
                   "note": "single-day screen on sampled sessions; looser than the study screen by design"},
        "per_date": per_date, "sp500_coverage": round(sp_cov, 4),
        "n_tickers": len(tickers), "tickers": tickers,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_JSON.write_text(json.dumps(doc, indent=1))
    print(f"candidates: {len(tickers)} tickers -> {CANDIDATES_JSON.relative_to(REPO)}")
    return doc


async def stage_earnings(tickers: list[str], rate_per_sec: float, max_tickers: int | None) -> dict:
    """Fill the shared earnings cache. A fetch ERROR is recorded and NOT cached
    (so it retries); an empty result IS cached as [] — 'FMP has no earnings for
    this symbol' is a fact about the symbol, and it is how funds that slipped
    through the type filter identify themselves."""
    EARNINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if not (EARNINGS_CACHE_DIR / f"{t}.json").exists()]
    if max_tickers:
        todo = todo[:max_tickers]
    print(f"earnings: {len(tickers) - len(todo)} cached, {len(todo)} to fetch "
          f"at <= {rate_per_sec:g}/s", flush=True)
    failures: dict[str, str] = {}
    gap = 1.0 / rate_per_sec
    for i, t in enumerate(todo, 1):
        t0 = asyncio.get_event_loop().time()
        try:
            rows = await FMPClient().get_earnings_surprise(t)   # fresh client: a 402 on one
            (EARNINGS_CACHE_DIR / f"{t}.json").write_text(      # symbol must not disable the rest
                json.dumps(rows if isinstance(rows, list) else []))
        except Exception as e:  # noqa: BLE001 — recorded, not swallowed
            failures[t] = f"{type(e).__name__}: {e}"[:160]
        if i % 200 == 0:
            print(f"  {i}/{len(todo)} fetched, {len(failures)} failures", flush=True)
        await asyncio.sleep(max(0.0, gap - (asyncio.get_event_loop().time() - t0)))

    n_rows = n_pairs = n_empty = 0
    for t in tickers:
        p = EARNINGS_CACHE_DIR / f"{t}.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        n_rows += len(rows)
        n_empty += not rows
        n_pairs += sum(1 for r in rows
                       if r.get("epsActual") is not None and r.get("epsEstimated") is not None)
    doc = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "requested": len(tickers), "cached": sum((EARNINGS_CACHE_DIR / f"{t}.json").exists() for t in tickers),
           "empty": n_empty, "rows": n_rows, "eps_pairs": n_pairs, "failures": failures}
    EARNINGS_MANIFEST.write_text(json.dumps(doc, indent=1))
    print(f"earnings: cached={doc['cached']} empty={n_empty} eps_pairs={n_pairs} "
          f"failures={len(failures)} -> {EARNINGS_MANIFEST.relative_to(REPO)}")
    return doc


def stage_prices(tickers: list[str], years: float) -> dict:
    want = sorted(set(tickers) | {"SPY"})
    data = fetch_ohlcv_polygon(want, years=years)
    prov = get_last_ohlcv_provenance()
    frames = []
    for t, df in data.items():
        d = df.copy()
        d["_ticker"] = t
        frames.append(d)
    combined = pd.concat(frames, ignore_index=True)
    PRICES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PRICES_PARQUET, index=False)
    sha = hashlib.sha256(PRICES_PARQUET.read_bytes()).hexdigest()
    side = {"provenance": prov, "rows": int(len(combined)), "tickers": int(combined["_ticker"].nunique()),
            "span": [str(combined["date"].min())[:10], str(combined["date"].max())[:10]],
            "sha256": sha, "generated_at": datetime.now(timezone.utc).isoformat()}
    PRICES_PARQUET.with_suffix(".provenance.json").write_text(json.dumps(side, indent=1, default=str))
    print(f"prices: {side['tickers']} tickers, {side['rows']} rows, {side['span']} sha {sha[:12]} "
          f"-> {PRICES_PARQUET.relative_to(REPO)}")
    return side


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["candidates", "earnings", "prices", "all"], default="all")
    ap.add_argument("--years", type=float, default=3.2)
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--min-dollar-vol", type=float, default=2_000_000.0)
    ap.add_argument("--fmp-rate", type=float, default=4.0, help="FMP calls per second (Starter = 300/min)")
    ap.add_argument("--max-tickers", type=int, default=None, help="smoke-run cap for the earnings stage")
    args = ap.parse_args()

    if args.stage in ("candidates", "all"):
        cand = asyncio.run(stage_candidates(args.years, args.min_price, args.min_dollar_vol))
    else:
        cand = json.loads(CANDIDATES_JSON.read_text())
    tickers = cand["tickers"]
    if args.stage in ("earnings", "all"):
        asyncio.run(stage_earnings(tickers, args.fmp_rate, args.max_tickers))
    if args.stage in ("prices", "all"):
        stage_prices(tickers, args.years)


if __name__ == "__main__":
    main()
