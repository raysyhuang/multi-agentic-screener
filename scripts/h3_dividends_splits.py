"""H3 (Stage-0 / gate G1): drift after dividend news and after forward stock splits.

Mechanism: a first dividend, or a large increase, is a costly signal of
management's confidence in future cash flow; markets are documented to under-
react to it (Michaely, Thaler & Womack 1995). A forward split is the cheaper
cousin of the same signal, and post-split drift is documented too (Ikenberry,
Rankine & Stice 1996), though weaker and more contested. Who is on the other
side: holders who treat the announcement as a one-day event.

PRE-REGISTERED. Committed BEFORE any dividend or split data was fetched (the
`fetch` stage writes data/cache/corp_actions/*; none exists at commit time).
Verify with `git log --format='%h %ad' --date=iso -- scripts/h3_dividends_splits.py`
against `fetched_at` in data/cache/corp_actions/manifest.json and `generated_at`
in the output.

  Universe       the wide parquet (outputs/research/ohlcv_polygon_wide_3y.parquet),
                 liquid as of the prior close (price >= $5, 20d mean $ volume >= $2M).
  H3a event      Polygon /v3/reference/dividends, dividend_type "CD", USD:
                   INITIATION = no CD declared by the ticker in the previous 400
                     calendar days (history fetched from 2022-01-01, so the
                     lookback exists for every in-window event);
                   INCREASE = cash_amount >= 1.25 x the ticker's previous CD with
                     the same `frequency`, declared within the previous 400 days.
                 Event date = declaration_date. Polygon does not say whether the
                 declaration came before or after the close, so ENTRY is the
                 SECOND session on/after the declaration date (look-ahead-free
                 for any announcement time).
  H3b event      Polygon /v3/reference/splits, FORWARD splits only
                 (split_to > split_from). The execution date is known in advance
                 (PIT-safe); ENTRY = the first session on/after the execution date.
                 Announcement-date drift is NOT testable (no announcement date).
  Metric         mean excess_20 vs the same-day, same-liquidity-tercile base rate.
  G1 PASS, evaluated SEPARATELY for H3a (initiations + increases pooled) and
  H3b, needs ALL of:
    (a) mean excess_20 >= +0.50%
    (b) moving-block (block 20) 95% CI lower bound > 0   [stricter than the
        date-cluster CI, which the H1 review showed is too narrow for
        overlapping 20-session windows; the cluster CI is also reported]
    (c) positive mean excess_20 in >= 2 of the calendar years with n >= 30
  Descriptive only (registry variants): initiations and increases separately;
  horizon 60; reverse splits; liquidity terciles.

Usage:
  python scripts/h3_dividends_splits.py --stage fetch
  python scripts/h3_dividends_splits.py --stage study --json-out outputs/research/h3_dividends_splits.json
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

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.polygon_client import BASE_URL, PolygonClient, _request_with_backoff  # noqa: E402
from src.research import event_study as es  # noqa: E402

CACHE = REPO / "data" / "cache" / "corp_actions"
HISTORY_START = date(2022, 1, 1)
LOOKBACK_DAYS = 400
INCREASE_MULT = 1.25
HORIZONS = [20, 60]
PRIMARY_H = 20
G1_MIN_EXCESS = 0.50
G1_MIN_YEAR_N = 30


async def _fetch_all(path: str, date_field: str, start: date, end: date) -> list[dict]:
    """Page through a Polygon reference endpoint one calendar month at a time
    (small windows keep each query well under the page cap; cursor pagination
    has silently truncated large listings before)."""
    client = PolygonClient()
    out: list[dict] = []
    m = date(start.year, start.month, 1)
    async with httpx.AsyncClient(timeout=60) as cl:
        while m <= end:
            nxt = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
            url = BASE_URL + path
            params = client._params(**{f"{date_field}.gte": str(m),
                                       f"{date_field}.lt": str(min(nxt, end + timedelta(days=1))),
                                       "limit": 1000})
            pages = 0
            while url:
                resp = await _request_with_backoff(cl, url, params)
                j = resp.json()
                out.extend(j.get("results") or [])
                url = j.get("next_url")
                params = client._params() if url else {}
                pages += 1
                if pages > 50:
                    raise SystemExit(f"refusing: >50 pages for {path} {m} — pagination runaway")
            m = nxt
    return out


def stage_fetch() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    end = date.today()
    divs = asyncio.run(_fetch_all("/v3/reference/dividends", "declaration_date", HISTORY_START, end))
    splits = asyncio.run(_fetch_all("/v3/reference/splits", "execution_date", HISTORY_START, end))
    (CACHE / "dividends.json").write_text(json.dumps(divs))
    (CACHE / "splits.json").write_text(json.dumps(splits))
    man = {"fetched_at": datetime.now(timezone.utc).isoformat(), "window": [str(HISTORY_START), str(end)],
           "dividends": len(divs), "splits": len(splits),
           "sha256": {f: hashlib.sha256((CACHE / f).read_bytes()).hexdigest()
                      for f in ("dividends.json", "splits.json")}}
    (CACHE / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    return man


def dividend_events(divs: list[dict], tickers: set[str]) -> list[dict]:
    """INITIATION and INCREASE events per the registered definitions. Only
    declarations on/before each event are consulted."""
    rows = [d for d in divs if d.get("dividend_type") == "CD" and (d.get("currency") or "USD") == "USD"
            and d.get("declaration_date") and d.get("cash_amount") and d.get("ticker")]
    by_t: dict[str, list[dict]] = {}
    for d in rows:
        t = str(d["ticker"]).replace(".", "-").upper()
        if t in tickers:
            by_t.setdefault(t, []).append(d)
    events = []
    for t, ds in by_t.items():
        ds.sort(key=lambda d: (d["declaration_date"], d.get("ex_dividend_date") or ""))
        seen_decl: set[str] = set()
        for k, d in enumerate(ds):
            decl = d["declaration_date"]
            if decl in seen_decl:          # several payouts declared together: judge the first
                continue
            seen_decl.add(decl)
            dd = date.fromisoformat(decl)
            prior = [p for p in ds[:k] if p["declaration_date"] < decl
                     and (dd - date.fromisoformat(p["declaration_date"])).days <= LOOKBACK_DAYS]
            if dd - timedelta(days=LOOKBACK_DAYS) < HISTORY_START:
                continue                    # lookback not covered by fetched history
            kind = None
            if not prior:
                kind = "initiation"
            else:
                same = [p for p in prior if p.get("frequency") == d.get("frequency")]
                if same and float(d["cash_amount"]) >= INCREASE_MULT * float(same[-1]["cash_amount"]):
                    kind = "increase"
            if kind:
                events.append({"ticker": t, "event_date": decl, "kind": kind})
    return events


def split_events(splits: list[dict], tickers: set[str]) -> list[dict]:
    out = []
    for s in splits:
        t = str(s.get("ticker") or "").replace(".", "-").upper()
        if t not in tickers or not s.get("execution_date"):
            continue
        fwd = float(s.get("split_to") or 0) > float(s.get("split_from") or 0)
        out.append({"ticker": t, "event_date": s["execution_date"], "kind": "forward" if fwd else "reverse"})
    return out


def place(events: list[dict], panel: es.Panel, offset: int) -> list[dict]:
    """entry = the session `offset` rows after the first session on/after the event date."""
    idx = panel.dates
    out = []
    for ev in events:
        ts = pd.Timestamp(ev["event_date"])
        if ts < idx[0] or ts > idx[-1]:
            continue
        i = int(idx.searchsorted(ts, side="left")) + offset
        if 1 <= i < len(idx):
            out.append({**ev, "entry_date": idx[i]})
    return out


def g1(summary: dict, years: dict) -> dict:
    ok_a = summary.get("n", 0) > 0 and summary["mean_excess"] >= G1_MIN_EXCESS
    ok_b = summary.get("n", 0) > 0 and summary["block_ci"][0] > 0
    judged = {y: v for y, v in years.items() if v["n"] >= G1_MIN_YEAR_N}
    pos = sum(1 for v in judged.values() if v["mean_excess"] > 0)
    return {"a_mean_ge_0.50": ok_a, "b_block_ci_lo_gt_0": ok_b,
            "c_positive_years": f"{pos}/{len(judged)} (n>={G1_MIN_YEAR_N})", "c_pass": pos >= 2,
            "PASS": bool(ok_a and ok_b and pos >= 2)}


def by_year(sub: pd.DataFrame, h: int) -> dict:
    s = sub[sub[f"excess_{h}"].notna()]
    return {int(y): {"n": int(len(g)), "mean_excess": float(g[f"excess_{h}"].mean())}
            for y, g in s.groupby(s["entry_date"].dt.year)}


def stage_study(prices_path: str, json_out: str | None) -> dict:
    raw = Path(prices_path).read_bytes()
    comb = pd.read_parquet(prices_path)
    prices = {t: g.drop(columns=["_ticker"]) for t, g in comb.groupby("_ticker") if t != "SPY"}
    panel = es.build_panel(prices)
    tickers = set(prices)
    divs = json.loads((CACHE / "dividends.json").read_text())
    splits = json.loads((CACHE / "splits.json").read_text())
    man = json.loads((CACHE / "manifest.json").read_text())

    dev = place(dividend_events(divs, tickers), panel, offset=1)   # 2nd session on/after declaration
    sev = place(split_events(splits, tickers), panel, offset=0)    # 1st session on/after execution
    out: dict = {"generated_at": datetime.now(timezone.utc).isoformat(),
                 "prices_sha256": hashlib.sha256(raw).hexdigest(), "corp_actions": man, "cells": {}}
    for fam, evs in (("dividend", dev), ("split", sev)):
        ex = es.event_excess(panel, evs, HORIZONS)
        if ex.empty:
            continue
        ex["entry_date"] = pd.to_datetime(ex["entry_date"])
        n_all = len(ex)
        ex = ex[ex["eligible"]]
        cuts = ({"H3a_pooled": ex, "initiation": ex[ex["kind"] == "initiation"],
                 "increase": ex[ex["kind"] == "increase"]} if fam == "dividend"
                else {"H3b_forward": ex[ex["kind"] == "forward"], "reverse": ex[ex["kind"] == "reverse"]})
        for name, sub in cuts.items():
            cell = {"n_events_placed": n_all if name.startswith("H3") else None,
                    **{f"h{h}": es.summarize_excess(sub, h, calendar=panel.dates) for h in HORIZONS},
                    "by_year_h20": by_year(sub, PRIMARY_H),
                    "by_bucket_h20": {int(b): es.summarize_excess(g, PRIMARY_H) for b, g in sub.groupby("bucket")}}
            out["cells"][name] = cell
            s = cell["h20"]
            if s.get("n"):
                print(f"{name:12s} n={s['n']:5d} dates={s['n_dates']:4d} excess20={s['mean_excess']:+.3f}% "
                      f"cluster[{s['cluster_ci'][0]:+.2f},{s['cluster_ci'][1]:+.2f}] "
                      f"block[{s['block_ci'][0]:+.2f},{s['block_ci'][1]:+.2f}] hit={s['hit']:.1%}")
    for key in ("H3a_pooled", "H3b_forward"):
        if key in out["cells"] and out["cells"][key]["h20"].get("n"):
            out[f"g1_{key}"] = g1(out["cells"][key]["h20"], out["cells"][key]["by_year_h20"])
            print(f"G1 {key}:", json.dumps(out[f"g1_{key}"]))
    if json_out:
        p = Path(json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1, default=str))
        print(f"wrote {p}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["fetch", "study"], required=True)
    ap.add_argument("--prices", default="outputs/research/ohlcv_polygon_wide_3y.parquet")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    if args.stage == "fetch":
        stage_fetch()
    else:
        stage_study(args.prices, args.json_out)


if __name__ == "__main__":
    main()
