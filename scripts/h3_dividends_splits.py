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

PRE-DATA AMENDMENTS (committed before any fetch succeeded; the event
definitions and G1 criteria above are unchanged):
  1. Polygon's next_url drops the date filter on /v3/reference/dividends, so no
     fetch follows a cursor.
  2. Some dividend rows have no declaration date, and prior-dividend history
     must still see them (an unseen prior dividend would fake an "initiation"),
     so history is keyed on declaration date OR ex-date.
  3. Whole-market date-window queries are unusable for dividends: ex-date days
     are flooded by thousands of mutual-fund share classes (1,000+ F-tickers on
     2022-03-09 alone). Dividends are therefore fetched PER UNIVERSE TICKER,
     full history, one page each — complete for exactly the names studied.
     Splits keep the date-window fetch (small volumes).

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


_SHARDS = [(None, "A")] + [(chr(c), chr(c + 1)) for c in range(ord("A"), ord("Z"))] + [("Z", None)]
PAGE = 1000


async def _fetch_all(path: str, date_field: str, start: date, end: date, step_days: int = 1) -> list[dict]:
    """Fetch a Polygon reference endpoint WITHOUT following `next_url`.

    Observed 2026-09-19: on /v3/reference/dividends the cursor in `next_url`
    does NOT carry the date filter — page two returned rows with no
    declaration date at all, and a month-long pull ran away past 50 pages. So
    each query covers a window small enough to fit one page; a window that
    comes back FULL (exactly `PAGE` rows, i.e. possibly truncated) is re-split
    by leading ticker character, and a full shard is an error rather than a
    silent loss. Every returned row must carry a date inside its window."""
    client = PolygonClient()
    out: list[dict] = []
    async with httpx.AsyncClient(timeout=60) as cl:

        async def _one(lo: date, hi: date, gte: str | None = None, lt: str | None = None) -> list[dict]:
            q = {f"{date_field}.gte": str(lo), f"{date_field}.lt": str(hi), "limit": PAGE}
            if gte:
                q["ticker.gte"] = gte
            if lt:
                q["ticker.lt"] = lt
            resp = await _request_with_backoff(cl, BASE_URL + path, client._params(**q))
            rows = resp.json().get("results") or []
            bad = [r for r in rows if not (str(lo) <= str(r.get(date_field) or "")[:10] < str(hi))]
            if bad:
                raise SystemExit(f"refusing: {len(bad)} rows outside [{lo},{hi}) on {path} — filter ignored")
            return rows

        d = start
        while d <= end:
            hi = min(d + timedelta(days=step_days), end + timedelta(days=1))
            rows = await _one(d, hi)
            if len(rows) >= PAGE:
                rows = []
                for gte, lt in _SHARDS:
                    part = await _one(d, hi, gte, lt)
                    if len(part) >= PAGE:
                        raise SystemExit(f"refusing: shard {gte}-{lt} on {d} is full ({PAGE}) — would truncate")
                    rows.extend(part)
            out.extend(rows)
            d = hi
    ids = [r.get("id") for r in out]
    if len(ids) != len(set(ids)):
        raise SystemExit("refusing: duplicate ids across windows")
    return out


async def _dividends_by_ticker(tickers: list[str], concurrency: int = 8) -> tuple[list[dict], dict]:
    """Every dividend Polygon has for each universe ticker — ONE query per
    ticker, full history, no date filter and no cursor. Complete for exactly
    the names the study uses, and independent of both failure modes seen on
    the whole-market query (a cursor that drops the date filter; ex-date days
    flooded by thousands of mutual-fund share classes). A ticker whose history
    does not fit one page is refused rather than truncated."""
    client = PolygonClient()
    sem = asyncio.Semaphore(concurrency)
    out: list[dict] = []
    failures: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=60) as cl:
        async def _one(t: str) -> None:
            async with sem:
                try:
                    resp = await _request_with_backoff(
                        cl, BASE_URL + "/v3/reference/dividends",
                        client._params(ticker=t.replace("-", "."), limit=PAGE))
                    j = resp.json()
                    rows = j.get("results") or []
                    if j.get("next_url") or len(rows) >= PAGE:
                        failures[t] = f"history exceeds one page ({len(rows)})"
                        return
                    out.extend(rows)
                except Exception as e:  # noqa: BLE001 — recorded, not swallowed
                    failures[t] = f"{type(e).__name__}: {e}"[:160]
        await asyncio.gather(*(_one(t) for t in tickers))
    return out, failures


def stage_fetch(prices_path: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    end = date.today()
    comb = pd.read_parquet(prices_path, columns=["_ticker"])
    tickers = sorted(set(comb["_ticker"]) - {"SPY"})
    divs, div_failures = asyncio.run(_dividends_by_ticker(tickers))
    splits = asyncio.run(_fetch_all("/v3/reference/splits", "execution_date", HISTORY_START, end, step_days=7))
    (CACHE / "dividends.json").write_text(json.dumps(divs))
    (CACHE / "splits.json").write_text(json.dumps(splits))
    man = {"fetched_at": datetime.now(timezone.utc).isoformat(), "window": [str(HISTORY_START), str(end)],
           "dividend_method": "per universe ticker, full history, one page each",
           "dividend_tickers_requested": len(tickers), "dividend_failures": div_failures,
           "dividends": len(divs), "splits": len(splits),
           "sha256": {f: hashlib.sha256((CACHE / f).read_bytes()).hexdigest()
                      for f in ("dividends.json", "splits.json")}}
    (CACHE / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    return man


def _split_factor_after(splits_by_t: dict[str, list[tuple[str, float]]], t: str, day: str) -> float:
    """prod(split_to/split_from) over the ticker's splits executed AFTER `day`:
    divides a per-share amount paid on `day`'s share basis into today's basis."""
    f = 1.0
    for d, r in splits_by_t.get(t, []):
        if d > day:
            f *= r
    return f


def dividend_events(divs: list[dict], tickers: set[str], splits: list[dict] | None = None) -> list[dict]:
    """INITIATION and INCREASE events per the registered definitions. Only
    declarations on/before each event are consulted.

    Amended after the Codex review of PR #123 (definitions unchanged):
      * payments are compared on ONE share basis — each cash amount is divided
        by the split factor of splits executed after its ex-date, so NVDA's
        post-split $0.01 (a 150% raise) is not read as a cut, and a reverse
        split cannot manufacture a >=25% "increase";
      * rows sharing ticker, declaration/ex-date and frequency are one payment
        (their components summed), and ties sort on (date, ex-date, amount),
        so the classification no longer depends on the input row order."""
    splits_by_t: dict[str, list[tuple[str, float]]] = {}
    for sp in es.clean_splits(splits or [], tickers):   # duplicates once, conflicts dropped
        splits_by_t.setdefault(sp["ticker"], []).append(
            (sp["execution_date"], sp["split_to"] / sp["split_from"]))
    rows = [d for d in divs if d.get("dividend_type") == "CD" and (d.get("currency") or "USD") == "USD"
            and (d.get("declaration_date") or d.get("ex_dividend_date")) and d.get("cash_amount")
            and d.get("ticker")]
    agg: dict[tuple, dict] = {}
    for d in rows:
        t = str(d["ticker"]).replace(".", "-").upper()
        if t not in tickers:
            continue
        key = d.get("declaration_date") or d["ex_dividend_date"]
        ex = d.get("ex_dividend_date") or key
        amt = float(d["cash_amount"]) / _split_factor_after(splits_by_t, t, ex)
        g = (t, key, ex, d.get("frequency"))
        if g in agg:
            agg[g]["cash_amount"] += amt
        else:
            agg[g] = {"ticker": t, "_key": key, "ex_dividend_date": ex, "frequency": d.get("frequency"),
                      "declaration_date": d.get("declaration_date"), "cash_amount": amt}
    by_t: dict[str, list[dict]] = {}
    for d in agg.values():
        by_t.setdefault(d["ticker"], []).append(d)
    events = []
    for t, ds in by_t.items():
        ds.sort(key=lambda d: (d["_key"], d.get("ex_dividend_date") or "", d["cash_amount"],
                               -1 if d.get("frequency") is None else d["frequency"]))
        seen_decl: set[str] = set()
        for k, d in enumerate(ds):
            decl = d.get("declaration_date")
            if not decl or decl in seen_decl:   # no announcement date, or several declared together
                continue
            seen_decl.add(decl)
            dd = date.fromisoformat(decl)
            prior = [p for p in ds[:k] if p["_key"] < decl
                     and (dd - date.fromisoformat(p["_key"])).days <= LOOKBACK_DAYS]
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
    """One event per (ticker, execution date), through the same cleaner as the
    price screen: identical duplicates count once, conflicting same-day rows
    are dropped rather than yielding two contradictory events."""
    return [{"ticker": s["ticker"], "event_date": s["execution_date"],
             "kind": "forward" if s["split_to"] > s["split_from"] else "reverse"}
            for s in es.clean_splits(splits, tickers)]


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


def stage_study(prices_path: str, json_out: str | None, raw_price_screen: bool = False,
                delist_return: float | None = None) -> dict:
    raw = Path(prices_path).read_bytes()
    comb = pd.read_parquet(prices_path)
    prices = {t: g.drop(columns=["_ticker"]) for t, g in comb.groupby("_ticker") if t != "SPY"}
    divs = json.loads((CACHE / "dividends.json").read_text())
    splits = json.loads((CACHE / "splits.json").read_text())
    panel = es.build_panel(prices, splits=splits if raw_price_screen else None)
    tickers = set(prices)
    man = json.loads((CACHE / "manifest.json").read_text())

    dev = place(dividend_events(divs, tickers, splits), panel, offset=1)   # 2nd session on/after declaration
    sev = place(split_events(splits, tickers), panel, offset=0)    # 1st session on/after execution
    out: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "raw_price_screen": raw_price_screen,
                 "delist_return": delist_return,
                 "prices_sha256": hashlib.sha256(raw).hexdigest(), "corp_actions": man, "cells": {}}
    for fam, evs in (("dividend", dev), ("split", sev)):
        ex = es.event_excess(panel, evs, HORIZONS, delist_return=delist_return)
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
    ap.add_argument("--delist-return", type=float, default=None,
                    help="impute this %% return for names that DISAPPEAR inside the window "
                         "(events and base rate); leaves end-of-dataset truncation as NaN")
    ap.add_argument("--raw-price-screen", action="store_true")
    args = ap.parse_args()
    if args.stage == "fetch":
        stage_fetch(args.prices)
    else:
        stage_study(args.prices, args.json_out, args.raw_price_screen, args.delist_return)


if __name__ == "__main__":
    main()
