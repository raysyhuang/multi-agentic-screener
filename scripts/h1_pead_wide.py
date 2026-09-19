"""H1 (Stage-0 / gate G1): does PEAD hold OUTSIDE the S&P 500?

Mechanism: post-earnings drift is under-reaction, and under-reaction is largest
where attention is thinnest. Every PEAD backtest in this repo ran on the current
S&P 500 replayed backwards (503 large caps, survivorship-biased); live PEAD fires
on <=1000 names down to $300M market cap. The population live trades has never
been measured. Who is on the other side: holders who anchor on the pre-report
price and sell a good report too early; we are paid for waiting 20 sessions.

PRE-REGISTERED (this file is committed BEFORE the wide-universe run; verify with
`git log --format='%h %ad' --date=iso -- scripts/h1_pead_wide.py` against the
`generated_at` stamped in the output JSON):

  Primary cohort   E1-gated beats (EPS surprise >= 10%, revenue surprise >= 2%,
                   report-bar reaction in [+2%, +12%]) on names NOT in the S&P 500
                   list, liquid as of the prior close (price >= $5, 20d mean
                   dollar volume >= $2M).
  Primary metric   mean EXCESS return, entry-bar open -> close 20 sessions later,
                   vs the same-day, same-liquidity-tercile base rate of ALL
                   eligible names (not SPY: small-cap beta must cancel).
  G1 PASS needs ALL of:
    (a) mean excess_20 >= +0.50%
    (b) date-cluster bootstrap 95% CI lower bound > 0
    (c) positive mean excess_20 in >= 2 of the 3 calendar years with n >= 30
  Anything else = REJECTED at G1. No threshold is tuned: the E1 and neglected
  gates are the LIVE ones, unchanged. Each additional cohort cut reported below
  is descriptive and is counted in the registry as a variant.
  Descriptive (not decisional): raw >=10% beats; E1 + neglected (decelerating
  YoY revenue growth); S&P-500 members (reproduction check against the published
  large-cap result); liquidity terciles; horizons 5 and 60.

Timing convention (same as scripts/pead_backtest.py and live): the SIGNAL bar is
the first session on/after the FMP report date, the reaction is that bar's close
over the prior close, and ENTRY is the next session's open. FMP does not say
whether a report was pre-market or after the close; for an after-close report
the signal bar predates the news, so its "reaction" is not the reaction. This
cuts against the E1 gate (it discards real beats), never in its favour.

Usage:
  python scripts/h1_pead_wide.py --json-out outputs/research/h1_pead_wide.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.data.earnings_cache import CACHE_DIR as EARNINGS_CACHE_DIR  # noqa: E402
from src.research import event_study as es  # noqa: E402
from src.research.sp500_tickers import SP500_TICKERS  # noqa: E402
from src.signals.post_earnings_drift import eps_surprise_pct, revenue_growth_acceleration  # noqa: E402

MIN_SURPRISE = 10.0          # live: settings.pead_min_surprise
MIN_REV_SURPRISE = 2.0       # live: settings.pead_min_revenue_surprise
REACTION_BAND = (2.0, 12.0)  # live: settings.pead_reaction_min_pct / max_pct
HORIZONS = [5, 20, 60]
PRIMARY_H = 20
G1_MIN_EXCESS = 0.50
G1_MIN_YEAR_N = 30


def _day(s) -> date | None:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def build_events(panel: es.Panel, tickers: list[str]) -> list[dict]:
    """One event per (ticker, report date) with surprise, revenue surprise,
    report-bar reaction and the point-in-time revenue-growth acceleration.
    Everything attached to an event uses information dated on/before its signal
    bar; the entry is the NEXT session in the panel calendar."""
    idx = panel.dates
    dates = list(idx)
    close = panel.close
    events = []
    for t in tickers:
        p = EARNINGS_CACHE_DIR / f"{t}.json"
        if t not in close.columns or not p.exists():
            continue
        rows = [r for r in json.loads(p.read_text()) if _day(r.get("date")) is not None]
        rows.sort(key=lambda r: r["date"])
        seen: set[date] = set()
        for k, r in enumerate(rows):
            rd = _day(r["date"])
            if rd in seen:
                continue
            seen.add(rd)
            sp = eps_surprise_pct(r.get("epsActual"), r.get("epsEstimated"))
            if sp is None:
                continue
            ts = pd.Timestamp(rd)
            if ts < dates[0] or ts > dates[-1]:
                continue
            i = int(idx.searchsorted(ts, side="left"))   # first session on/after the report date
            if i < 1 or i + 1 >= len(dates):
                continue
            col = close.columns.get_loc(t)
            c_sig, c_prev = close.iat[i, col], close.iat[i - 1, col]
            reaction = (c_sig / c_prev - 1) * 100 if pd.notna(c_sig) and pd.notna(c_prev) and c_prev > 0 else None
            rev = eps_surprise_pct(r.get("revenueActual"), r.get("revenueEstimated"))
            accel = revenue_growth_acceleration(
                (x["date"], x.get("revenueActual")) for x in rows[: k + 1])
            events.append({"ticker": t, "report_date": str(rd), "signal_date": dates[i],
                           "entry_date": dates[i + 1], "surprise": sp, "rev_surprise": rev,
                           "reaction": reaction, "rev_accel": accel})
    return events


def cohorts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw = df[df["surprise"] >= MIN_SURPRISE]
    e1 = raw[(raw["rev_surprise"].fillna(-1e9) >= MIN_REV_SURPRISE)
             & raw["reaction"].between(*REACTION_BAND)]
    neg = e1[e1["rev_accel"].notna() & (e1["rev_accel"] <= 0)]
    return {"raw_beat10": raw, "e1": e1, "e1_neglected": neg}


def by_year(sub: pd.DataFrame, h: int) -> dict:
    out = {}
    col = f"excess_{h}"
    s = sub[sub[col].notna()]
    for y, g in s.groupby(s["entry_date"].dt.year):
        out[int(y)] = {"n": int(len(g)), "mean_excess": float(g[col].mean())}
    return out


def g1_verdict(summary: dict, years: dict) -> dict:
    ok_a = summary.get("n", 0) > 0 and summary["mean_excess"] >= G1_MIN_EXCESS
    ok_b = summary.get("n", 0) > 0 and summary["cluster_ci"][0] > 0
    judged = {y: v for y, v in years.items() if v["n"] >= G1_MIN_YEAR_N}
    pos = sum(1 for v in judged.values() if v["mean_excess"] > 0)
    ok_c = pos >= 2
    return {"a_mean_excess_ge_0.50": ok_a, "b_cluster_ci_lo_gt_0": ok_b,
            "c_positive_years": f"{pos}/{len(judged)} (n>={G1_MIN_YEAR_N})", "c_pass": ok_c,
            "PASS": bool(ok_a and ok_b and ok_c)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", default="outputs/research/ohlcv_polygon_wide_3y.parquet")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    raw_bytes = Path(args.prices).read_bytes()
    combined = pd.read_parquet(args.prices)
    prices = {t: g.drop(columns=["_ticker"]) for t, g in combined.groupby("_ticker")}
    print(f"prices: {len(prices)} tickers, {len(combined)} rows, "
          f"{str(combined['date'].min())[:10]} -> {str(combined['date'].max())[:10]}")
    panel = es.build_panel({t: d for t, d in prices.items() if t != "SPY"})

    sp500 = {t.replace(".", "-").upper() for t in SP500_TICKERS}
    events = build_events(panel, [t for t in prices if t != "SPY"])
    ex = es.event_excess(panel, events, HORIZONS)
    ex["in_sp500"] = ex["ticker"].isin(sp500)
    ex["entry_date"] = pd.to_datetime(ex["entry_date"])
    n_all, n_elig = len(ex), int(ex["eligible"].sum())
    print(f"events with an EPS surprise: {n_all}; liquid as of the prior close: {n_elig}")
    ex = ex[ex["eligible"]]

    result: dict = {"generated_at": datetime.now(timezone.utc).isoformat(),
                    "prices_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "n_events_total": n_all, "n_events_eligible": n_elig,
                    "params": {"min_surprise": MIN_SURPRISE, "min_rev_surprise": MIN_REV_SURPRISE,
                               "reaction_band": REACTION_BAND, "horizons": HORIZONS,
                               "min_price": es.MIN_PRICE, "min_dollar_vol": es.MIN_DOLLAR_VOL},
                    "cells": {}}
    for uni_name, uni in (("non_sp500", ex[~ex["in_sp500"]]), ("sp500", ex[ex["in_sp500"]])):
        for cname, sub in cohorts(uni).items():
            cell: dict = {"per_year_events": {int(y): int(n) for y, n in
                                              sub.groupby(sub["entry_date"].dt.year).size().items()}}
            for h in HORIZONS:
                cell[f"h{h}"] = es.summarize_excess(sub, h)
            cell["by_year_h20"] = by_year(sub, PRIMARY_H)
            cell["by_bucket_h20"] = {int(b): es.summarize_excess(g, PRIMARY_H)
                                     for b, g in sub.groupby("bucket")}
            result["cells"][f"{uni_name}|{cname}"] = cell
            s = cell[f"h{PRIMARY_H}"]
            if s.get("n"):
                print(f"{uni_name:10s} {cname:13s} n={s['n']:5d} dates={s['n_dates']:4d} "
                      f"excess20={s['mean_excess']:+.3f}% cluster[{s['cluster_ci'][0]:+.3f},"
                      f"{s['cluster_ci'][1]:+.3f}] iid[{s['iid_ci'][0]:+.3f},{s['iid_ci'][1]:+.3f}] "
                      f"hit={s['hit']:.1%} fwd={s['mean_fwd']:+.2f} base={s['mean_base']:+.2f}")

    primary = result["cells"]["non_sp500|e1"]
    result["g1"] = g1_verdict(primary[f"h{PRIMARY_H}"], primary["by_year_h20"])
    print("\nG1 (primary = non_sp500|e1, excess_20):", json.dumps(result["g1"]))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1, default=str))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
