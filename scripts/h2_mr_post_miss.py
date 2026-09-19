"""H2 (Stage-0 / gate G1): should MR refuse oversold names right after an earnings MISS?

Mechanism: MR buys RSI(2)<=10 oversold names and is paid when the selling was
noise. After a big earnings miss the selling is information, and bad news is
under-reacted to just as good news is (the short leg of PEAD) — so the "oversold"
reading is a falling knife, not a rubber band. Live already blacks out the days
BEFORE a report (`earnings_blackout_days`); nothing covers the days AFTER one.
Who is on the other side: nobody — this is a filter that removes trades where WE
are the ones under-reacting.

PRE-REGISTERED. Committed before this script was ever run. (Unlike H1/H4, the
wide price parquet already existed at commit time; what had not happened is any
H2 computation. Verify: `git log --format='%h %ad' --date=iso -- scripts/h2_mr_post_miss.py`
against `generated_at` in the output.)

  Trades           live-faithful MR from `scripts/gen_mr_trades.LIVE_MR` with
                   min_score = 75 (the LIVE selectivity — two earlier MR
                   findings existed at min_score 50/60 and vanished at 75, so
                   only 75 is decisional), through the unified exit engine with
                   gap-through fills, on the wide universe, kept only when the
                   name is liquid as of the prior close (price >= $5, 20d mean
                   dollar volume >= $2M).
  POST_MISS        the ticker's most recent report on/before the MR signal date
                   had EPS surprise <= -10% AND the MR signal falls within 10
                   sessions after that report's signal bar.
  G1 PASS needs ALL of:
    (a) n(POST_MISS) >= 30
    (b) mean pnl(POST_MISS) - mean pnl(REST) < 0 with a date-cluster 95% CI
        UPPER bound < 0
    (c) mean pnl(POST_MISS) < 0 (the filter must remove losers, not merely
        smaller winners)
    (d) the difference in (b) is negative in >= 2 of the 3 calendar years that
        have n(POST_MISS) >= 10
  Anything else = REJECTED at G1 — including "right sign, CI crosses zero",
  which is what the short-interest and days-to-cover filters produced.
  Descriptive only (each is a registry variant, none can rescue a fail):
  windows of 5 and 20 sessions; miss threshold -5%; min_score 50; the mirror
  image POST_BEAT (surprise >= +10%).

Usage:
  python scripts/h2_mr_post_miss.py --json-out outputs/research/h2_mr_post_miss.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from scripts.gen_mr_trades import LIVE_MR  # noqa: E402
from scripts.h1_pead_wide import build_events  # noqa: E402
from scripts.h4_consecutive_beats import diff_cluster_ci  # noqa: E402
from src.research import event_study as es  # noqa: E402
from src.research.signal_backtest import run_model_backtest  # noqa: E402

LIVE_MIN_SCORE = 75.0
MISS = -10.0
WINDOW = 10
MIN_N = 30
MIN_YEAR_N = 10


def tag_trades(trades: pd.DataFrame, events: list[dict], panel: es.Panel) -> pd.DataFrame:
    """Attach `last_surprise` and `sessions_since_report`: the most recent report
    whose SIGNAL bar is on/before the MR signal date, and how many sessions ago
    it was. Only backward-looking information is used."""
    idx = panel.dates
    by_t: dict[str, list[tuple[int, float]]] = {}
    for ev in events:
        by_t.setdefault(ev["ticker"], []).append(
            (int(idx.searchsorted(pd.Timestamp(ev["signal_date"]))), float(ev["surprise"])))
    for v in by_t.values():
        v.sort()
    last, since = [], []
    for t, sd in zip(trades["ticker"], trades["signal_date"]):
        i = int(idx.searchsorted(pd.Timestamp(sd)))
        prior = [(k, s) for k, s in by_t.get(t, []) if k <= i]
        if prior:
            k, s = prior[-1]
            last.append(s)
            since.append(i - k)
        else:
            last.append(None)
            since.append(None)
    out = trades.copy()
    out["last_surprise"], out["sessions_since_report"] = last, since
    return out


def split(df: pd.DataFrame, *, miss: float, window: int, beat: bool = False):
    recent = df["sessions_since_report"].notna() & (df["sessions_since_report"] <= window)
    hit = (df["last_surprise"] >= -miss) if beat else (df["last_surprise"] <= miss)
    flag = recent & hit.fillna(False)
    return df[flag], df[~flag]


def cell(df: pd.DataFrame, *, miss: float = MISS, window: int = WINDOW, beat: bool = False) -> dict:
    a, b = split(df, miss=miss, window=window, beat=beat)
    out = {"n_flag": int(len(a)), "n_rest": int(len(b)),
           "mean_flag": float(a["pnl_pct"].mean()) if len(a) else None,
           "mean_rest": float(b["pnl_pct"].mean()) if len(b) else None,
           "wr_flag": float((a["pnl_pct"] > 0).mean()) if len(a) else None,
           "wr_rest": float((b["pnl_pct"] > 0).mean()) if len(b) else None}
    if len(a) >= 2 and len(b) >= 2:
        point, lo, hi = diff_cluster_ci(a, b, "pnl_pct")
        out["diff"] = {"point": point, "cluster_ci": [lo, hi]}
    return out


def run_mr(prices: dict[str, pd.DataFrame], panel: es.Panel, min_score: float) -> pd.DataFrame:
    res = run_model_backtest("mean_reversion", prices, {**LIVE_MR, "min_score": min_score})
    rows = [{"ticker": t.ticker, "signal_date": pd.Timestamp(t.signal_date),
             "entry_date": pd.Timestamp(t.entry_date), "pnl_pct": t.pnl_pct,
             "score": t.score, "exit_reason": t.exit_reason} for t in res.trades]
    df = pd.DataFrame(rows)
    ok = [(d in panel.eligible.index) and (t in panel.eligible.columns) and bool(panel.eligible.at[d, t])
          for t, d in zip(df["ticker"], df["entry_date"])]
    print(f"MR min_score={min_score:g}: {len(df)} trades, {sum(ok)} liquid as of the prior close")
    return df[ok].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", default="outputs/research/ohlcv_polygon_wide_3y.parquet")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    raw_bytes = Path(args.prices).read_bytes()
    combined = pd.read_parquet(args.prices)
    prices = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
              for t, g in combined.groupby("_ticker") if t != "SPY"}
    panel = es.build_panel(prices)
    events = build_events(panel, list(prices))

    result: dict = {"generated_at": datetime.now(timezone.utc).isoformat(),
                    "prices_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "params": {"live_min_score": LIVE_MIN_SCORE, "miss": MISS, "window": WINDOW,
                               "mr": LIVE_MR}, "cells": {}}
    live = tag_trades(run_mr(prices, panel, LIVE_MIN_SCORE), events, panel)
    primary = cell(live)
    result["cells"]["primary_ms75_miss10_w10"] = primary
    by_year = {}
    for y, g in live.groupby(live["entry_date"].dt.year):
        c = cell(g)
        by_year[int(y)] = {"n_flag": c["n_flag"], "diff": (c.get("diff") or {}).get("point")}
    result["cells"]["primary_by_year"] = by_year
    for name, kw in (("w5", {"window": 5}), ("w20", {"window": 20}), ("miss5", {"miss": -5.0}),
                     ("post_beat10", {"beat": True})):
        result["cells"][f"desc_ms75_{name}"] = cell(live, **kw)
    loose = tag_trades(run_mr(prices, panel, 50.0), events, panel)
    result["cells"]["desc_ms50_miss10_w10"] = cell(loose)

    d = primary.get("diff") or {}
    judged = {y: v for y, v in by_year.items() if v["n_flag"] >= MIN_YEAR_N and v["diff"] is not None}
    neg_years = sum(1 for v in judged.values() if v["diff"] < 0)
    g1 = {"a_n_ge_30": primary["n_flag"] >= MIN_N,
          "b_diff_ci_hi_lt_0": bool(d) and d["cluster_ci"][1] < 0,
          "c_flag_mean_lt_0": primary["mean_flag"] is not None and primary["mean_flag"] < 0,
          "d_negative_years": f"{neg_years}/{len(judged)} (n_flag>={MIN_YEAR_N})", "d_pass": neg_years >= 2}
    g1["PASS"] = bool(g1["a_n_ge_30"] and g1["b_diff_ci_hi_lt_0"] and g1["c_flag_mean_lt_0"] and g1["d_pass"])
    result["g1"] = g1
    for k, v in result["cells"].items():
        print(k, json.dumps(v))
    print("G1:", json.dumps(g1))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1, default=str))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
