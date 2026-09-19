"""H5 (Stage-0 / gate G1): "good company, mispriced after a real drawdown".

Origin: the Drift desk of the four-desk research pipeline Ray shared
(2026-09-19) — growth AND valuation support AND a recent drawdown. Ray chose to
test the idea, not build the desk.

Mechanism: investors extrapolate a sharp price fall onto the business; when
revenue is still growing and earnings are positive and cheaply priced, the fall
is partly overreaction that unwinds over months. Who is on the other side:
momentum and de-risking sellers who sell the price, not the business.

PRE-REGISTERED. Committed BEFORE this script has run. The inputs already exist
(the wide price parquet and the refreshed FMP earnings cache, fingerprint
5262c738a5f4...), as they did for H2; what the commit time proves is that no H5
computation preceded the criteria. Verify with
`git log --format='%h %ad' --date=iso -- scripts/h5_quality_drawdown.py`.

  Universe      wide parquet, liquid as of the prior close ($5 / $2M), 4,700+
                names incl. delisted. Current-index membership is NOT used: a
                dip-buying study on survivors is biased upward by construction.
  Fundamentals  from the FMP per-ticker earnings cache only (FMP's quarterly
                key-metrics/ratios are 402 on the Starter plan). For each report
                k: revenue YoY = rev_k / rev_(k-4) - 1 (needs 5 reports with
                positive revenueActual); TTM EPS = sum of epsActual over reports
                k-3..k (all four non-null). A report dated D is usable from the
                close of the SECOND session on/after D (look-ahead-free for any
                release time).
  Signal day S  (evaluated at S's close; ENTRY = next session's open)
                  drawdown: close_S <= 0.75 x max(close over the 252 sessions
                    ending at S), with >= 200 sessions of history
                  growth:   revenue YoY >= +10%
                  earnings: TTM EPS > 0
                  valuation: close_S / TTM EPS <= 20
  QUALITY cohort = all four true. CONTROL cohort = drawdown true, fundamentals
                available, and NOT (growth AND earnings AND valuation).
  Event         the first S on which a cohort's condition holds after >= 20
                sessions of it not holding (one event per episode, not one per
                day).
  Metric        excess return vs the same-day, same-liquidity-tercile base rate.
                PRIMARY HORIZON = 60 sessions (the mechanism is months, not
                weeks); the default G1 line of +0.5%/20d is scaled to +1.5%/60d.
  G1 PASS needs ALL of:
    (a) QUALITY mean excess_60 >= +1.50%
    (b) QUALITY moving-block (block 60) 95% CI lower bound > 0
    (c) QUALITY positive in >= 2 calendar years with n >= 30
    (d) QUALITY - CONTROL mean excess_60 > 0 with a moving-block (block 60)
        95% CI lower bound > 0 — without (d) this is MR-style rebound, which
        the registry already covers.
  Descriptive only (registry variants): horizon 20; drawdown 35%; P/E <= 15;
  liquidity terciles.
  Caveat stated in advance: the 252-session lookback means events start in
  mid-2024, so the window covers ~2 calendar years.

Usage:
  python scripts/h5_quality_drawdown.py --json-out outputs/research/h5_quality_drawdown.json
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.h1_pead_wide import earnings_fingerprint, report_rows  # noqa: E402
from src.research import event_study as es  # noqa: E402

DD = 0.75
LOOKBACK = 252
MIN_HISTORY = 200
MIN_GROWTH = 0.10
MAX_PE = 20.0
REARM = 20
HORIZONS = [20, 60]
PRIMARY_H = 60
G1_MIN_EXCESS = 1.50
G1_MIN_YEAR_N = 30


def fundamentals_by_session(ticker: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-session (revenue YoY, TTM EPS) as KNOWN at each session's close: a
    report dated D becomes usable at the second session on/after D."""
    rows = report_rows(ticker)
    rev, eps, dates = [], [], []
    for r in rows:
        try:
            rv = float(r.get("revenueActual"))
        except (TypeError, ValueError):
            rv = float("nan")
        e = r.get("epsActual")
        rev.append(rv if rv > 0 else float("nan"))
        eps.append(float(e) if e is not None else float("nan"))
        dates.append(pd.Timestamp(str(r["date"])[:10]))
    out = pd.DataFrame(index=idx, data={"rev_yoy": np.nan, "ttm_eps": np.nan})
    recs = []
    for k in range(len(rows)):
        yoy = rev[k] / rev[k - 4] - 1 if k >= 4 and not (np.isnan(rev[k]) or np.isnan(rev[k - 4])) else np.nan
        last4 = eps[k - 3:k + 1] if k >= 3 else []
        ttm = float(sum(last4)) if len(last4) == 4 and not any(np.isnan(x) for x in last4) else np.nan
        i = int(idx.searchsorted(dates[k], side="left")) + 1     # second session on/after D
        if i < len(idx):
            recs.append((i, yoy, ttm))
    for i, yoy, ttm in recs:                                   # later reports overwrite forward
        out.iloc[i:, 0] = yoy
        out.iloc[i:, 1] = ttm
    return out


def onsets(flag: pd.Series, rearm: int = REARM) -> list[int]:
    """Positions where `flag` turns true after >= `rearm` consecutive falses."""
    f = flag.fillna(False).to_numpy(dtype=bool)
    out, off = [], rearm
    for i, v in enumerate(f):
        if v and off >= rearm:
            out.append(i)
        off = 0 if v else off + 1
    return out


def build_events(panel: es.Panel, *, dd: float = DD, max_pe: float = MAX_PE) -> tuple[list[dict], list[dict]]:
    idx = panel.dates
    close = panel.close
    peak = close.rolling(LOOKBACK, min_periods=MIN_HISTORY).max()
    quality, control = [], []
    for t in close.columns:
        c = close[t]
        if c.notna().sum() < MIN_HISTORY:
            continue
        f = fundamentals_by_session(t, idx)
        down = c <= dd * peak[t]
        have = f["rev_yoy"].notna() & f["ttm_eps"].notna()
        pe = c / f["ttm_eps"]
        good = have & (f["rev_yoy"] >= MIN_GROWTH) & (f["ttm_eps"] > 0) & (pe <= max_pe)
        q_flag = down & good
        c_flag = down & have & ~good
        for flag, bucket in ((q_flag, quality), (c_flag, control)):
            for i in onsets(flag):
                if i + 1 < len(idx):
                    bucket.append({"ticker": t, "signal_date": idx[i], "entry_date": idx[i + 1]})
    return quality, control


def by_year(sub: pd.DataFrame, h: int) -> dict:
    s = sub[sub[f"excess_{h}"].notna()]
    return {int(y): {"n": int(len(g)), "mean_excess": float(g[f"excess_{h}"].mean())}
            for y, g in s.groupby(s["entry_date"].dt.year)}


def study(panel: es.Panel, **kw) -> dict:
    q, c = build_events(panel, **kw)
    exq = es.event_excess(panel, q, HORIZONS)
    exc = es.event_excess(panel, c, HORIZONS)
    for ex in (exq, exc):
        ex["entry_date"] = pd.to_datetime(ex["entry_date"])
    exq, exc = exq[exq["eligible"]], exc[exc["eligible"]]
    res: dict = {}
    for h in HORIZONS:
        a = exq[exq[f"excess_{h}"].notna()]
        b = exc[exc[f"excess_{h}"].notna()]
        res[f"h{h}"] = {
            "quality": es.summarize_excess(exq, h, calendar=panel.dates),
            "control": es.summarize_excess(exc, h, calendar=panel.dates),
            "diff_block": list(es.block_diff_ci(a[f"excess_{h}"].tolist(), list(a["entry_date"]),
                                                b[f"excess_{h}"].tolist(), list(b["entry_date"]),
                                                panel.dates, block=h)),
        }
    res["quality_by_year_h60"] = by_year(exq, PRIMARY_H)
    res["quality_by_bucket_h60"] = {int(b): es.summarize_excess(g, PRIMARY_H) for b, g in exq.groupby("bucket")}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", default="outputs/research/ohlcv_polygon_wide_3y.parquet")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    raw = Path(args.prices).read_bytes()
    comb = pd.read_parquet(args.prices)
    prices = {t: g.drop(columns=["_ticker"]) for t, g in comb.groupby("_ticker") if t != "SPY"}
    panel = es.build_panel(prices)
    out: dict = {"generated_at": datetime.now(timezone.utc).isoformat(),
                 "prices_sha256": hashlib.sha256(raw).hexdigest(),
                 "earnings": earnings_fingerprint(list(prices)), "cells": {}}
    out["cells"]["primary"] = study(panel)
    out["cells"]["desc_dd35"] = study(panel, dd=0.65)
    out["cells"]["desc_pe15"] = study(panel, max_pe=15.0)

    p = out["cells"]["primary"]
    for h in HORIZONS:
        q, c, d = p[f"h{h}"]["quality"], p[f"h{h}"]["control"], p[f"h{h}"]["diff_block"]
        print(f"h{h}: QUALITY n={q.get('n')} ex={q.get('mean_excess', float('nan')):+.3f} "
              f"block{[round(x, 2) for x in q.get('block_ci', [])]} | CONTROL n={c.get('n')} "
              f"ex={c.get('mean_excess', float('nan')):+.3f} | diff={d[0]:+.3f} block[{d[1]:+.2f},{d[2]:+.2f}]")
    q = p[f"h{PRIMARY_H}"]["quality"]
    d = p[f"h{PRIMARY_H}"]["diff_block"]
    judged = {y: v for y, v in p["quality_by_year_h60"].items() if v["n"] >= G1_MIN_YEAR_N}
    pos = sum(1 for v in judged.values() if v["mean_excess"] > 0)
    g1 = {"a_mean_ge_1.50": bool(q.get("n")) and q["mean_excess"] >= G1_MIN_EXCESS,
          "b_block_ci_lo_gt_0": bool(q.get("n")) and q["block_ci"][0] > 0,
          "c_positive_years": f"{pos}/{len(judged)} (n>={G1_MIN_YEAR_N})", "c_pass": pos >= 2,
          "d_quality_minus_control_block_lo_gt_0": bool(d[1] > 0)}
    g1["PASS"] = bool(g1["a_mean_ge_1.50"] and g1["b_block_ci_lo_gt_0"] and g1["c_pass"]
                      and g1["d_quality_minus_control_block_lo_gt_0"])
    out["g1"] = g1
    print("G1:", json.dumps(g1))
    for k in ("desc_dd35", "desc_pe15"):
        c = out["cells"][k][f"h{PRIMARY_H}"]
        print(f"{k}: QUALITY n={c['quality'].get('n')} ex={c['quality'].get('mean_excess', float('nan')):+.3f} "
              f"diff={c['diff_block'][0]:+.3f} block[{c['diff_block'][1]:+.2f},{c['diff_block'][2]:+.2f}]")
    if args.json_out:
        o = Path(args.json_out)
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(out, indent=1, default=str))
        print(f"wrote {o}")


if __name__ == "__main__":
    main()
