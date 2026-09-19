"""H4 (Stage-0 / gate G1): is drift stronger after a SECOND consecutive big beat?

Mechanism: a >=10% EPS beat means the consensus was wrong; when the very next
quarter is ALSO a >=10% beat, analysts demonstrably failed to update after the
first one. If estimates anchor, prices anchored on them do too, and the second
surprise is under-reacted to again. Who is on the other side: holders and
analysts treating each beat as a one-off. (The opposite prior is also credible —
a serial beater is "known" and priced — which is why this is tested, not assumed.)

PRE-REGISTERED (committed before the wide-universe data existed; verify with
`git log --format='%h %ad' --date=iso -- scripts/h4_consecutive_beats.py`
against `generated_at` in the output):

  Universe         every name in the wide parquet, liquid as of the prior close
                   (price >= $5, 20d mean dollar volume >= $2M). S&P membership
                   is reported as a split, not used as a filter.
  Event            EPS surprise >= 10% (the live PEAD threshold, unchanged).
  Cohorts          CONSECUTIVE = the same ticker's immediately preceding report
                   (60-130 calendar days earlier) was also a >= 10% beat.
                   FIRST = the preceding report exists in that window and was
                   < 10%. Events with no preceding report in the window are
                   excluded from both and counted.
  Primary metric   mean excess_20 (entry-bar open -> close 20 sessions later vs
                   the same-day, same-liquidity-tercile base rate).
  G1 PASS needs ALL of, on the CONSECUTIVE cohort:
    (a) mean excess_20 >= +0.50%
    (b) date-cluster bootstrap 95% CI lower bound > 0
    (c) positive mean excess_20 in >= 2 of the 3 calendar years with n >= 30
  AND the claim that gives H4 a reason to exist:
    (d) mean excess_20(CONSECUTIVE) - mean excess_20(FIRST) > 0 with a
        date-cluster CI lower bound > 0 (difference bootstrap over the pooled
        entry dates).
  (a)-(c) without (d) means "consecutive beats drift, but no more than any
  beat" — that is PEAD, already registered, and H4 is REJECTED as a separate
  idea. No threshold is tuned. E1/neglected gates are NOT applied here (they
  are H1's); applying them would be a second variant.

Usage:
  python scripts/h4_consecutive_beats.py --json-out outputs/research/h4_consecutive_beats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from scripts.h1_pead_wide import (  # noqa: E402
    G1_MIN_YEAR_N, MIN_SURPRISE, PRIMARY_H, build_events, by_year, earnings_fingerprint, g1_verdict,
    report_rows,
)
from src.signals.post_earnings_drift import eps_surprise_pct  # noqa: E402
from src.research import event_study as es  # noqa: E402
from src.research.sp500_tickers import SP500_TICKERS  # noqa: E402

PREV_GAP_DAYS = (60, 130)


def tag_previous(events: list[dict], history: dict[str, list[tuple[str, float | None]]] | None = None) -> list[dict]:
    """Attach `prev_surprise` = the same ticker's immediately preceding REPORT's
    surprise if it falls 60-130 calendar days earlier, else None.

    `history` maps ticker -> every report as (ISO date, surprise-or-None), from
    the FULL earnings history — not just the events inside the price window, so
    the first in-window event still finds its predecessor. A preceding report
    whose surprise is unknown is still the predecessor (it is a boundary, not
    skipped), which leaves prev_surprise None. Without `history` the events
    themselves are used (the registered behaviour). Only earlier reports are
    ever consulted."""
    if history is None:
        history = {}
        for ev in events:
            history.setdefault(ev["ticker"], []).append((ev["report_date"], ev["surprise"]))
    hist = {t: sorted(v) for t, v in history.items()}
    for ev in events:
        ev["prev_surprise"] = None
        prior = [h for h in hist.get(ev["ticker"], []) if h[0] < ev["report_date"]]
        if not prior:
            continue
        pdate, psurp = prior[-1]
        gap = (date.fromisoformat(ev["report_date"]) - date.fromisoformat(pdate)).days
        if PREV_GAP_DAYS[0] <= gap <= PREV_GAP_DAYS[1]:
            ev["prev_surprise"] = psurp
    return events


def diff_cluster_ci(a: pd.DataFrame, b: pd.DataFrame, col: str, n_boot: int = 10_000,
                    seed: int = es.SEED) -> tuple[float, float, float]:
    """mean(a) - mean(b) with a percentile CI that resamples ENTRY DATES from the
    pooled calendar and recomputes both cohort means on the drawn dates."""
    def _by_date(df):
        g = df.groupby(df["entry_date"].astype(str).str[:10])[col]
        return g.sum().to_dict(), g.count().to_dict()
    sa, ca = _by_date(a)
    sb, cb = _by_date(b)
    dates = sorted(set(sa) | set(sb))
    rng = random.Random(seed)
    m, out = len(dates), []
    for _ in range(n_boot):
        ta = na = tb = nb = 0.0
        for _j in range(m):
            d = dates[rng.randrange(m)]
            ta += sa.get(d, 0.0)
            na += ca.get(d, 0)
            tb += sb.get(d, 0.0)
            nb += cb.get(d, 0)
        if na and nb:
            out.append(ta / na - tb / nb)
    out.sort()
    point = float(a[col].mean() - b[col].mean())
    return point, out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", default="outputs/research/ohlcv_polygon_wide_3y.parquet")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--timing", choices=["registered", "volume"], default="registered")
    ap.add_argument("--predecessors", choices=["window", "full"], default="full",
                    help="window = the registered run's lookup (events inside the price window); "
                         "full = every report in the earnings history (Codex fix)")
    args = ap.parse_args()

    raw_bytes = Path(args.prices).read_bytes()
    combined = pd.read_parquet(args.prices)
    prices = {t: g.drop(columns=["_ticker"]) for t, g in combined.groupby("_ticker") if t != "SPY"}
    panel = es.build_panel(prices)
    events = build_events(panel, list(prices), timing=args.timing)
    history = None
    if args.predecessors == "full":
        history = {t: [(str(r["date"])[:10], eps_surprise_pct(r.get("epsActual"), r.get("epsEstimated")))
                       for r in report_rows(t)] for t in prices}
    events = tag_previous(events, history)
    ex = es.event_excess(panel, events, [PRIMARY_H])
    ex["entry_date"] = pd.to_datetime(ex["entry_date"])
    sp500 = {t.replace(".", "-").upper() for t in SP500_TICKERS}
    ex["in_sp500"] = ex["ticker"].isin(sp500)

    beats = ex[ex["eligible"] & (ex["surprise"] >= MIN_SURPRISE)]
    col = f"excess_{PRIMARY_H}"
    no_prev = beats[beats["prev_surprise"].isna()]
    cons = beats[beats["prev_surprise"].notna() & (beats["prev_surprise"] >= MIN_SURPRISE)]
    first = beats[beats["prev_surprise"].notna() & (beats["prev_surprise"] < MIN_SURPRISE)]

    result: dict = {"generated_at": datetime.now(timezone.utc).isoformat(),
                    "timing": args.timing, "predecessors": args.predecessors,
                    "prices_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "earnings": earnings_fingerprint(list(prices)),
                    "n_beats_eligible": int(len(beats)), "n_no_previous_report": int(len(no_prev)),
                    "cells": {}}
    for name, sub in (("consecutive", cons), ("first", first)):
        cell = {"all": es.summarize_excess(sub, PRIMARY_H, calendar=panel.dates), "by_year": by_year(sub, PRIMARY_H),
                "sp500": es.summarize_excess(sub[sub["in_sp500"]], PRIMARY_H),
                "non_sp500": es.summarize_excess(sub[~sub["in_sp500"]], PRIMARY_H)}
        result["cells"][name] = cell
        s = cell["all"]
        print(f"{name:12s} n={s['n']:5d} dates={s['n_dates']:4d} excess20={s['mean_excess']:+.3f}% "
              f"cluster[{s['cluster_ci'][0]:+.3f},{s['cluster_ci'][1]:+.3f}] hit={s['hit']:.1%}")

    c_ok, f_ok = cons[cons[col].notna()], first[first[col].notna()]
    point, lo, hi = diff_cluster_ci(c_ok, f_ok, col)
    g1 = g1_verdict(result["cells"]["consecutive"]["all"], result["cells"]["consecutive"]["by_year"])
    g1["d_consecutive_minus_first"] = {"point": point, "cluster_ci": [lo, hi], "pass": lo > 0}
    g1["PASS"] = bool(g1["PASS"] and lo > 0)
    result["g1"] = g1
    print(f"consecutive - first = {point:+.3f}pp  cluster CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"G1 (n>={G1_MIN_YEAR_N}/yr):", json.dumps(g1))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1, default=str))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
