"""H1 follow-up decomposition — POST-HOC, DESCRIPTIVE ONLY.

Written AFTER h1_pead_wide.py returned its verdict (REJECTED at G1). Nothing
here can change that verdict or pass any gate; every cut below is logged in
docs/research_registry.md as a variant and any lead it surfaces must be
re-registered as a new hypothesis and tested on data not used here.

It answers three questions the G1 result raised:
  1. Survivorship: do current S&P 500 members beat the same-day matched base
     rate even with NO event? (The 3Y backtests replay the March-2026 member
     list backwards, so membership itself is look-ahead.)
  2. The population live PEAD actually trades: E1 / neglected on the WHOLE
     survivorship-free universe, overall and by liquidity tercile.
  3. S&P E1 measured against an S&P-only same-day base, so the comparison is
     member vs member rather than member vs everyone.

Usage:
  python scripts/h1_decomposition.py --json-out outputs/research/h1_decomposition.json
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

from scripts.h1_pead_wide import build_events, cohorts, earnings_fingerprint  # noqa: E402
from src.research import event_study as es  # noqa: E402
from src.research.sp500_tickers import SP500_TICKERS  # noqa: E402

H = 20


def unconditional_excess(panel: es.Panel, fwd: pd.DataFrame, base: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Every eligible (date, ticker) cell's forward return minus its same-day,
    same-bucket base rate — no event conditioning at all."""
    f = fwd[cols].where(panel.eligible[cols])
    b = panel.bucket[cols]
    parts = [f.where(b == k).sub(base[k], axis=0).stack() for k in base.columns]
    return pd.concat(parts).dropna()   # pandas 3 stack() keeps NaNs; they are not observations


def _splits_or_none(on: bool):
    import json as _json
    from pathlib import Path as _P
    return _json.loads((_P(__file__).resolve().parents[1] / "data/cache/corp_actions/splits.json").read_text()) if on else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", default="outputs/research/ohlcv_polygon_wide_3y.parquet")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--raw-price-screen", action="store_true",
                    help="apply the $5 floor to RAW (split-unadjusted) prices using data/cache/corp_actions/splits.json")
    ap.add_argument("--timing", choices=["registered", "volume"], default="registered")
    args = ap.parse_args()

    raw = Path(args.prices).read_bytes()
    comb = pd.read_parquet(args.prices)
    prices = {t: g.drop(columns=["_ticker"]) for t, g in comb.groupby("_ticker") if t != "SPY"}
    panel = es.build_panel(prices, splits=_splits_or_none(args.raw_price_screen))
    sp = {t.replace(".", "-").upper() for t in SP500_TICKERS}
    fwd = es.forward_returns(panel, H)
    base = es.base_rate(panel, fwd)
    sp_cols = [c for c in fwd.columns if c in sp]
    other_cols = [c for c in fwd.columns if c not in sp]

    out: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "raw_price_screen": bool(args.raw_price_screen), "timing": args.timing,
                 "prices_sha256": hashlib.sha256(raw).hexdigest(), "posthoc": True,
                 "earnings": earnings_fingerprint(list(prices))}

    u_sp = unconditional_excess(panel, fwd, base, sp_cols)
    u_ot = unconditional_excess(panel, fwd, base, other_cols)
    sp_daily = u_sp.groupby(level=0).agg(["sum", "count"])
    x_sp = []
    d_sp = []
    for d, row in sp_daily.iterrows():   # one value per date: that date's mean excess across members
        x_sp.append(row["sum"] / row["count"])
        d_sp.append(d)
    q1_block = es.block_boot_ci(x_sp, d_sp, panel.dates, block=H)
    sp_bucket_share = (panel.bucket[sp_cols].where(panel.eligible[sp_cols])
                       .stack().value_counts(normalize=True).sort_index())
    yrs = u_sp.index.get_level_values(0).year
    out["q1_unconditional_excess20"] = {
        "sp500_members": {"mean": float(u_sp.mean()), "n_obs": int(len(u_sp)),
                          "mean_of_daily_means": float(sp_daily["sum"].div(sp_daily["count"]).mean()),
                          "block_ci_of_daily_means": list(q1_block),
                          "share_by_liquidity_bucket": {int(k): float(v) for k, v in sp_bucket_share.items()},
                          "by_year": {int(y): float(v) for y, v in u_sp.groupby(yrs).mean().items()}},
        "non_members": {"mean": float(u_ot.mean()), "n_obs": int(len(u_ot))},
    }
    print(f"Q1 no-event excess_20: S&P members {u_sp.mean():+.3f}% (n={len(u_sp):,}, block CI of daily means "
          f"[{q1_block[0]:+.3f},{q1_block[1]:+.3f}]; bucket shares {sp_bucket_share.round(3).to_dict()}) | "
          f"non-members {u_ot.mean():+.3f}% (n={len(u_ot):,})")

    ev = build_events(panel, list(prices), timing=args.timing)
    ex = es.event_excess(panel, ev, [H, 60])
    ex["entry_date"] = pd.to_datetime(ex["entry_date"])
    ex = ex[ex["eligible"]]
    q2: dict = {}
    for name, sub in cohorts(ex).items():
        q2[name] = {"h20": es.summarize_excess(sub, H, calendar=panel.dates),
                    "h60": es.summarize_excess(sub, 60, calendar=panel.dates),
                    "by_bucket_h20": {int(b): es.summarize_excess(g, H) for b, g in sub.groupby("bucket")}}
        s = q2[name]["h20"]
        print(f"Q2 whole universe {name:13s} n={s['n']:5d} excess20={s['mean_excess']:+.3f} "
              f"cluster[{s['cluster_ci'][0]:+.2f},{s['cluster_ci'][1]:+.2f}]")
    out["q2_whole_universe"] = q2

    sp_base = fwd[sp_cols].where(panel.eligible[sp_cols]).mean(axis=1)
    q3: dict = {}
    for name, sub in cohorts(ex[ex["ticker"].isin(sp)]).items():
        sub = sub[sub[f"fwd_{H}"].notna()].copy()
        sub["x"] = sub[f"fwd_{H}"] - sub["entry_date"].map(sp_base)
        lo, hi = es.cluster_boot_ci(sub["x"].tolist(), sub["entry_date"].astype(str).str[:10].tolist())
        q3[name] = {"n": int(len(sub)), "mean_excess": float(sub["x"].mean()), "cluster_ci": [lo, hi]}
        print(f"Q3 S&P {name:13s} vs S&P-only base: n={len(sub)} excess20={sub['x'].mean():+.3f} "
              f"cluster[{lo:+.2f},{hi:+.2f}]")
    out["q3_sp500_vs_sp500_base"] = q3

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1, default=str))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
