"""Sniper gap-risk test — can we identify the gap-down tail EX-ANTE?

Finding (trail_sweep_FINDINGS): sniper's worst 5% of trades erase 93% of gross
return, and the "gap signature" (held <=2d, lost >=5%) is 93 trades worth -715%.
Avoiding them would double expectancy (0.54% -> 1.31%). But that is trivially
true in-sample. The real question: does a trailing overnight-gap-volatility
feature — computed only from data BEFORE entry — predict which names gap, so we
can filter them at entry without look-ahead?

Discipline (learned from the trail sweep): test on the full cohort, check
sub-period stability, and deflate for the thresholds tried. A filter that only
works in-aggregate or in one window is overfitting.

gap_vol(ticker, asof) = trailing-60d 90th-percentile of |open_t / close_{t-1} - 1|
  (how big this name's overnight moves get, using only bars strictly before asof).

--cohort PATH applies the SAME feature to the LIVE sniper trades in a frozen
dashboard bundle (trades["sniper|mas_official"]), fetched Polygon-strict, and
reports whether the thresholds implied by the backtest cohort would have removed
the live `time_stop` losers (day-1 gap-throughs) without removing the
`trail_stop` exits. The live cohort is n~30: descriptive only, never a decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from statistics import mean

import pandas as pd

from src.research.signal_backtest import (
    fetch_ohlcv,
    get_last_ohlcv_provenance,
    run_model_backtest,
)
from src.research.sp500_tickers import SP500_TICKERS

SNIPER_ENTRY = dict(use_spy=True, min_score=70, atr_pct_floor=5.0,
                    stop_atr_mult=1.5, target_atr_mult=3.0, holding_period=7,
                    gap_through=True, sniper_time_stop_days=1,
                    trail_activate_pct=0.5, trail_distance_pct=0.3)  # live config

GAP_WINDOW = 60  # trailing trading days for the gap-vol estimate
LIVE_STREAM = "sniper|mas_official"
PCTILES = [0.95, 0.90, 0.80, 0.70]


def _load(cache_file, years):
    if cache_file and pd is not None:
        combined = pd.read_parquet(cache_file)
        return {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
                for t, g in combined.groupby("_ticker")}
    tickers = list(SP500_TICKERS) + ["SPY"]
    return fetch_ohlcv(tickers, years=years)


def gap_vol_asof(df: pd.DataFrame, asof: date) -> float | None:
    """Trailing 90th-pct absolute overnight gap using ONLY bars before asof."""
    d = df[df["date"] < pd.Timestamp(asof)].tail(GAP_WINDOW + 1)
    if len(d) < 20:
        return None
    o = d["open"].to_numpy()
    pc = d["close"].shift(1).to_numpy()
    gaps = abs(o[1:] / pc[1:] - 1.0)
    gaps = gaps[~pd.isna(gaps)]
    if len(gaps) < 15:
        return None
    return float(pd.Series(gaps).quantile(0.90) * 100)  # percent


def _stats(pnls):
    if not pnls:
        return (0, 0.0, 0.0, 0.0)
    wr = sum(1 for x in pnls if x > 0) / len(pnls)
    worst = sorted(pnls)[: max(1, len(pnls) // 20)]  # worst 5%
    return (len(pnls), wr, mean(pnls), sum(worst))


def _normalize_dates(price: dict) -> None:
    for t, df in price.items():
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])


def load_live_cohort(path: Path, stream: str = LIVE_STREAM) -> tuple[list[dict], str]:
    """Closed live trades of one stream from a frozen dashboard bundle.

    Returns (rows, sha256 of the file). Streams are never blended — one stream
    key, exactly as the bundle stores it. Rows without a realised pnl or a
    signal_date are skipped (they can't be scored or attributed).
    """
    raw = path.read_bytes()
    data = json.loads(raw)
    out = []
    for r in data["trades"][stream]:
        if r.get("pnl_pct") is None or not r.get("signal_date"):
            continue
        out.append({
            "ticker": r["ticker"],
            "signal_date": date.fromisoformat(r["signal_date"]),
            "entry_date": r.get("entry_date"),
            "exit_reason": r.get("exit_reason"),
            "pnl": float(r["pnl_pct"]),
            "mfe": r.get("mfe"),
            "mae": r.get("mae"),
        })
    return out, hashlib.sha256(raw).hexdigest()


def attach_gap_vol(rows: list[dict], price: dict) -> list[dict]:
    """Add ex-ante gap_vol (None if the name/history is missing) to each row."""
    out = []
    for r in rows:
        df = price.get(r["ticker"])
        gv = gap_vol_asof(df, r["signal_date"]) if df is not None else None
        out.append({**r, "gap_vol": gv})
    return out


def threshold_effect(scored: list[dict], thr: float) -> dict:
    """What dropping gap_vol > thr does to a cohort, split by exit reason."""
    kept = [r for r in scored if r["gap_vol"] <= thr]
    dropped = [r for r in scored if r["gap_vol"] > thr]
    by_reason: dict[str, int] = {}
    for r in dropped:
        k = r.get("exit_reason") or "?"
        by_reason[k] = by_reason.get(k, 0) + 1
    n, wr, avg, w5 = _stats([r["pnl"] for r in kept])
    return {
        "threshold": thr, "kept": n, "dropped": len(dropped),
        "dropped_by_reason": by_reason,
        "kept_wr": wr, "kept_avg": avg, "kept_worst5_sum": w5,
        "dropped_avg": mean([r["pnl"] for r in dropped]) if dropped else None,
    }


def backtest_thresholds(scored: list[dict]) -> dict[str, float]:
    """gap_vol at each percentile of the BACKTEST cohort — the thresholds a live
    filter would actually be set from (never from the live cohort itself)."""
    gvs = sorted(r["gap_vol"] for r in scored)
    return {f"p{int(p * 100)}": gvs[int(len(gvs) * p)] for p in PCTILES}


def _year_windows(scored: list[dict]) -> list[tuple[date, date]]:
    lo = min(r["date"] for r in scored)
    return [(date(lo.year + k, lo.month, lo.day), date(lo.year + k + 1, lo.month, lo.day))
            for k in range(3)]


def run_backtest_cohort(price: dict) -> tuple[list[dict], dict]:
    """The original analysis (stdout unchanged); also returns the scored rows and
    a JSON summary so the live run can reuse its thresholds + per-year stability."""
    res = run_model_backtest("sniper", price, SNIPER_ENTRY)
    trades = res.trades
    print(f"Sniper cohort: N={len(trades)}")

    # Attach ex-ante gap_vol to each trade.
    rows = []
    for t in trades:
        df = price.get(t.ticker)
        gv = gap_vol_asof(df, t.signal_date) if df is not None else None
        # signal_date arrives as a pandas Timestamp from the scan window; the
        # per-year windows below are plain dates, and pandas>=2 refuses to
        # compare the two.
        rows.append({"ticker": t.ticker, "date": pd.Timestamp(t.signal_date).date(),
                     "pnl": t.pnl_pct, "regime": t.regime, "gap_vol": gv,
                     "exit_reason": t.exit_reason})
    scored = [r for r in rows if r["gap_vol"] is not None]
    print(f"  with gap_vol: {len(scored)}/{len(rows)}")

    summary: dict = {"n": len(scored), "quintiles": [], "thresholds": {}}

    # 1) Is gap_vol predictive? Bucket by quintile.
    gv_sorted = sorted(scored, key=lambda r: r["gap_vol"])
    q = len(gv_sorted) // 5
    print("\n-- expectancy by gap_vol quintile (Q1=calmest, Q5=gappiest) --")
    print(f"{'bucket':<8}{'gap_vol range':>18}{'N':>6}{'WR':>7}{'avg%':>8}{'worst5%sum':>12}")
    for i in range(5):
        b = gv_sorted[i*q:(i+1)*q] if i < 4 else gv_sorted[i*q:]
        n, wr, avg, w5 = _stats([r["pnl"] for r in b])
        lo, hi = b[0]["gap_vol"], b[-1]["gap_vol"]
        print(f"Q{i+1:<7}{f'{lo:.1f}-{hi:.1f}%':>18}{n:>6}{wr:>7.1%}{avg:>8.3f}{w5:>12.1f}")
        summary["quintiles"].append({"q": i + 1, "lo": lo, "hi": hi, "n": n,
                                     "wr": wr, "avg": avg, "worst5_sum": w5})

    # 2) Ex-ante filter: drop the gappiest names above a threshold. Test a few
    #    thresholds; report BASELINE vs filtered + per-year stability.
    base_n, base_wr, base_avg, base_w5 = _stats([r["pnl"] for r in scored])
    print(f"\nBASELINE (all with gap_vol): N={base_n} WR={base_wr:.1%} "
          f"avg={base_avg:+.3f}% worst5%={base_w5:.0f}%")
    summary["baseline"] = {"n": base_n, "wr": base_wr, "avg": base_avg, "worst5_sum": base_w5}
    gvs = sorted(r["gap_vol"] for r in scored)
    print("\n-- filter: drop entries with gap_vol > threshold --")
    print(f"{'thresh(pctile)':<16}{'kept':>6}{'dropped':>8}{'WR':>7}{'avg%':>8}"
          f"{'worst5%':>10}{'yr1':>8}{'yr2':>8}{'yr3':>8}")
    years = _year_windows(scored)
    for pctile in PCTILES:
        thr = gvs[int(len(gvs) * pctile)]
        kept = [r for r in scored if r["gap_vol"] <= thr]
        n, wr, avg, w5 = _stats([r["pnl"] for r in kept])
        yr_avgs = []
        for a, b in years:
            yp = [r["pnl"] for r in kept if a <= r["date"] < b]
            yr_avgs.append(mean(yp) if yp else 0.0)
        print(f"{f'p{int(pctile*100)} ({thr:.1f}%)':<16}{n:>6}{base_n-n:>8}{wr:>7.1%}"
              f"{avg:>8.3f}{w5:>10.1f}" + "".join(f"{y:>8.3f}" for y in yr_avgs))
        dropped_bt = [r for r in scored if r["gap_vol"] > thr]
        summary["thresholds"][f"p{int(pctile * 100)}"] = {
            "threshold": thr, "kept": n, "dropped": base_n - n, "wr": wr, "avg": avg,
            "worst5_sum": w5, "per_year_avg": yr_avgs,
            "dropped_avg": mean([r["pnl"] for r in dropped_bt]) if dropped_bt else None,
        }
    print("\n(baseline per-year avg for reference:", end=" ")
    base_years = []
    for a, b in years:
        yp = [r["pnl"] for r in scored if a <= r["date"] < b]
        base_years.append(mean(yp) if yp else 0.0)
        print(f"{base_years[-1]:.3f}", end=" ")
    print(")")
    summary["baseline"]["per_year_avg"] = base_years
    summary["year_windows"] = [(a.isoformat(), b.isoformat()) for a, b in years]
    return scored, summary


def run_live_cohort(cohort: Path, bt_thresholds: dict[str, float]) -> dict:
    """Score the live sniper trades with the same ex-ante feature and report what
    the backtest-derived thresholds would have removed, by exit reason."""
    rows, sha = load_live_cohort(cohort)
    print(f"\n=== LIVE COHORT {cohort} ===\n  sha256 {sha}\n  rows: {len(rows)}")
    tickers = sorted({r["ticker"] for r in rows})
    price = fetch_ohlcv(tickers + ["SPY"], years=1.5, source="polygon",
                        strict=True, no_cache=True)
    prov = get_last_ohlcv_provenance()
    print("PROVENANCE:", json.dumps(prov, default=str)[:600])
    _normalize_dates(price)

    scored_all = attach_gap_vol(rows, price)
    scored = [r for r in scored_all if r["gap_vol"] is not None]
    missing = [r["ticker"] for r in scored_all if r["gap_vol"] is None]
    print(f"  with gap_vol: {len(scored)}/{len(scored_all)}"
          + (f"  (no feature: {','.join(missing)})" if missing else ""))

    def _f(x, w):
        return f"{(x if x is not None else float('nan')):>{w}.2f}"

    # (a) per-row table — stays in stdout/JSON, never in the findings md.
    print(f"\n{'ticker':<7}{'signal':>11}{'exit':>11}{'pnl%':>8}{'mfe':>7}{'mae':>8}{'gap_vol':>9}")
    for r in sorted(scored, key=lambda r: r["gap_vol"]):
        print(f"{r['ticker']:<7}{r['signal_date'].isoformat():>11}{(r['exit_reason'] or '?'):>11}"
              f"{r['pnl']:>8.2f}{_f(r['mfe'], 7)}{_f(r['mae'], 8)}{r['gap_vol']:>9.2f}")

    # (b) expectancy by quintile, same shape as the backtest table.
    gv_sorted = sorted(scored, key=lambda r: r["gap_vol"])
    q = max(1, len(gv_sorted) // 5)
    quintiles = []
    print("\n-- LIVE expectancy by gap_vol quintile --")
    print(f"{'bucket':<8}{'gap_vol range':>18}{'N':>6}{'WR':>7}{'avg%':>8}{'time_stops':>12}")
    for i in range(5):
        b = gv_sorted[i*q:(i+1)*q] if i < 4 else gv_sorted[i*q:]
        if not b:
            continue
        n, wr, avg, _ = _stats([r["pnl"] for r in b])
        ts = sum(1 for r in b if r["exit_reason"] == "time_stop")
        rng = f"{b[0]['gap_vol']:.1f}-{b[-1]['gap_vol']:.1f}%"
        print(f"Q{i+1:<7}{rng:>18}{n:>6}{wr:>7.1%}{avg:>8.3f}{ts:>12}")
        quintiles.append({"q": i + 1, "lo": b[0]["gap_vol"], "hi": b[-1]["gap_vol"], "n": n,
                          "wr": wr, "avg": avg, "time_stops": ts})

    # (c) thresholds: the backtest-implied ones (what a live filter would be set
    #     from) AND the live cohort's own percentiles (descriptive comparison).
    reasons: dict[str, int] = {}
    for r in scored:
        k = r["exit_reason"] or "?"
        reasons[k] = reasons.get(k, 0) + 1
    base_n, base_wr, base_avg, _ = _stats([r["pnl"] for r in scored])
    print(f"\nLIVE BASELINE: N={base_n} WR={base_wr:.1%} avg={base_avg:+.3f}%  exits={reasons}")
    effects = {}
    print("\n-- filter on LIVE cohort: drop gap_vol > threshold --")
    print(f"{'threshold':<22}{'kept':>6}{'dropped':>8}{'drop time_stop':>15}{'drop trail':>11}"
          f"{'keptWR':>8}{'kept avg%':>10}{'dropped avg%':>13}")
    live_gvs = sorted(r["gap_vol"] for r in scored)
    candidates = [(f"backtest {k}", v) for k, v in bt_thresholds.items()]
    candidates += [(f"live p{int(p * 100)}",
                    live_gvs[min(len(live_gvs) - 1, int(len(live_gvs) * p))])
                   for p in PCTILES]
    for label, thr in candidates:
        e = threshold_effect(scored, thr)
        effects[label] = e
        d = e["dropped_by_reason"]
        print(f"{f'{label} ({thr:.1f}%)':<22}{e['kept']:>6}{e['dropped']:>8}"
              f"{d.get('time_stop', 0):>15}{d.get('trail_stop', 0):>11}{e['kept_wr']:>8.1%}"
              f"{e['kept_avg']:>10.3f}{_f(e['dropped_avg'], 13)}")

    return {
        "input": str(cohort), "input_sha256": sha, "stream": LIVE_STREAM,
        "n_rows": len(scored_all), "n_scored": len(scored), "no_feature": missing,
        "exit_reasons": reasons,
        "baseline": {"n": base_n, "wr": base_wr, "avg": base_avg},
        "quintiles": quintiles,
        "threshold_effects": effects,
        "rows": [{**r, "signal_date": r["signal_date"].isoformat()} for r in scored_all],
        "provenance": prov,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--cache-file", default="outputs/research/ohlcv_3y_cache.parquet")
    ap.add_argument("--cohort", help="frozen dashboard bundle; score its live sniper "
                                     "trades with the same ex-ante feature")
    ap.add_argument("--json-out", help="write summary tables (+ provenance) here; "
                                       "must be under outputs/research/")
    args = ap.parse_args()

    price = _load(args.cache_file, args.years)
    print(f"Loaded {len(price)} tickers")

    # Normalize date column to Timestamp for comparison.
    _normalize_dates(price)

    bt_scored, bt_summary = run_backtest_cohort(price)

    live = None
    if args.cohort:
        live = run_live_cohort(Path(args.cohort), backtest_thresholds(bt_scored))

    if args.json_out:
        out = Path(args.json_out)
        if "outputs/research" not in str(out):
            raise SystemExit("--json-out must be under outputs/research/ (gitignored)")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"backtest": bt_summary, "live": live,
                                   "cache_file": args.cache_file, "entry": SNIPER_ENTRY},
                                  indent=2, default=str))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
