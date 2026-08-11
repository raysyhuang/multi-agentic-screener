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
"""
from __future__ import annotations

import argparse
from datetime import date
from statistics import mean

import pandas as pd

from src.research.signal_backtest import fetch_ohlcv, run_model_backtest
from src.research.sp500_tickers import SP500_TICKERS

SNIPER_ENTRY = dict(use_spy=True, min_score=70, atr_pct_floor=5.0,
                    stop_atr_mult=1.5, target_atr_mult=3.0, holding_period=7,
                    gap_through=True, sniper_time_stop_days=1,
                    trail_activate_pct=0.5, trail_distance_pct=0.3)  # live config

GAP_WINDOW = 60  # trailing trading days for the gap-vol estimate


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--cache-file", default="outputs/research/ohlcv_3y_cache.parquet")
    args = ap.parse_args()

    price = _load(args.cache_file, args.years)
    print(f"Loaded {len(price)} tickers")

    # Normalize date column to Timestamp for comparison.
    for t, df in price.items():
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

    res = run_model_backtest("sniper", price, SNIPER_ENTRY)
    trades = res.trades
    print(f"Sniper cohort: N={len(trades)}")

    # Attach ex-ante gap_vol to each trade.
    rows = []
    for t in trades:
        df = price.get(t.ticker)
        gv = gap_vol_asof(df, t.signal_date) if df is not None else None
        rows.append({"ticker": t.ticker, "date": t.signal_date, "pnl": t.pnl_pct,
                     "regime": t.regime, "gap_vol": gv})
    scored = [r for r in rows if r["gap_vol"] is not None]
    print(f"  with gap_vol: {len(scored)}/{len(rows)}")

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

    # 2) Ex-ante filter: drop the gappiest names above a threshold. Test a few
    #    thresholds; report BASELINE vs filtered + per-year stability.
    base_n, base_wr, base_avg, base_w5 = _stats([r["pnl"] for r in scored])
    print(f"\nBASELINE (all with gap_vol): N={base_n} WR={base_wr:.1%} "
          f"avg={base_avg:+.3f}% worst5%={base_w5:.0f}%")
    gvs = sorted(r["gap_vol"] for r in scored)
    print("\n-- filter: drop entries with gap_vol > threshold --")
    print(f"{'thresh(pctile)':<16}{'kept':>6}{'dropped':>8}{'WR':>7}{'avg%':>8}"
          f"{'worst5%':>10}{'yr1':>8}{'yr2':>8}{'yr3':>8}")
    lo = min(r["date"] for r in scored)
    years = [(date(lo.year+k, lo.month, lo.day), date(lo.year+k+1, lo.month, lo.day)) for k in range(3)]
    for pctile in [0.95, 0.90, 0.80, 0.70]:
        thr = gvs[int(len(gvs) * pctile)]
        kept = [r for r in scored if r["gap_vol"] <= thr]
        n, wr, avg, w5 = _stats([r["pnl"] for r in kept])
        yr_avgs = []
        for a, b in years:
            yp = [r["pnl"] for r in kept if a <= r["date"] < b]
            yr_avgs.append(mean(yp) if yp else 0.0)
        print(f"{f'p{int(pctile*100)} ({thr:.1f}%)':<16}{n:>6}{base_n-n:>8}{wr:>7.1%}"
              f"{avg:>8.3f}{w5:>10.1f}" + "".join(f"{y:>8.3f}" for y in yr_avgs))
    print("\n(baseline per-year avg for reference:", end=" ")
    for a, b in years:
        yp = [r["pnl"] for r in scored if a <= r["date"] < b]
        print(f"{mean(yp) if yp else 0:.3f}", end=" ")
    print(")")


if __name__ == "__main__":
    main()
