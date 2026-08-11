"""Rank-IC analysis of sniper score components vs forward returns.

Answers: do the 5 sniper components (bb_squeeze, vol_compression,
relative_strength, trend_alignment, momentum_base) actually rank forward
returns among emitted signals — and is the 30/25/20/15/10 weighting justified?

Method:
  - Point-in-time scan over S&P 500 (same harness as signal_backtest.py),
    signal-date regime map, allow_bear=True so bear rows are included but
    tagged for subsetting. Wide gates (atr_pct_floor=3.5, min_score=0 at
    scan level -> effective floor 60/65 inside score_sniper) to maximize
    score variance; the production subset (score>=70, atr_pct>=5) is
    reported separately.
  - Targets: raw forward returns from T+1 open to close at +3/+5/+7/+10
    trading days (no stops — cleaner for component validation), plus
    as-traded PnL under frozen V3 exits (stop 1.5xATR, target 3xATR,
    hold 7, trail 1.0/0.5).
  - Pooled Spearman IC (with p-values), daily cross-sectional mean IC
    (days with >=8 signals — robust to time clustering), per-regime ICs,
    component pairwise correlations, and composite quintile spread.

Caveat: pooled p-values are optimistic because overlapping holding windows
cluster observations in time; treat the daily cross-sectional IC as primary.

Usage:
    python scripts/sniper_component_ic.py --years 3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.signal_backtest import (  # noqa: E402
    MARKET_DATA_ONLY_TICKERS,
    build_signal_date_regime_map,
    fetch_ohlcv,
    get_sp500_tickers,
    scan_sniper,
    simulate_trade,
)

COMPONENTS = [
    "bb_squeeze",
    "vol_compression",
    "relative_strength",
    "trend_alignment",
    "momentum_base",
]
FWD_HORIZONS = [3, 5, 7, 10]
MARKET_TICKERS = ["SPY", "QQQ", "^VIX", "^TNX", "^IRX"]

# Frozen sniper V3 exit params (see MEMORY.md "Sniper Track")
V3_STOP_ATR = 1.5
V3_TARGET_ATR = 3.0
V3_MAX_HOLD = 7
V3_TRAIL_ACTIVATE = 1.0
V3_TRAIL_DISTANCE = 0.5


def collect_signals(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Scan all tickers point-in-time and attach forward returns per signal."""
    regime_map = build_signal_date_regime_map(price_data)
    if not regime_map:
        raise RuntimeError("Signal-date regime map is empty — SPY/QQQ missing?")

    spy_df = price_data.get("SPY")
    rows: list[dict] = []
    tickers = [t for t in price_data if t not in MARKET_DATA_ONLY_TICKERS and t != "SPY"]

    for n, ticker in enumerate(tickers, 1):
        df = price_data[ticker]
        signals = scan_sniper(
            ticker,
            df,
            spy_df=spy_df,
            signal_regime_by_date=regime_map,
            min_score=0.0,          # keep everything score_sniper emits (floor 60/65)
            atr_pct_floor=3.5,      # wide gate; prod (5.0) reported as a subset
            stop_atr_mult=V3_STOP_ATR,
            target_atr_mult=V3_TARGET_ATR,
            holding_period=V3_MAX_HOLD,
            allow_bear=True,        # include bear rows, tagged via regime map
        )
        if n % 50 == 0:
            print(f"  scanned {n}/{len(tickers)} tickers, {len(rows)} signals so far")

        for signal_date, sig in signals:
            future = df[df["date"] > signal_date].sort_values("date")
            if len(future) < 2:
                continue
            entry_open = float(future.iloc[0]["open"])
            if entry_open <= 0:
                continue

            row: dict = {
                "ticker": ticker,
                "signal_date": signal_date,
                "regime": regime_map.get(signal_date, "unknown"),
                "score": sig.score,
                **{c: sig.components.get(c) for c in COMPONENTS},
            }
            # Recover ATR from the stop distance (stop = close - 1.5*ATR)
            atr = (sig.entry_price - sig.stop_loss) / V3_STOP_ATR
            row["atr_pct"] = round(atr / sig.entry_price * 100, 3) if sig.entry_price else None

            for h in FWD_HORIZONS:
                if len(future) > h:
                    row[f"fwd_{h}d"] = float(future.iloc[h]["close"]) / entry_open - 1
                else:
                    row[f"fwd_{h}d"] = np.nan

            trade = simulate_trade(
                df,
                signal_date,
                stop_loss=sig.stop_loss,
                target=sig.target_1,
                max_hold=V3_MAX_HOLD,
                trail_activate_pct=V3_TRAIL_ACTIVATE,
                trail_distance_pct=V3_TRAIL_DISTANCE,
            )
            row["trade_pnl_pct"] = trade["pnl_pct"] if trade else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def pooled_ic(df: pd.DataFrame, factor: str, target: str) -> dict:
    sub = df[[factor, target]].dropna()
    if len(sub) < 5 or sub[factor].nunique() <= 1 or sub[target].nunique() <= 1:
        return {"ic": None, "p": None, "n": len(sub)}
    ic, p = spearmanr(sub[factor], sub[target])
    return {"ic": round(float(ic), 4), "p": round(float(p), 4), "n": len(sub)}


def daily_cross_sectional_ic(
    df: pd.DataFrame, factor: str, target: str, min_per_day: int = 8
) -> dict:
    """Mean of per-day Spearman ICs across days with enough breadth."""
    ics = []
    for _, day in df.dropna(subset=[factor, target]).groupby("signal_date"):
        if len(day) < min_per_day:
            continue
        if day[factor].nunique() <= 1 or day[target].nunique() <= 1:
            continue
        ic, _ = spearmanr(day[factor], day[target])
        if np.isfinite(ic):
            ics.append(ic)
    if len(ics) < 5:
        return {"mean_ic": None, "t_stat": None, "n_days": len(ics)}
    arr = np.array(ics)
    t = float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))) if arr.std(ddof=1) > 0 else None
    return {
        "mean_ic": round(float(arr.mean()), 4),
        "t_stat": round(t, 2) if t is not None else None,
        "n_days": len(arr),
    }


def quintile_spread(df: pd.DataFrame, factor: str, target: str) -> dict:
    sub = df[[factor, target]].dropna()
    if len(sub) < 25 or sub[factor].nunique() < 5:
        return {"q5_mean": None, "q1_mean": None, "spread": None}
    q = pd.qcut(sub[factor].rank(method="first"), 5, labels=False)
    q5 = float(sub.loc[q == 4, target].mean())
    q1 = float(sub.loc[q == 0, target].mean())
    return {
        "q5_mean": round(q5 * 100, 3),
        "q1_mean": round(q1 * 100, 3),
        "spread": round((q5 - q1) * 100, 3),
    }


def analyze(df: pd.DataFrame) -> dict:
    factors = COMPONENTS + ["score"]
    targets = [f"fwd_{h}d" for h in FWD_HORIZONS] + ["trade_pnl_pct"]

    report: dict = {
        "n_signals": len(df),
        "date_range": [str(df["signal_date"].min()), str(df["signal_date"].max())],
        "regime_counts": df["regime"].value_counts().to_dict(),
        "pooled_ic": {},
        "daily_cs_ic": {},
        "prod_subset": {},
        "per_regime_ic_fwd7": {},
        "component_correlation": {},
        "quintile_spread_fwd7": {},
        "component_variance": {},
    }

    for f in factors:
        report["pooled_ic"][f] = {t: pooled_ic(df, f, t) for t in targets}
        report["daily_cs_ic"][f] = daily_cross_sectional_ic(df, f, "fwd_7d")
        report["quintile_spread_fwd7"][f] = quintile_spread(df, f, "fwd_7d")
        report["component_variance"][f] = {
            "nunique": int(df[f].nunique()),
            "std": round(float(df[f].std()), 2),
        }

    prod = df[(df["score"] >= 70) & (df["atr_pct"] >= 5.0)]
    report["prod_subset"]["n"] = len(prod)
    if len(prod) >= 25:
        report["prod_subset"]["ic_fwd7"] = {f: pooled_ic(prod, f, "fwd_7d") for f in factors}
        report["prod_subset"]["ic_trade_pnl"] = {
            f: pooled_ic(prod, f, "trade_pnl_pct") for f in factors
        }

    for regime, grp in df.groupby("regime"):
        if len(grp) >= 25:
            report["per_regime_ic_fwd7"][regime] = {
                f: pooled_ic(grp, f, "fwd_7d") for f in factors
            }

    corr = df[COMPONENTS].corr(method="spearman").round(3)
    report["component_correlation"] = corr.to_dict()

    return report


def print_report(report: dict) -> None:
    print("\n" + "=" * 74)
    print("SNIPER COMPONENT IC REPORT")
    print("=" * 74)
    print(f"Signals: {report['n_signals']}  range: {report['date_range']}")
    print(f"Regimes: {report['regime_counts']}")

    print("\n-- Pooled Spearman IC (factor vs target) --")
    header = f"{'factor':<20}" + "".join(
        f"{t:>14}" for t in ["fwd_3d", "fwd_5d", "fwd_7d", "fwd_10d", "trade_pnl"]
    )
    print(header)
    for f, targets in report["pooled_ic"].items():
        cells = []
        for t in ["fwd_3d", "fwd_5d", "fwd_7d", "fwd_10d", "trade_pnl_pct"]:
            r = targets[t]
            cells.append(
                f"{r['ic']:+.3f}({r['p']:.2f})" if r["ic"] is not None else "n/a"
            )
        print(f"{f:<20}" + "".join(f"{c:>14}" for c in cells))

    print("\n-- Daily cross-sectional IC vs fwd_7d (primary metric) --")
    for f, r in report["daily_cs_ic"].items():
        if r["mean_ic"] is not None:
            print(f"  {f:<20} mean_ic={r['mean_ic']:+.4f}  t={r['t_stat']}  days={r['n_days']}")
        else:
            print(f"  {f:<20} insufficient breadth (days={r['n_days']})")

    print("\n-- Composite quintile spread, fwd_7d (Q5 - Q1, pct) --")
    for f, r in report["quintile_spread_fwd7"].items():
        if r["spread"] is not None:
            print(f"  {f:<20} Q5={r['q5_mean']:+.2f}%  Q1={r['q1_mean']:+.2f}%  spread={r['spread']:+.2f}pp")

    print(f"\n-- Production subset (score>=70 & atr_pct>=5): n={report['prod_subset']['n']} --")
    if "ic_fwd7" in report["prod_subset"]:
        for f, r in report["prod_subset"]["ic_fwd7"].items():
            print(f"  {f:<20} ic_fwd7={r['ic']:+.3f} (p={r['p']:.2f}, n={r['n']})")

    print("\n-- Per-regime pooled IC vs fwd_7d --")
    for regime, factors in report["per_regime_ic_fwd7"].items():
        line = ", ".join(
            f"{f}={r['ic']:+.3f}" for f, r in factors.items() if r["ic"] is not None
        )
        print(f"  {regime}: {line}")

    print("\n-- Component variance (step-function granularity) --")
    for f, v in report["component_variance"].items():
        print(f"  {f:<20} distinct_values={v['nunique']}  std={v['std']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0, help="limit tickers (0 = all)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out-dir", default="outputs/research")
    args = parser.parse_args()

    tickers = get_sp500_tickers()
    if args.limit:
        tickers = tickers[: args.limit]
    fetch_list = sorted(set(tickers) | set(MARKET_TICKERS))

    price_data = fetch_ohlcv(fetch_list, years=args.years, no_cache=args.no_cache)
    print(f"Scanning {len(price_data)} tickers for sniper signals (point-in-time)...")

    df = collect_signals(price_data)
    if df.empty:
        print("No signals collected — nothing to analyze.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    csv_path = out_dir / f"sniper_component_ic_signals_{stamp}.csv"
    json_path = out_dir / f"sniper_component_ic_report_{stamp}.json"

    df.to_csv(csv_path, index=False)
    print(f"Signal-level data saved to {csv_path} ({len(df)} rows)")

    report = analyze(df)
    json_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report saved to {json_path}")

    print_report(report)


if __name__ == "__main__":
    main()
