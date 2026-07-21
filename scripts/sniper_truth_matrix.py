"""Sniper truth matrix — measure the TRUE, live-faithful sniper expectancy.

The frozen sniper backtest (692 trades, 82% WR, +4.13% avg) looked excellent,
but live forward tracking was much worse. Investigation found the backtest ran
a *different, more optimistic* strategy than production:

  * spy_df was never passed to score_sniper live (relative_strength pinned at 50)
  * sniper_min_score=70 was never enforced live (internal floor was 60/65)
  * live has a 1-day sniper time_stop that the backtest never modeled
  * the backtest filled every stop at the exact stop price (no gap-through-open)

This script runs the same sniper scan through the now-unified exit engine under
five configurations that isolate each divergence, so we can attribute the gap
and read off the true go-forward expectancy:

  A  baseline    spy=on  min70  no-timestop  no-gap    -> reproduce the ~82% number
  B  +fill       spy=on  min70  no-timestop  gap       -> cost of realistic fills
  C  +timestop   spy=on  min70  1d-timestop  gap       -> cost of the live time_stop
  D  live-as-was spy=OFF min60  1d-timestop  gap       -> should land near live ~69%
  E  live-fixed  spy=on  min70  1d-timestop  gap       -> TRUE expectancy after the 1a fixes

Run E is the strategy production now runs after the Phase-1a bug fixes. Each run
also gets a concurrency-capped equity curve (per-trade sums are not an account
multiple) so total return reflects a real account, not unlimited concurrency.

Usage:
  python scripts/sniper_truth_matrix.py                 # full SP500, 3Y (yfinance, cached)
  python scripts/sniper_truth_matrix.py --years 3 --limit 60   # faster smoke run
  python scripts/sniper_truth_matrix.py --max-concurrent 3     # match sniper_max_positions
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.research.signal_backtest import (
    fetch_ohlcv,
    run_model_backtest,
)
from src.research.sp500_tickers import SP500_TICKERS

# Reuse the concurrency-capped equity-curve engine.
import scripts.sniper_equity_curve as eq

# Frozen V3 sniper params, shared by every run. Per-run toggles are layered on top.
V3_BASE = {
    "atr_pct_floor": 5.0,
    "stop_atr_mult": 1.5,
    "target_atr_mult": 3.0,
    "holding_period": 7,
    "trail_activate_pct": 1.0,
    "trail_distance_pct": 0.5,
}

# The five configurations. `min_score` 60 with use_spy off reproduces the live
# bug (internal 60/65 floors apply); 70 matches the backtest baseline gate.
RUNS = [
    ("A_baseline",    dict(use_spy=True,  min_score=70, sniper_time_stop_days=0, gap_through=False)),
    ("B_fill",        dict(use_spy=True,  min_score=70, sniper_time_stop_days=0, gap_through=True)),
    ("C_timestop",    dict(use_spy=True,  min_score=70, sniper_time_stop_days=1, gap_through=True)),
    ("D_live_as_was", dict(use_spy=False, min_score=60, sniper_time_stop_days=1, gap_through=True)),
    ("E_live_fixed",  dict(use_spy=True,  min_score=70, sniper_time_stop_days=1, gap_through=True)),
]

# Working assumption for the observed live win rate (see MEMORY: ~0.69 haircut).
LIVE_WR_OBSERVED = 0.69


def _trades_to_equity_trades(trades) -> list[eq.Trade]:
    """Map backtest Trade objects to the equity-curve script's Trade dataclass."""
    return [
        eq.Trade(
            ticker=t.ticker,
            entry=t.entry_date,
            exit=t.exit_date,
            pnl_pct=t.pnl_pct,
            regime=t.regime,
            score=t.score,
        )
        for t in trades
    ]


def _write_trades_csv(trades, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "entry_date", "exit_date", "pnl_pct", "regime", "score", "exit_reason"])
        for t in trades:
            w.writerow([t.ticker, t.entry_date, t.exit_date, round(t.pnl_pct, 4), t.regime, t.score, t.exit_reason])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0, help="cap universe size for a fast smoke run (0 = full SP500)")
    ap.add_argument("--max-concurrent", type=int, default=3, help="equity-curve position cap (sniper_max_positions=3)")
    ap.add_argument("--fraction", type=float, default=0.20, help="fixed-fraction sizing per position")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache-file", default="", help="load a pre-downloaded OHLCV parquet (columns _ticker,date,open,high,low,close,volume) directly, bypassing yfinance")
    ap.add_argument("--out-dir", default="outputs/research")
    args = ap.parse_args()

    if args.cache_file:
        import pandas as pd
        print(f"Loading price data from {args.cache_file} ...")
        combined = pd.read_parquet(args.cache_file)
        price_data = {
            ticker: grp.drop(columns=["_ticker"]).reset_index(drop=True)
            for ticker, grp in combined.groupby("_ticker")
        }
        if args.limit:
            keep = set(list(SP500_TICKERS)[: args.limit]) | {"SPY"}
            price_data = {t: d for t, d in price_data.items() if t in keep}
    else:
        tickers = list(SP500_TICKERS)
        if args.limit:
            tickers = tickers[: args.limit]
        if "SPY" not in tickers:
            tickers.append("SPY")
        print(f"Fetching {len(tickers)} tickers × {args.years}Y (yfinance, cached)...")
        price_data = fetch_ohlcv(tickers, years=args.years, no_cache=args.no_cache)
    print(f"Loaded {len(price_data)} tickers with price data")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")

    summary: dict = {"generated": stamp, "years": args.years, "universe": len(price_data),
                     "max_concurrent": args.max_concurrent, "fraction": args.fraction, "runs": {}}

    # NOTE: eqDD% is the real account-equity max drawdown (from the
    # concurrency-capped curve). The PerformanceMetrics.max_drawdown_pct is a
    # drawdown on the SUM of per-trade returns and can exceed 100% — it is not
    # an account drawdown and is deliberately not shown here.
    header = (f"{'run':<14}{'N':>6}{'WR':>8}{'avg%':>8}{'expect%':>9}{'PF':>6}"
              f"{'Sharpe':>8}{'equity×':>9}{'CAGR%':>8}{'eqDD%':>7}{'skip':>7}")
    rows = [header, "-" * len(header)]

    for name, toggles in RUNS:
        params = {**V3_BASE, **toggles}
        result = run_model_backtest("sniper", price_data, params)
        m = result.metrics

        # Concurrency-capped equity curve (real account, not a per-trade sum).
        eq_trades = _trades_to_equity_trades(result.trades)
        if eq_trades:
            eq_res = eq.simulate(
                eq_trades, mode="fixed_fraction", fraction=args.fraction,
                max_concurrent=args.max_concurrent, start_capital=100_000.0,
            )
        else:
            eq_res = None

        _write_trades_csv(result.trades, out_dir / f"sniper_truth_{name}_{stamp}.csv")

        rows.append(
            f"{name:<14}{m.total_trades:>6}{m.win_rate:>8.1%}{m.avg_return_pct:>8.2f}"
            f"{m.expectancy:>9.3f}{m.profit_factor:>6.2f}{m.sharpe_ratio:>8.2f}"
            f"{(eq_res.multiple if eq_res else 0):>9.2f}{(eq_res.cagr_pct if eq_res else 0):>8.1f}"
            f"{(eq_res.max_drawdown_pct if eq_res else 0):>7.1f}"
            f"{(eq_res.skipped_saturation if eq_res else 0):>7}"
        )

        summary["runs"][name] = {
            "toggles": toggles,
            "metrics": asdict(m),
            "by_regime": {r: asdict(v) for r, v in result.by_regime.items()},
            "by_exit_reason": result.by_exit_reason,
            "equity_curve": {
                "multiple": eq_res.multiple if eq_res else None,
                "cagr_pct": eq_res.cagr_pct if eq_res else None,
                "max_drawdown_pct": eq_res.max_drawdown_pct if eq_res else None,
                "taken": eq_res.taken if eq_res else 0,
                "skipped_saturation": eq_res.skipped_saturation if eq_res else 0,
                "peak_concurrent": eq_res.peak_concurrent if eq_res else 0,
            } if eq_res else None,
        }

    print("\n" + "\n".join(rows))

    # Sanity: Run D (live-as-was) reconstructs the strategy production actually
    # ran before the Phase-1a fixes. The ~0.69 figure is an UNVERIFIED working
    # assumption (the equity-curve haircut default), not a measured live WR, so
    # treat a mismatch as information about that assumption, not proof of a bug.
    d_wr = summary["runs"]["D_live_as_was"]["metrics"]["win_rate"]
    print(f"\nSanity — Run D (live-as-was) WR = {d_wr:.1%} vs assumed live ~{LIVE_WR_OBSERVED:.0%} (unverified)")
    if abs(d_wr - LIVE_WR_OBSERVED) <= 0.06:
        print("  Run D lands near the assumed live WR — gap explained.")
    else:
        print("  Run D is below the assumed live WR: either the ~69% assumption was")
        print("  optimistic (small sample) or an unmodeled divergence remains. Confirm")
        print("  against the measured live scorecard (scripts/sniper_scorecard.py).")

    e = summary["runs"]["E_live_fixed"]["metrics"]
    print(f"\nTRUE go-forward expectancy (Run E): WR={e['win_rate']:.1%}  "
          f"avg={e['avg_return_pct']:+.2f}%  expectancy={e['expectancy']:+.3f}%  "
          f"payoff={e['payoff_ratio']:.2f}  Sharpe={e['sharpe_ratio']:.2f}")

    out_json = out_dir / f"sniper_truth_matrix_{stamp}.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
