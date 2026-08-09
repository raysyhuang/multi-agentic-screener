"""Does taking MORE sniper picks per day beat taking better ones?

Follow-up to rank_quality_FINDINGS.md, which showed the top-2-of-N choice is
close to random (ranker captures ~7% of available selection value) while summed
per-trade P&L nearly doubled from k=2 to k=6. That sum column ignores capital
entirely — a real account has slots, and a wider daily quota fills them faster
and starts SKIPPING signals. This replays each k through the same
concurrency-capped equity simulator the dashboard uses, so the comparison is
against compounded return and drawdown rather than a sum that assumes infinite
capital.

Relative arms only: the 3Y universe under-samples sniper's ATR%>=5 population
(see research-sniper-backtest-universe), so absolutes are not trustworthy — the
RANKING BETWEEN k values on a fixed population is.

Usage:
    python scripts/sniper_pick_count.py [--max-concurrent 3] [--cohort PATH]
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.portfolio import BookTrade, simulate_book

DEFAULT_COHORT = "outputs/research/sniper_truth_E_live_fixed_2026-07-26.csv"


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(["entry_date", "score"], ascending=[True, False]).copy()
    df["rank"] = df.groupby("entry_date").cumcount() + 1
    df["year"] = df["entry_date"].astype(str).str[:4]
    return df


def _book(df: pd.DataFrame) -> list[BookTrade]:
    out = []
    for r in df.itertuples():
        try:
            out.append(BookTrade(entry=date.fromisoformat(str(r.entry_date)[:10]),
                                 exit=date.fromisoformat(str(r.exit_date)[:10]),
                                 pnl_pct=float(r.pnl_pct)))
        except (ValueError, TypeError):
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=DEFAULT_COHORT)
    ap.add_argument("--max-concurrent", type=int, default=3,
                    help="live sniper_max_positions is 3")
    args = ap.parse_args()

    df = _load(Path(args.cohort))
    print(f"cohort {args.cohort}: {len(df)} trades, "
          f"{df['entry_date'].nunique()} entry days\n")

    for cap in sorted({args.max_concurrent, 5, 10}):
        print("=" * 96)
        print(f"CONCURRENCY CAP = {cap} slots"
              f"{'   <- live sniper_max_positions' if cap == args.max_concurrent else ''}")
        print("=" * 96)
        print(f"  {'k':>3}{'signals':>9}{'taken':>7}{'skipped':>9}{'avg/trade':>11}"
              f"{'return':>10}{'maxDD':>9}{'Sharpe':>8}   per-year avg")
        for k in (1, 2, 3, 4, 6, 10):
            sub = df[df["rank"] <= k]
            res = simulate_book(_book(sub), max_concurrent=cap)
            ys = "  ".join(f"{y}:{g['pnl_pct'].mean():+.2f}"
                           for y, g in sub.groupby("year"))
            sharpe = "–" if res["sharpe"] is None else f"{res['sharpe']:.2f}"
            print(f"  {k:>3}{len(sub):>9}{res['taken']:>7}{res['skipped']:>9}"
                  f"{sub['pnl_pct'].mean():>+10.3f}%{res['total_return_pct']:>+9.1f}%"
                  f"{res['max_drawdown_pct']:>8.1f}%{sharpe:>8}   {ys}")
        print()

    # The honest control: does the WIDER quota beat the live one after capital?
    print("=" * 96)
    print("VERDICT INPUT — live cap (3 slots), k=2 vs the best wider k")
    print("=" * 96)
    base = simulate_book(_book(df[df["rank"] <= 2]), max_concurrent=args.max_concurrent)
    for k in (4, 6):
        alt = simulate_book(_book(df[df["rank"] <= k]), max_concurrent=args.max_concurrent)
        d_ret = alt["total_return_pct"] - base["total_return_pct"]
        d_dd = alt["max_drawdown_pct"] - base["max_drawdown_pct"]
        print(f"  k={k}: return {d_ret:+.1f}pp, maxDD {d_dd:+.1f}pp, "
              f"skipped {alt['skipped']} vs {base['skipped']} "
              f"({'BETTER' if d_ret > 0 and d_dd <= 0 else 'mixed/worse'})")
    print(f"\n  note: at {args.max_concurrent} slots the cap, not the quota, is the "
          f"binding constraint once k exceeds it — that is the point of the test.")
    print(f"  base k=2: taken={base['taken']} skipped={base['skipped']} "
          f"return={base['total_return_pct']:+.1f}% maxDD={base['max_drawdown_pct']:.1f}% "
          f"mean/trade={st.mean([t.pnl_pct for t in _book(df[df['rank'] <= 2])]):+.3f}%")


if __name__ == "__main__":
    main()
