#!/usr/bin/env python
"""Analyze win rate and expectancy by day-of-week from trade CSVs.

Reads trade logs from the V1.2 backtest and computes per-weekday statistics
to determine if certain entry days should be blocked.

Usage:
    python scripts/analyze_weekday.py outputs/research/backtest_baseline_1Y_trades.csv
    python scripts/analyze_weekday.py outputs/research/phase2/phase2_champion_1Y_trades.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def analyze(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} trades from {csv_path}\n")

    # Parse entry_date and extract weekday
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["weekday"] = df["entry_date"].dt.dayofweek
    df["weekday_name"] = df["entry_date"].dt.day_name()
    df["win"] = df["pnl_pct"] > 0

    # Per-weekday stats
    print(f"{'Day':12s} {'Trades':>7s} {'WR':>7s} {'Avg PnL':>9s} {'Med PnL':>9s} {'Expectancy':>11s} {'Sharpe':>7s}")
    print("-" * 70)

    weekday_stats = []
    for wd in range(5):
        subset = df[df["weekday"] == wd]
        if subset.empty:
            continue
        name = subset["weekday_name"].iloc[0]
        n = len(subset)
        wr = subset["win"].mean()
        avg_pnl = subset["pnl_pct"].mean()
        med_pnl = subset["pnl_pct"].median()
        expectancy = avg_pnl  # same as avg for single-unit trades
        std = subset["pnl_pct"].std()
        sharpe = avg_pnl / std if std > 0 else 0

        weekday_stats.append({
            "weekday": wd,
            "name": name,
            "trades": n,
            "win_rate": wr,
            "avg_pnl": avg_pnl,
            "median_pnl": med_pnl,
            "sharpe": sharpe,
        })

        print(f"{name:12s} {n:7d} {wr:7.1%} {avg_pnl:+8.3f}% {med_pnl:+8.3f}% {expectancy:+10.3f}% {sharpe:7.3f}")

    # Statistical significance test (chi-squared on WR)
    if len(weekday_stats) >= 2:
        from scipy import stats as sp_stats
        contingency = []
        for ws in weekday_stats:
            wins = int(ws["win_rate"] * ws["trades"])
            losses = ws["trades"] - wins
            contingency.append([wins, losses])
        chi2, p_value, dof, _ = sp_stats.chi2_contingency(contingency)
        print(f"\nChi-squared test for WR independence across weekdays:")
        print(f"  chi2 = {chi2:.2f}, p-value = {p_value:.4f}, dof = {dof}")
        if p_value < 0.05:
            print("  Result: SIGNIFICANT — weekday effect exists")
        else:
            print("  Result: NOT significant — no reliable weekday effect")

    # Recommendation
    if weekday_stats:
        worst = min(weekday_stats, key=lambda x: x["win_rate"])
        best = min(weekday_stats, key=lambda x: -x["win_rate"])
        gap = best["win_rate"] - worst["win_rate"]
        print(f"\n  Best day:  {best['name']} (WR={best['win_rate']:.1%})")
        print(f"  Worst day: {worst['name']} (WR={worst['win_rate']:.1%})")
        print(f"  Gap: {gap:.1%}")
        if gap >= 0.03:
            print(f"  Recommendation: Consider blocking {worst['name']} entries (weekday={worst['weekday']})")
        else:
            print(f"  Recommendation: Gap too small to justify weekday filter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day-of-week trade analysis")
    parser.add_argument("csv_path", help="Path to trade CSV file")
    args = parser.parse_args()
    analyze(args.csv_path)
