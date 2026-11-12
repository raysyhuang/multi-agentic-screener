#!/usr/bin/env python3
"""Reproduce historical diagnostics for W10H2K1 and W12H4K1.

Inputs are the tracked Queen answer key and the bundled FMP SPY daily prices.
This is an independent analysis/benchmark script, not a new strategy engine.
"""
from __future__ import annotations

import csv
import gzip
import statistics
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ANS = ROOT / "research-exchange/queen/answer-key-full-20261109"


def read_prices():
    with (HERE / "spy_prices_fmp_2014-2025.csv").open() as f:
        rows = list(csv.DictReader(f))
    return {r["date"]: (float(r["close"]), float(r["adjClose"])) for r in rows}


def on_or_before(prices, d, field):
    d = date.fromisoformat(d)
    while d.isoformat() not in prices:
        d = date.fromordinal(d.toordinal() - 1)
    return prices[d.isoformat()][field]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


prices = read_prices()
with (ANS / "paths_summary.csv").open() as f:
    summary = list(csv.DictReader(f))

target = [r for r in summary if r["K"] == "1" and (r["W"], r["H"]) in {("10", "2"), ("12", "4")}]
phase_rows = []
for r in target:
    start, end = r["entry"], r["end"]
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    spy_price_mult = on_or_before(prices, end, 0) / on_or_before(prices, start, 0)
    spy_total_mult = on_or_before(prices, end, 1) / on_or_before(prices, start, 1)
    phase_rows.append({
        "path_id": r["path_id"], "entry": start, "end": end,
        "strategy_CAGR": r["CAGR"], "strategy_MaxDD": r["daily_MDD"],
        "strategy_final_multiple": r["final_multiple"],
        "SPY_price_CAGR": (spy_price_mult ** (365.25 / days) - 1),
        "SPY_dividend_adjusted_CAGR": (spy_total_mult ** (365.25 / days) - 1),
        "beats_SPY_price": float(r["CAGR"]) > spy_price_mult ** (365.25 / days) - 1,
        "beats_SPY_dividend_adjusted": float(r["CAGR"]) > spy_total_mult ** (365.25 / days) - 1,
    })
write_csv(HERE / "phase_benchmark_comparison.csv", phase_rows)

# Annual path returns, based on the last available daily NAV and SPY close in each year.
annual_rows = []
for w, h in ((10, 2), (12, 4)):
    cols = [f"W{w}H{h}K1P{i}" for i in range(1, 13)]
    by_year = {c: {} for c in cols}
    navfile = ANS / f"nav_daily/nav_daily_W{w}.csv.gz"
    with gzip.open(navfile, "rt") as f:
        for row in csv.DictReader(f):
            yr = int(row["date"][:4])
            for c in cols:
                if row[c]:
                    by_year[c].setdefault(yr, []).append((row["date"], float(row[c])))
    for c in cols:
        for yr in range(2016, 2025):
            prev = by_year[c].get(yr - 1, [])
            cur = by_year[c].get(yr, [])
            if not prev or not cur:
                continue
            sd, sv = prev[-1]
            ed, ev = cur[-1]
            spy_price = on_or_before(prices, ed, 0) / on_or_before(prices, sd, 0) - 1
            spy_total = on_or_before(prices, ed, 1) / on_or_before(prices, sd, 1) - 1
            annual_rows.append({
                "W": w, "H": h, "phase": c[-2:], "year": yr,
                "strategy_return": ev / sv - 1,
                "SPY_price_return": spy_price,
                "SPY_dividend_adjusted_return": spy_total,
                "beats_SPY_price": ev / sv - 1 > spy_price,
                "beats_SPY_dividend_adjusted": ev / sv - 1 > spy_total,
            })
write_csv(HERE / "annual_phase_returns.csv", annual_rows)

for w, h in ((10, 2), (12, 4)):
    x = [r for r in phase_rows if r["path_id"].startswith(f"W{w}H{h}K1")]
    cagrs = [float(r["strategy_CAGR"]) for r in x]
    dds = [float(r["strategy_MaxDD"]) for r in x]
    print(
        f"W{w}H{h}: n={len(x)}, median_CAGR={statistics.median(cagrs):.6%}, "
        f"range=[{min(cagrs):.6%},{max(cagrs):.6%}], "
        f"median_MaxDD={statistics.median(dds):.6%}, "
        f"beats_SPY_price={sum(r['beats_SPY_price'] for r in x)}/12, "
        f"beats_SPY_dividend_adjusted={sum(r['beats_SPY_dividend_adjusted'] for r in x)}/12"
    )
    ys = [r for r in annual_rows if r["W"] == w]
    for yr in range(2016, 2025):
        yy = [r for r in ys if r["year"] == yr]
        if yy:
            med = statistics.median(r["strategy_return"] for r in yy)
            beat = sum(r["beats_SPY_price"] for r in yy)
            print(f"  {yr}: median={med:.3%}; phases beating SPY price={beat}/12")
