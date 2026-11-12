#!/usr/bin/env python3
"""Independent aggregate-output checks for Queen's Taiwan W/H Top3 grid.

Uses only published CSV output files, not Queen's engine or its old /workspace
loader. Thus it verifies aggregation, time-slice arithmetic, phase sensitivity,
and lo/hi classification, NOT raw market-data construction.

Run from any directory:
    python research-exchange/gpt/taiwan_grid_output_audit.py

Stdlib only. Outputs concise, human-readable review diagnostics.
Research use only.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent.parent
PACK = HERE / "queen" / "taiwan-wh-grid-20261112"


def rows(filename):
    with (PACK / filename).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def key(row):
    return row["univ"], int(row["W"]), int(row["H"]), int(row["M"]), row["window"]


def subset(records, pool, w, h, window):
    return sorted(
        (r for r in records.values()
         if r["univ"] == pool and int(r["W"]) == w
         and int(r["H"]) == h and r["window"] == window),
        key=lambda x: int(x["M"]),
    )


def rel(row):
    return float(row["tr_mult"]) / float(row["b0050_tr_mult"])


def cohort(records, pool, w, h, phase=12):
    out = {}
    for window in ("full", "10y", "5y"):
        r = records[pool, w, h, phase, window]
        out[window] = r
    return out


def segments(records, pool, w, h):
    r = cohort(records, pool, w, h)
    assert len({r[k]["last_end"] for k in r}) == 1, "non-nested exit date"
    before = rel(r["full"]) / rel(r["10y"])
    mid = rel(r["10y"]) / rel(r["5y"])
    recent = rel(r["5y"])
    return before, mid, recent


def describe(records, pool, w, h):
    print(f"\n{pool} W{w}H{h} / December phase")
    earlier, middle, recent = segments(records, pool, w, h)
    print(f"  Pre-2016 relative to 0050: {earlier:.4f}x")
    print(f"  2016–2020 relative to 0050: {middle:.4f}x")
    print(f"  2021–2025 relative to 0050: {recent:.4f}x")
    for window in ("5y", "10y", "full"):
        ss = subset(records, pool, w, h, window)
        dec = next(r for r in ss if int(r["M"]) == 12)
        values = [rel(x) for x in ss]
        print(
            f"  {window:4s}: Dec TR={float(dec['tr_mult']):.4f}x, "
            f"0050 TR={float(dec['b0050_tr_mult']):.4f}x, "
            f"win-months={sum(x > 1 for x in values)}/12, "
            f"median relative={median(values):.4f}x, "
            f"min relative={min(values):.4f}x, "
            f"Dec MDD={float(dec['tr_maxdd']):.2%}"
        )


def main():
    lo = {key(r): r for r in rows("grid_summary_lo.csv")}
    hi = {key(r): r for r in rows("grid_summary_hi.csv")}
    concentration = {
        (r["univ"], int(r["W"]), int(r["H"]), r["window"]): r
        for r in rows("concentration_check_lo.csv")
    }
    assert len(lo) == 15_552 and len(hi) == 15_552
    assert len({k[:4] for k in lo}) == 5_184
    assert lo.keys() == hi.keys()

    flip = [
        k for k in lo
        if (rel(lo[k]) > 1) != (rel(hi[k]) > 1)
    ]
    assert len(flip) == 36, len(flip)
    assert all(k[4] == "full" for k in flip)
    december_flips = [k for k in flip if k[3] == 12]
    assert december_flips == [("TW150",5,4,12,"full")], december_flips
    print(f"rows per bound=15,552; unique grid paths=5,184")
    print(f"lo/hi benchmark flips={len(flip)}; December-phase flips={december_flips}")

    for pool in ("TW150","MC100"):
        for w,h in ((6,6),(9,4),(9,3),(7,4),(2,4)):
            describe(lo, pool, w, h)
            c = concentration.get((pool,w,h,"full"))
            if c:
                print(f"  Full log excess after removing best two periods: "
                      f"{float(c['log_excess_ex_top2']):+.2f}")

    # A deliberately restricted retrospective temporal cross-validation:
    # Only H dividing 12 ensures identical Dec-anchored period boundaries
    # across the 5y/10y/full nested windows. NOT true forward out-of-sample:
    # rules and selection criterion were evaluated after seeing the data.
    surviving = []
    for w in range(1,13):
        for h in (1,2,3,4,6,12):
            if all(
                all(value > 1 for value in segments(lo, pool, w, h))
                for pool in ("TW150","MC100")
            ):
                surviving.append((w,h))
    print("\nRetrospective cross-universe, three-era survivors (Dec phase):")
    print(surviving)
    assert surviving == [(2,4)], surviving
    print("\nMethod caveats: provisional membership, FinMind-only prices, "
          "dividend/CR bounds, phase dependence, selection bias, "
          "no raw-price replay from this script.")


if __name__ == "__main__":
    main()
