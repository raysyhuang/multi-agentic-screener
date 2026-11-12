#!/usr/bin/env python3
"""Check snapshot membership consistency of Queen's shared frozen input pack.

Independent audit by GPT, 2026-11-12. This does NOT audit underlying SEC index
membership or authorize changes to the frozen pack. Stdlib only.

Run from any directory:
    python research-exchange/gpt/check_membership_intervals.py

Prints per-signal member mismatches between membership_intervals.csv and
eligibility_by_signal.csv.gz. Differentiates membership from 365-day
eligibility: a member can be ineligible for trading.
"""
from pathlib import Path
import csv
import gzip
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "shared" / "inputs-pack-full-20261109"


def read(name):
    path = PACK / name
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def audit():
    intervals = list(read("membership_intervals.csv"))
    member = defaultdict(set)
    eligible = defaultdict(set)
    for r in read("eligibility_by_signal.csv.gz"):
        signal = r["signal_date"]
        if r["member_at_signal"] == "True":
            member[signal].add(r["engine_ticker"])
        if r["eligible_final"] == "True":
            eligible[signal].add(r["engine_ticker"])

    mismatch_dates = []
    false_eligible = []
    for day in sorted(member):
        snapshot = {
            r["engine_ticker"] for r in intervals
            if r["start_date"] <= day
            and (not r["end_date_exclusive"] or day < r["end_date_exclusive"])
        }
        missing = sorted(member[day] - snapshot)
        extras = sorted(snapshot - member[day])
        if missing or extras:
            mismatch_dates.append((day, len(member[day]), len(snapshot), missing, extras))
            for ticker in missing:
                if ticker in eligible[day]:
                    false_eligible.append((day, ticker))

    print(f"signals={len(member)}  interval_rows={len(intervals)}")
    print(f"mismatched_signal_days={len(mismatch_dates)}")
    for day, a, b, missing, extra in mismatch_dates:
        print(f"{day}: membership={a}, intervals={b}, missing={missing}, extra={extra}")
    print(f"missing_but_eligible={false_eligible}")
    if false_eligible:
        print("WARNING: these mismatches may affect ranking eligibility.")
    else:
        print("No missing interval-member is eligible_final on checked signals.")
    return len(mismatch_dates)


if __name__ == "__main__":
    audit()
