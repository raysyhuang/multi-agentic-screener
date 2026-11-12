#!/usr/bin/env python3
"""GPT independently authored ranking implementation.

Consumes only shared/inputs-pack-full-20261109, never imports Queen's engine or
reference_rebuild.py. Outputs 1,584 month-end W1..12 rankings and the W10/H2,
W12/H4, W12/H12 phase-specific Top1 selections. Research-only, no live trades.

Usage from repository root:
    python research-exchange/gpt/independent_ranker.py

All annual and rolling signals are point-in-time relative to the supplied,
provisional membership and corporate-action data. Inputs are not an independently
audited second data vendor.
"""
import csv
import gzip
import datetime as dt
import pathlib
from collections import defaultdict

OUT = pathlib.Path(__file__).resolve().parent
PACK = OUT.parent / "shared" / "inputs-pack-full-20261109"


def load_table(name):
    opener = gzip.open if name.endswith(".gz") else open
    with opener(PACK / name, "rt", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def month_end_dates():
    dates = defaultdict(list)
    for row in load_table("trading_sessions.csv"):
        day = row["session"]
        dates[day[:7]].append(day)
    return sorted(max(v) for v in dates.values())


def build_rankings():
    months = month_end_dates()
    month_idx = {day: i for i, day in enumerate(months)}
    eligible = defaultdict(set)
    for row in load_table("eligibility_by_signal.csv.gz"):
        if row["eligible_final"] == "True":
            eligible[row["signal_date"]].add(row["engine_ticker"])

    prices = {}
    for row in load_table("rank_prices_month_end.csv.gz"):
        prices[row["engine_ticker"], row["month_end"]] = float(row["rank_price"])

    corrections = defaultdict(list)
    for row in load_table("rank_price_overrides.csv"):
        corrections[row["engine_ticker"], row["month_end"]].append(
            (row["signal_from"], row["signal_to"], float(row["rank_price"]))
        )

    actions = defaultdict(list)
    for row in load_table("corporate_actions_applied.csv"):
        actions[row["engine_ticker"]].append(row)

    def price(ticker, at_date, signal):
        for frm, through, adjusted in corrections.get((ticker, at_date), []):
            if frm <= signal <= through:
                return adjusted
        return prices.get((ticker, at_date))

    all_rankings = {}
    for signal in sorted(eligible):
        idx = month_idx[signal]
        for w in range(1, 13):
            if idx < w:
                continue
            earlier = months[idx-w]
            winners = []
            invalid = []
            for ticker in eligible[signal]:
                begin = price(ticker, earlier, signal)
                end = price(ticker, signal, signal)
                multiplier = 1.0
                reason = None
                for action in actions[ticker]:
                    kind = action["kind"]
                    when = action["ex_date"]
                    if kind == "legacy_data_error_BLOCKER":
                        if action["blocker_from"] <= signal <= action["blocker_to"]:
                            reason = "legacy_data_error"
                    elif kind == "unresolved_BLOCKER" and earlier < when <= signal:
                        reason = "unresolved_event"
                    elif kind in ("spin", "special_div") and earlier < when <= signal:
                        if not action["factor_F"]:
                            reason = "missing_action_factor"
                        else:
                            multiplier *= float(action["factor_F"])
                if not begin or not end or begin <= 0 or end <= 0:
                    reason = reason or "missing_price"
                if reason:
                    invalid.append((ticker, reason))
                else:
                    winners.append((ticker, end / (begin * multiplier) - 1.0))
            winners.sort(key=lambda t: (-t[1], t[0]))
            all_rankings[signal, w] = {
                "eligible": len(eligible[signal]),
                "top": winners,
                "invalid": sorted(invalid),
            }
    return all_rankings


def write_csv(path, rows, cols):
    with (OUT / path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def choose_monthly_paths(rankings):
    month_days = defaultdict(list)
    for r in load_table("trading_sessions.csv"):
        date = r["session"]
        month_days[date[:7]].append(date)
    first = {m: min(days) for m, days in month_days.items()}
    last_days = sorted(rankings)
    rows = []
    for w, h in ((10, 2), (12, 4), (12, 12)):
        for phase in range(1, 13):
            for j in range(0, 120, h):
                monthnum = (phase - 1) + j
                year = 2015 + monthnum // 12
                month = monthnum % 12 + 1
                trade = first[f"{year:04d}-{month:02d}"]
                previous_monthnum = (year * 12 + month - 1) - 1
                prevyear, prevmonth0 = divmod(previous_monthnum, 12)
                signal = max(d for d in month_days[f"{prevyear:04d}-{prevmonth0+1:02d}"])
                candidate = rankings[signal, w]
                picked = candidate["top"][:3] if not candidate["invalid"] else []
                rows.append({
                    "config": f"W{w}H{h}K1", "phase": phase,
                    "signal_date": signal, "trade_date": trade,
                    "status": "UNKNOWN" if candidate["invalid"] else "RANKED",
                    "top1": picked[0][0] if picked else "",
                    "top1_lookback_return": format(picked[0][1], ".15g") if picked else "",
                    "unrankable_count": len(candidate["invalid"]),
                })
    return rows


if __name__ == "__main__":
    rankings = build_rankings()
    ranked_rows = []
    for (signal, w), result in sorted(rankings.items()):
        tops = result["top"][:3] if not result["invalid"] else []
        record = {
            "signal_date": signal, "W": w,
            "eligible_count": result["eligible"],
            "unrankable_count": len(result["invalid"]),
            "unrankable_names": "|".join(x[0] for x in result["invalid"]),
            "status": "UNKNOWN" if result["invalid"] else "RANKED",
        }
        for rank in range(1, 4):
            record[f"top{rank}"] = tops[rank-1][0] if len(tops) >= rank else ""
            record[f"ret{rank}"] = (
                format(tops[rank-1][1], ".15g") if len(tops) >= rank else ""
            )
        ranked_rows.append(record)
    cols = ["signal_date", "W", "eligible_count", "unrankable_count",
            "unrankable_names", "status", "top1", "ret1", "top2", "ret2",
            "top3", "ret3"]
    write_csv("independent_rankings_1584.csv", ranked_rows, cols)

    picks = choose_monthly_paths(rankings)
    write_csv("independent_k1_monthly_picks.csv", picks,
              ["config", "phase", "signal_date", "trade_date", "status",
               "top1", "top1_lookback_return", "unrankable_count"])
    unknown = sum(x["status"] == "UNKNOWN" for x in ranked_rows)
    print(f"Ranking tables: {len(ranked_rows)}; UNKNOWN: {unknown}; "
          f"phase-specific K1 selections: {len(picks)}")
