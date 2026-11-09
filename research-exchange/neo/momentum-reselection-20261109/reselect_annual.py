# Independent selector wrapper for the published inputs pack.
# It deliberately imports no Queen engine or answer key.
import csv, gzip, collections
from pathlib import Path

PACK = Path(__file__).resolve().parents[2] / "shared/inputs-pack-20261109"
OUT = Path(__file__).with_name("annual_top1_reselected.csv")

def rows(name):
    p = PACK / name
    f = gzip.open(p, "rt") if name.endswith(".gz") else p.open()
    return list(csv.DictReader(f))

sessions = [r["session"] for r in rows("trading_sessions.csv")]
months = collections.defaultdict(list)
for d in sessions: months[d[:7]].append(d)
month_ends = sorted(max(v) for v in months.values())
month_ends = [d for d in month_ends if d >= "2013-11-29"]

eligible = collections.defaultdict(list)
for r in rows("eligibility_by_signal.csv.gz"):
    if r["eligible_final"] == "True": eligible[r["signal_date"]].append(r["engine_ticker"])
prices = {(r["engine_ticker"], r["month_end"]): float(r["rank_price"])
          for r in rows("rank_prices_month_end.csv.gz")}
overrides = collections.defaultdict(list)
for r in rows("rank_price_overrides.csv"):
    overrides[(r["engine_ticker"], r["month_end"])].append(
        (r["signal_from"], r["signal_to"], float(r["rank_price"])))
actions = collections.defaultdict(list)
for r in rows("corporate_actions_applied.csv"): actions[r["engine_ticker"]].append(r)

def price(t, d, signal):
    for lo, hi, value in overrides[(t, d)]:
        if lo <= signal <= hi: return value
    return prices.get((t, d))

def rank(signal, W):
    base = month_ends[month_ends.index(signal)-W]
    ranked, missing = [], []
    for t in sorted(eligible[signal]):
        ev = actions[t]
        if any(e["kind"] == "legacy_data_error_BLOCKER" and e["blocker_from"] <= signal <= e["blocker_to"] for e in ev):
            missing.append(t); continue
        if any(e["kind"] == "unresolved_BLOCKER" and base < e["ex_date"] <= signal for e in ev):
            missing.append(t); continue
        p0, p1 = price(t, base, signal), price(t, signal, signal)
        if p0 and p1:
            for e in ev:
                if e["kind"] in ("spin", "special_div") and base < e["ex_date"] <= signal:
                    if not e["factor_F"]: p0 = None; break
                    p0 *= float(e["factor_F"])
        if p0 and p1 and p0 > 0: ranked.append((p1/p0-1, t))
        else: missing.append(t)
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return base, ranked, missing

out = []
for y in range(2014, 2024):
    signal = max(months[f"{y}-12"])
    base, ranked, missing = rank(signal, 12)
    out.append({"holding_year": y+1, "signal": signal, "base_month_end": base,
                "winner": ranked[0][1], "winner_return": repr(ranked[0][0]),
                "runner_up": ranked[1][1], "runner_up_return": repr(ranked[1][0]),
                "n_ranked": len(ranked), "n_unrankable": len(missing)})
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
print(" ".join(f"{r['holding_year']}:{r['winner']}" for r in out))
