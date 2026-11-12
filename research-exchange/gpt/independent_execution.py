#!/usr/bin/env python3
"""Independent GPT execution/fee/drawdown audit, standard-library only.

Replays all 5,184 W1..12/H1..12/Top1..3/phase1..12 paths using GPT's
independently produced monthly rankings and Queen's frozen SHARED INPUTS.
It does NOT import Queen's engine or reference implementation.

Run at any working directory:
    python research-exchange/gpt/independent_ranker.py
    python research-exchange/gpt/independent_execution.py --verify

Output: research-exchange/gpt/independent_full_grid_5184.csv
This validates implementation consistency under shared provisional inputs,
not the correctness of upstream index membership or market-data history.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import gzip
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INP = ROOT / "shared" / "inputs-pack-full-20261109"
GPT = ROOT / "gpt"
QUEEN = ROOT / "queen" / "answer-key-full-20261109"
FEE = 0.0025

def read(path):
    path = Path(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def float_or_zero(v):
    return float(v) if v else 0.0

def execution_key(ticker, day):
    if ticker == "AGN":
        return "AGN_LEGACY" if day < "2015-03-17" else "AGN_PLC"
    if ticker == "ACT":
        return "AGN_PLC"
    return ticker

def month_of_index(offset):
    return f"{2015 + offset // 12:04d}-{offset % 12 + 1:02d}"

class Ledger:
    def __init__(self):
        self.sessions = [r["session"] for r in read(INP / "trading_sessions.csv")]
        self.day_at = {v: i for i, v in enumerate(self.sessions)}
        self.first_of_month = {}
        for day in self.sessions:
            self.first_of_month.setdefault(day[:7], day)
        self.ranking = {
            (r["signal_date"], int(r["W"])): r
            for r in read(GPT / "independent_rankings_1584.csv")
        }
        self.buy_data = {
            (r["trade_date"], r["engine_ticker"], r["window_end"]): r
            for r in read(INP / "execution_availability.csv.gz")
        }
        self.close = {
            (r["price_series"], r["execution_key"], r["date"]): float(r["close"])
            for r in read(INP / "execution_daily_closes.csv.gz")
        }
        self.last = {
            (r["price_series"], r["execution_key"]): r["series_last_date"]
            for r in read(INP / "series_info.csv")
        }
        self.events = defaultdict(list)
        for r in read(INP / "holding_events.csv"):
            self.events[r["price_series"], r["execution_key"]].append(r)
        self.delist = defaultdict(list)
        for r in read(INP / "untradable_deal_dates.csv"):
            self.delist[r["engine_ticker"]].append(r["deal_date"])

    def price(self, series, key, date):
        return self.close.get((series, key, date))

    def run(self, w: int, h: int, k: int, phase: int):
        start = self.first_of_month[month_of_index(phase-1)]
        end = self.first_of_month[month_of_index(phase+119)]

        rebalances = {}
        for m in range(phase-1, phase+119, h):
            trade = self.first_of_month[month_of_index(m)]
            if trade >= end:
                break
            next_trade = self.first_of_month.get(month_of_index(m+h))
            rebalances[trade] = end if next_trade is None or next_trade >= end else next_trade

        holdings = {}    # economic key -> (shares, price series, current close)
        cash = 1.0
        previous_session = start
        peak = float("-inf")
        mdd = 0.0
        executed = 0
        trades = 0
        selections = []
        value = 1.0

        def invalid(reason, key="", date=""):
            return dict(
                known=False, reason=reason, ticker=key, date=date,
                multiple=None, cagr=None, mdd=None,
                attempted_rebalances=executed, trades=trades
            )

        for di in range(self.day_at[start], self.day_at[end]+1):
            day = self.sessions[di]

            # Corporate actions are processed BEFORE today's mark-to-market.
            for econ, holding in list(holdings.items()):
                n, series, _ = holding
                touched = [
                    ev for ev in self.events[series, econ]
                    if previous_session < ev["ex_date"] <= day
                ]
                if any(ev["type"] == "unresolved" for ev in touched):
                    evt = next(ev for ev in touched if ev["type"] == "unresolved")
                    return invalid("corporate_action_unresolved", evt["event_id"])

                for ev in touched:
                    if ev["action"] in ("skip", "ignore"):
                        continue
                    if ev["action"] == "fail":
                        return invalid("corporate_action_unresolved", ev["event_id"], day)

                    amount = float_or_zero(ev["cash_per_share"])
                    ratio = float_or_zero(ev["ratio"])
                    if ev["type"] in ("spin", "special_div"):
                        child_value = n * ratio * float_or_zero(ev["child_first_close"])
                        cash += n * amount + child_value*(1-FEE)
                    elif ev["type"] == "cash":
                        cash += n * amount
                        holdings.pop(econ, None)
                    elif ev["type"] == "stock":
                        target = ev["child_or_acquirer"]
                        target_series = ev["acquirer_series"]
                        new_price = self.price(target_series, target, day)
                        if not new_price:
                            return invalid("corporate_action_unresolved", f"{econ}->{target}", day)
                        existing_qty = 0.0
                        if target in holdings and target != econ:
                            qty, old_series, _ = holdings[target]
                            old_price = self.price(old_series, target, day)
                            if old_price is None:
                                return invalid("daily_gap", target, day)
                            existing_qty = qty*old_price/new_price
                        cash += n*amount
                        holdings.pop(econ, None)
                        holdings[target] = (existing_qty + n*ratio, target_series, new_price)

            value = cash
            for econ, (qty, series, _) in list(holdings.items()):
                close = self.price(series, econ, day)
                if close is None:
                    last_date = self.last.get((series, econ), "9999-12-31")
                    kind = ("corporate_action_unresolved" if day > last_date else "daily_gap")
                    return invalid(kind, econ, day)
                holdings[econ] = (qty, series, close)
                value += qty*close

            if day in rebalances:
                signal = self.sessions[di-1]
                rank = self.ranking[signal, w]
                if rank["status"] == "UNKNOWN":
                    return invalid("rank_missing_unbounded", rank["unrankable_names"], day)
                names = [rank[f"top{i+1}"] for i in range(k)]
                keys = [execution_key(t, day) for t in names]
                if len(set(keys)) != len(keys):
                    return invalid("duplicate_economic_execution_key", "|".join(names), day)

                series_for = {}
                for ticker, ek in zip(names, keys):
                    if ek in holdings:
                        series_for[ek] = holdings[ek][1]
                        continue
                    trade = self.buy_data.get((day, ticker, rebalances[day]))
                    if trade is None:
                        return invalid("NOT_IN_PACK", ticker, day)
                    if trade["status"] == "not_tradable":
                        parts = trade["detail"].split("@")
                        return invalid("pick_not_tradable", parts[0], parts[-1] if len(parts)>1 else day)
                    if trade["status"] != "ok":
                        return invalid("exec_price_missing", ticker, day)
                    series_for[ek] = trade["price_series"]

                before = {ek: qty*p for ek,(qty,_,p) in holdings.items()}
                ep = value
                # Independently solve the turnover-fee fixed point.
                for _ in range(200):
                    target = ep/k
                    all_names = set(keys) | set(before)
                    turnover = sum(
                        abs((target if e in keys else 0.0) - before.get(e, 0.0))
                        for e in all_names
                    )
                    next_ep = value - FEE*turnover
                    if abs(next_ep-ep) <= 1e-12*value:
                        ep = next_ep
                        break
                    ep = next_ep

                target = ep/k
                updated = {}
                for ek in keys:
                    series = series_for[ek]
                    close = self.price(series, ek, day)
                    if not close:
                        return invalid("exec_price_missing", ek, day)
                    shares = target/close
                    prev_qty = holdings.get(ek, (0.0, "", 0.0))[0]
                    updated[ek] = (shares, series, close)
                    if abs(shares-prev_qty) > 1e-12*max(prev_qty, 1e-30):
                        trades += 1
                trades += sum(ek not in updated for ek in holdings)
                holdings = updated
                cash = 0.0
                value = ep
                selections.append((signal, day, names[0]))
                executed += 1

            if day == end:
                stock_value = sum(qty*p for qty,_,p in holdings.values())
                value = cash + (1-FEE)*stock_value
                trades += len(holdings)

            peak = max(peak, value)
            mdd = min(mdd, value/peak-1.0)
            previous_session = day

        # Engine-equivalent after-run K1 tradability check.
        if k == 1:
            for signal, day, ticker in selections:
                if any(signal < x <= day for x in self.delist.get(ticker, [])):
                    return invalid("pick_not_tradable", ticker, day)

        years = (dt.date.fromisoformat(end)-dt.date.fromisoformat(start)).days/365.25
        return dict(
            known=True, reason="", ticker="", date="",
            multiple=value, cagr=value**(1/years)-1, mdd=mdd,
            attempted_rebalances=executed, trades=trades
        )

def generate(verify: bool):
    ledger = Ledger()
    expected = {}
    if verify:
        for r in read(QUEEN / "paths_summary.csv"):
            expected[r["path_id"]] = r

    cols = [
        "path_id","W","H","K","phase","status","fail_reason",
        "fail_ticker","fail_date","final_multiple","CAGR","daily_MDD",
        "attempted_rebalances","n_trades"
    ]
    output = GPT / "independent_full_grid_5184.csv"
    mismatches, known, unknown = [], 0, 0
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for w in range(1,13):
            for h in range(1,13):
                for k in (1,2,3):
                    for phase in range(1,13):
                        ident = f"W{w}H{h}K{k}P{phase}"
                        r = ledger.run(w,h,k,phase)
                        known += int(r["known"])
                        unknown += int(not r["known"])
                        writer.writerow(dict(
                            path_id=ident, W=w,H=h,K=k,phase=phase,
                            status="KNOWN" if r["known"] else "UNKNOWN",
                            fail_reason=r["reason"], fail_ticker=r["ticker"],
                            fail_date=r["date"],
                            final_multiple=r["multiple"] if r["known"] else "",
                            CAGR=r["cagr"] if r["known"] else "",
                            daily_MDD=r["mdd"] if r["known"] else "",
                            attempted_rebalances=r["attempted_rebalances"],
                            n_trades=r["trades"],
                        ))
                        if verify:
                            q = expected[ident]
                            qknown = q["status"] != "UNKNOWN"
                            if r["known"] != qknown:
                                mismatches.append((ident,"status"))
                            if r["known"]:
                                for key,qname in (("multiple","final_multiple"),("cagr","CAGR"),("mdd","daily_MDD")):
                                    if abs(r[key]-float(q[qname])) > 1e-9*max(1.0,abs(float(q[qname]))):
                                        mismatches.append((ident,qname))
                            elif (r["reason"],r["ticker"],r["date"]) != (q["fail_reason"],q["fail_ticker"],q["fail_date"]):
                                mismatches.append((ident,"UNKNOWN detail"))
                            # Queen suppresses all reported rebalances for any
                            # post-hoc invalidated pick_not_tradable path. GPT
                            # preserves attempted_rebalances for audit, not a
                            # difference in execution math.
    print(f"GPT execution: total=5184 known={known} UNKNOWN={unknown}; "
          f"verification discrepancies={len(mismatches)}")
    if mismatches:
        for r in mismatches[:30]:
            print("MISMATCH", r)
        raise SystemExit(1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true",
                   help="compare against Queen's answer-key only AFTER computing")
    opts = p.parse_args()
    generate(opts.verify)
