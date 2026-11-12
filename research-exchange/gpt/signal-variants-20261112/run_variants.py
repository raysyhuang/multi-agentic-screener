#!/usr/bin/env python3
"""Compare baseline, one-month skip, and SPY trend-filter variants.

Only tests W10H2 and W12H4, K=1, across the same 12 start-month phases.
Uses the tracked provisional input pack, GPT execution ledger mechanics, and
FMP SPY daily prices bundled alongside this script.
"""
import csv
import datetime as dt
import importlib.util
import math
import sys
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GPT = ROOT / "research-exchange/gpt"
PACK = ROOT / "research-exchange/shared/inputs-pack-full-20261109"
ANSWER = ROOT / "research-exchange/queen/answer-key-full-20261109"
FEE = 0.0025

spec = importlib.util.spec_from_file_location("independent_execution", GPT / "independent_execution.py")
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)


def rows(path):
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def month_data():
    days = defaultdict(list)
    for r in rows(PACK / "trading_sessions.csv"):
        days[r["session"][:7]].append(r["session"])
    months = sorted(max(v) for v in days.values())
    first = {m: min(v) for m, v in days.items()}
    return months, {d: i for i, d in enumerate(months)}, first


def alternative_rankings(skip_month):
    months, idx_by_day, _ = month_data()
    eligible = defaultdict(set)
    for r in rows(PACK / "eligibility_by_signal.csv.gz"):
        if r["eligible_final"] == "True":
            eligible[r["signal_date"]].add(r["engine_ticker"])
    prices = {(r["engine_ticker"], r["month_end"]): float(r["rank_price"])
              for r in rows(PACK / "rank_prices_month_end.csv.gz")}
    overrides = defaultdict(list)
    for r in rows(PACK / "rank_price_overrides.csv"):
        overrides[r["engine_ticker"], r["month_end"]].append(
            (r["signal_from"], r["signal_to"], float(r["rank_price"]))
        )
    actions = defaultdict(list)
    for r in rows(PACK / "corporate_actions_applied.csv"):
        actions[r["engine_ticker"]].append(r)

    def px(ticker, day, signal):
        for lo, hi, val in overrides.get((ticker, day), []):
            if lo <= signal <= hi:
                return val
        return prices.get((ticker, day))

    out = {}
    for signal, members in sorted(eligible.items()):
        i = idx_by_day[signal]
        for w in (10, 12):
            start_i = i - w
            end_i = i - 1 if skip_month else i
            if start_i < 0:
                continue
            start_date, end_date = months[start_i], months[end_i]
            ranked, invalid = [], []
            for ticker in members:
                start, end = px(ticker, start_date, signal), px(ticker, end_date, signal)
                mult, reason = 1.0, None
                for act in actions[ticker]:
                    kind, when = act["kind"], act["ex_date"]
                    if kind == "legacy_data_error_BLOCKER" and act["blocker_from"] <= signal <= act["blocker_to"]:
                        reason = "legacy_data_error"
                    elif kind == "unresolved_BLOCKER" and start_date < when <= end_date:
                        reason = "unresolved_event"
                    elif kind in ("spin", "special_div") and start_date < when <= end_date:
                        if not act["factor_F"]:
                            reason = "missing_action_factor"
                        else:
                            mult *= float(act["factor_F"])
                if not start or not end or start <= 0 or end <= 0:
                    reason = reason or "missing_price"
                if reason:
                    invalid.append((ticker, reason))
                else:
                    ranked.append((ticker, end / (start * mult) - 1.0))
            ranked.sort(key=lambda x: (-x[1], x[0]))
            out[signal, w] = {"eligible": len(members), "top": ranked, "invalid": sorted(invalid)}
    return out


def read_spy():
    rec = rows(HERE / "spy_prices_fmp_2013-2025.csv")
    return {r["date"]: (float(r["close"]), float(r["adjClose"])) for r in rec}


def prior_price(prices, day, index=0):
    d = dt.date.fromisoformat(day)
    while d.isoformat() not in prices:
        d -= dt.timedelta(days=1)
    return prices[d.isoformat()][index]


def cagr(mult, start, end):
    days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days
    return mult ** (365.25 / days) - 1


class Runner:
    def __init__(self, rankmap, use_filter=False):
        self.ledger = engine.Ledger()
        self.rankmap = rankmap
        self.use_filter = use_filter
        self.months, _, self.first = month_data()
        self.spy = read_spy()
        self.spy_ma = {}
        for i, month_end in enumerate(self.months):
            if i >= 9 and month_end >= "2014-12-31":
                vals = [prior_price(self.spy, self.months[j], 0) for j in range(i-9, i+1)]
                self.spy_ma[month_end] = sum(vals) / 10

    def run(self, w, h, phase):
        L = self.ledger
        start = self.first[f"{2015+(phase-1)//12:04d}-{(phase-1)%12+1:02d}"]
        last_month_index = phase + 119
        end_month = f"{2015+last_month_index//12:04d}-{last_month_index%12+1:02d}"
        end = self.first[end_month]
        rebalances = {}
        for m in range(phase-1, phase+119, h):
            ym = f"{2015+m//12:04d}-{m%12+1:02d}"
            trade = self.first[ym]
            if trade >= end:
                break
            nm = m + h
            next_month = f"{2015+nm//12:04d}-{nm%12+1:02d}"
            next_trade = self.first.get(next_month, end)
            rebalances[trade] = end if next_trade >= end else next_trade

        holdings, cash = {}, 1.0
        previous_session, peak, mdd = start, float("-inf"), 0.0
        value, executed, trades, selections = 1.0, 0, 0, []
        cash_days = 0
        sessions = L.sessions

        def invalid(reason, ticker="", day=""):
            return {"known": False, "reason": reason, "ticker": ticker, "date": day,
                    "multiple": None, "cagr": None, "mdd": None,
                    "rebalances": executed, "trades": trades}

        for di in range(L.day_at[start], L.day_at[end]+1):
            day = sessions[di]
            for econ, (qty, series, _) in list(holdings.items()):
                touched = [ev for ev in L.events[series, econ] if previous_session < ev["ex_date"] <= day]
                if any(ev["type"] == "unresolved" for ev in touched):
                    ev = next(ev for ev in touched if ev["type"] == "unresolved")
                    return invalid("corporate_action_unresolved", ev["event_id"], day)
                for ev in touched:
                    if ev["action"] in ("skip", "ignore"):
                        continue
                    if ev["action"] == "fail":
                        return invalid("corporate_action_unresolved", ev["event_id"], day)
                    amount, ratio = float(ev["cash_per_share"] or 0), float(ev["ratio"] or 0)
                    if ev["type"] in ("spin", "special_div"):
                        child = qty * ratio * float(ev["child_first_close"] or 0)
                        cash += qty * amount + child * (1-FEE)
                    elif ev["type"] == "cash":
                        cash += qty * amount
                        holdings.pop(econ, None)
                    elif ev["type"] == "stock":
                        target, ts = ev["child_or_acquirer"], ev["acquirer_series"]
                        newpx = L.price(ts, target, day)
                        if not newpx:
                            return invalid("corporate_action_unresolved", f"{econ}->{target}", day)
                        existing = 0.0
                        if target in holdings and target != econ:
                            q, oldseries, _ = holdings[target]
                            oldpx = L.price(oldseries, target, day)
                            if oldpx is None:
                                return invalid("daily_gap", target, day)
                            existing = q * oldpx / newpx
                        cash += qty * amount
                        holdings.pop(econ, None)
                        holdings[target] = (existing + qty*ratio, ts, newpx)

            value = cash
            for econ, (qty, series, _) in list(holdings.items()):
                px = L.price(series, econ, day)
                if px is None:
                    last = L.last.get((series, econ), "9999-12-31")
                    return invalid("corporate_action_unresolved" if day > last else "daily_gap", econ, day)
                holdings[econ] = (qty, series, px)
                value += qty * px

            if day in rebalances:
                signal = sessions[di-1]
                desired_cash = False
                if self.use_filter:
                    signal_month = max(d for d in self.months if d <= signal)
                    ma = self.spy_ma.get(signal_month)
                    desired_cash = ma is None or prior_price(self.spy, signal_month, 0) <= ma
                rank = self.rankmap.get((signal, w))
                if not desired_cash and (rank is None or rank["invalid"]):
                    detail = "|".join(x[0] for x in rank["invalid"]) if rank else ""
                    return invalid("rank_missing_unbounded", detail, day)
                names = [] if desired_cash else [rank["top"][0][0]]
                keys = [engine.execution_key(t, day) for t in names]
                if len(set(keys)) != len(keys):
                    return invalid("duplicate_economic_execution_key", "|".join(names), day)
                series_for = {}
                for ticker, ek in zip(names, keys):
                    if ek in holdings:
                        series_for[ek] = holdings[ek][1]
                        continue
                    trade = L.buy_data.get((day, ticker, rebalances[day]))
                    if trade is None:
                        # The availability table is sparse and only enumerates
                        # the original picked paths. For alternate picks, use a
                        # direct-ticker series only when both fill-window dates
                        # are explicitly present in the pack's full daily-close
                        # table. Ambiguous/alias series remain unresolved.
                        candidates = [f"fmp:{ticker}", f"acq:{ticker}"]
                        available = [s for s in candidates
                                     if (s, ticker, day) in L.close
                                     and (s, ticker, rebalances[day]) in L.close]
                        if not available:
                            return invalid("NOT_IN_PACK", ticker, day)
                        trade = {"status": "ok", "price_series": available[0]}
                    if trade["status"] == "not_tradable":
                        parts = trade["detail"].split("@")
                        return invalid("pick_not_tradable", parts[0], parts[-1] if len(parts)>1 else day)
                    if trade["status"] != "ok":
                        return invalid("exec_price_missing", ticker, day)
                    series_for[ek] = trade["price_series"]

                before = {ek: qty*px for ek, (qty, _, px) in holdings.items()}
                ep = value
                for _ in range(200):
                    target = ep if names else 0.0
                    turnover = sum(abs((target if e in keys else 0.0)-before.get(e, 0.0)) for e in set(keys)|set(before))
                    n_ep = value - FEE*turnover
                    if abs(n_ep-ep) <= 1e-12*value:
                        ep = n_ep
                        break
                    ep = n_ep
                target = ep if names else 0.0
                new_holdings = {}
                for ek in keys:
                    series = series_for[ek]
                    px = L.price(series, ek, day)
                    if not px:
                        return invalid("exec_price_missing", ek, day)
                    new_holdings[ek] = (target/px, series, px)
                    old_qty = holdings.get(ek, (0.0, "", 0.0))[0]
                    if abs(target/px-old_qty) > 1e-12*max(old_qty, 1e-30):
                        trades += 1
                trades += sum(ek not in new_holdings for ek in holdings)
                holdings, cash, value = new_holdings, ep-target, ep
                if names:
                    selections.append((signal, day, names[0]))
                executed += 1

            if day == end:
                stock = sum(qty*px for qty, _, px in holdings.values())
                value = cash + (1-FEE)*stock
                trades += len(holdings)
            if not holdings:
                cash_days += 1
            peak = max(peak, value)
            mdd = min(mdd, value/peak-1)
            previous_session = day

        for signal, day, ticker in selections:
            if any(signal < x <= day for x in L.delist.get(ticker, [])):
                return invalid("pick_not_tradable", ticker, day)
        years = (dt.date.fromisoformat(end)-dt.date.fromisoformat(start)).days/365.25
        return {"known": True, "reason": "", "ticker": "", "date": "", "start": start, "end": end, "multiple": value,
                "cagr": value**(1/years)-1, "mdd": mdd, "rebalances": executed, "trades": trades,
                "cash_days": cash_days, "cash_pct": cash_days/(L.day_at[end]-L.day_at[start]+1)}


def main():
    base = engine.Ledger().ranking
    base_rankings = {}
    # Match the executor's ranking row schema to the direct ranker representation.
    for (signal, w), r in base.items():
        top = []
        for i in range(1, 4):
            if r[f"top{i}"]:
                top.append((r[f"top{i}"], float(r[f"ret{i}"])))
        invalid = [(x, "") for x in r["unrankable_names"].split("|") if x]
        base_rankings[signal, w] = {"top": top, "invalid": invalid}
    skip_rankings = alternative_rankings(skip_month=True)
    outputs = []
    for w, h in ((10,2),(12,4)):
        for variant, ranking, filtered in (
            ("baseline", base_rankings, False),
            ("skip1", skip_rankings, False),
            ("spy_sma10", base_rankings, True),
        ):
            runner = Runner(ranking, filtered)
            for phase in range(1,13):
                x = runner.run(w,h,phase)
                outputs.append({"variant": variant, "W": w, "H": h, "K": 1,
                                "phase": phase, "status": "KNOWN" if x["known"] else "UNKNOWN",
                                "start": x.get("start", ""), "end": x.get("end", ""),
                                "fail_reason": x["reason"], "fail_ticker": x["ticker"], "fail_date": x["date"],
                                "final_multiple": x["multiple"], "CAGR": x["cagr"],
                                "daily_MDD": x["mdd"], "rebalances": x["rebalances"], "trades": x["trades"],
                                "cash_days": x.get("cash_days", ""), "cash_pct": x.get("cash_pct", "")})
    out_path = HERE / "variant_paths.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(outputs[0]))
        writer.writeheader(); writer.writerows(outputs)
    answer = {(r["path_id"], r["status"]): r for r in rows(ANSWER / "paths_summary.csv")}
    # Strict reproduction check for all 24 baseline paths.
    errors = []
    for x in outputs:
        if x["variant"] != "baseline":
            continue
        ident = f"W{x['W']}H{x['H']}K1P{x['phase']}"
        q = next(r for r in answer.values() if r["path_id"] == ident)
        for a, b in (("CAGR","CAGR"),("daily_MDD","daily_MDD"),("final_multiple","final_multiple")):
            if abs(float(x[a])-float(q[b])) > 1e-10:
                errors.append((ident,a,float(x[a])-float(q[b])))
    if errors:
        raise RuntimeError(f"baseline reproduction failed: {errors[:5]}")
    spy = read_spy()
    print("Baseline reproduction: 24/24 match answer key within 1e-10")
    for w,h in ((10,2),(12,4)):
        for variant in ("baseline","skip1","spy_sma10"):
            rr=[r for r in outputs if r["W"]==w and r["variant"]==variant]
            known=[r for r in rr if r["status"]=="KNOWN"]
            cs=sorted(r["CAGR"] for r in known)
            dds=sorted(r["daily_MDD"] for r in known)
            beat=sum(r["CAGR"] > cagr(prior_price(spy,r["end"],1)/prior_price(spy,r["start"],1),r["start"],r["end"])
                     for r in known)
            if known:
                cash_pct = statistics.median(float(r["cash_pct"]) for r in known)
                print(f"W{w}H{h} {variant}: known={len(known)}/12 median_CAGR={statistics.median(cs):.4%} range=[{min(cs):.4%},{max(cs):.4%}] median_MaxDD={statistics.median(dds):.4%} worst_MaxDD={min(dds):.4%} median_cash={cash_pct:.1%} phases_beating_SPY_total={beat}/{len(known)}")
            else:
                print(f"W{w}H{h} {variant}: known=0/12; no complete paths")


if __name__ == "__main__":
    main()
