#!/usr/bin/env python3
"""Independent recheck (Grok Bot for Ray, 2026-11-12): membership_intervals.csv vs
eligibility_by_signal.csv.gz member_at_signal, all signal dates. Read-only on the pack.
Rule: member at S iff start_date <= S < end_date_exclusive (blank end = open).
Usage: python3 independent_recheck.py <pack_dir> <out_dir>"""
import sys, pandas as pd, numpy as np, json
pack, out = sys.argv[1], sys.argv[2]
iv = pd.read_csv(f"{pack}/membership_intervals.csv", dtype=str, keep_default_na=False)
el = pd.read_csv(f"{pack}/eligibility_by_signal.csv.gz", dtype=str, keep_default_na=False)
for c in ["member_at_signal","eligible_365d","eligible_via_rename_bridge","v18_dedupe_removed","eligible_final"]:
    assert set(el[c]) <= {"True","False"}, c
    el[c] = el[c] == "True"
iv["s"] = pd.to_datetime(iv.start_date)
iv["e"] = pd.to_datetime(iv.end_date_exclusive.replace("", "2262-01-01"))
dates = sorted(el.signal_date.unique())
facts = {"interval_rows": len(iv), "interval_unique_tickers": iv.engine_ticker.nunique(),
         "open_intervals": int((iv.end_date_exclusive=="").sum()),
         "max_start_date": iv.start_date.max(),
         "max_end_date_exclusive": iv.end_date_exclusive[iv.end_date_exclusive!=""].max(),
         "end_dates_in_2025_or_later": int((iv.end_date_exclusive>="2025").sum()),
         "eligibility_rows": len(el), "signal_dates": len(dates),
         "first_signal": dates[0], "last_signal": dates[-1],
         "eligibility_rows_member_false": int((~el.member_at_signal).sum()),
         "eligibility_dup_date_ticker": int(el.duplicated(["signal_date","engine_ticker"]).sum())}
# overlapping intervals for same ticker
ov = 0
for t, g in iv.sort_values("s").groupby("engine_ticker"):
    ov += int((g.s.values[1:] < g.e.values[:-1]).sum())
facts["overlapping_interval_pairs_same_ticker"] = ov
rows, detail = [], []
idx = el.set_index(["signal_date","engine_ticker"])
for d in dates:
    D = pd.Timestamp(d)
    from_iv = set(iv.loc[(iv.s <= D) & (D < iv.e), "engine_ticker"])
    sub = el[el.signal_date == d]
    from_el = set(sub.loc[sub.member_at_signal, "engine_ticker"])
    only_iv, only_el = sorted(from_iv - from_el), sorted(from_el - from_iv)
    def flags(t):
        if (d, t) in idx.index:
            r = idx.loc[(d, t)]
            return dict(in_eligibility_file=True, member_at_signal=bool(r.member_at_signal), eligible_365d=bool(r.eligible_365d),
                        eligible_via_rename_bridge=bool(r.eligible_via_rename_bridge), v18_dedupe_removed=bool(r.v18_dedupe_removed),
                        eligible_final=bool(r.eligible_final))
        return dict(in_eligibility_file=False, member_at_signal=None, eligible_365d=None, eligible_via_rename_bridge=None,
                    v18_dedupe_removed=None, eligible_final=None)
    for side, names in (("only_in_eligibility", only_el), ("only_in_intervals", only_iv)):
        for t in names:
            detail.append(dict(signal_date=d, engine_ticker=t, side=side, **flags(t)))
    nf = lambda names: sum(1 for t in names if flags(t)["eligible_final"])
    rows.append(dict(signal_date=d, n_members_from_intervals=len(from_iv), n_members_from_eligibility=len(from_el),
                     n_eligible_final=int(sub.eligible_final.sum()), match=not (only_iv or only_el),
                     n_only_in_intervals=len(only_iv), n_only_in_eligibility=len(only_el),
                     names_only_in_intervals=";".join(only_iv), names_only_in_eligibility=";".join(only_el),
                     n_mismatched_eligible_final_true=nf(only_iv)+nf(only_el)))
s = pd.DataFrame(rows); dt = pd.DataFrame(detail)
s.to_csv(f"{out}/per_signal_day_diffs.csv", index=False)
dt.to_csv(f"{out}/mismatched_names_flags.csv", index=False)
# per-name summary of the gap names
if len(dt):
    g = dt.groupby(["engine_ticker","side"]).agg(first_mismatch=("signal_date","min"), last_mismatch=("signal_date","max"),
        n_dates=("signal_date","count"), any_eligible_365d=("eligible_365d","max"), any_eligible_final=("eligible_final","max")).reset_index()
    first_member = el[el.member_at_signal].groupby("engine_ticker").signal_date.min()
    g["first_member_signal_in_eligibility"] = g.engine_ticker.map(first_member)
    g["prior_intervals_for_ticker"] = g.engine_ticker.map(lambda t: ";".join(f"{a}..{b or 'open'}" for a,b in iv.loc[iv.engine_ticker==t,["start_date","end_date_exclusive"]].values))
    g.to_csv(f"{out}/mismatched_names_summary.csv", index=False)
ye = s[s.signal_date.str[5:7]=="12"]
facts["year_end_signals"] = ye.signal_date.tolist()
facts["year_end_2014_2024_all_match"] = bool(ye[ye.signal_date<"2025"].match.all())
facts["mismatch_dates"] = s.loc[~s.match, ["signal_date","n_members_from_intervals","n_members_from_eligibility","n_only_in_intervals","n_only_in_eligibility"]].to_dict("records")
facts["n_matching_dates"] = int(s.match.sum())
facts["any_mismatched_name_eligible_final_true"] = bool(s.n_mismatched_eligible_final_true.sum() > 0)
facts["union_names_only_in_eligibility"] = sorted(set(dt.loc[dt.side=="only_in_eligibility","engine_ticker"])) if len(dt) else []
facts["union_names_only_in_intervals"] = sorted(set(dt.loc[dt.side=="only_in_intervals","engine_ticker"])) if len(dt) else []
json.dump(facts, open(f"{out}/independent_facts.json","w"), indent=1, default=str)
print(json.dumps(facts, indent=1, default=str))
