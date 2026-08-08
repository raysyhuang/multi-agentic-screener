"""E3 — universe letter-bias: does the missed-letter cohort carry signal EV?

Pre-registered kill criterion: missed cohort contributes <5% of total MR signal
EV -> letter bias is cosmetic.

Method: simulate the ACTUAL select_ohlcv_tickers (round-robin) daily over the 3Y
parquet at the live cap ratio (38%), get per-ticker-day selection outcomes; find
MR trigger days (RSI2<=10) and their forward EV (T+1 open -> T+3 close, the MR
hold, incl. 10bp/side); compare EV captured under round-robin vs a dollar-volume
rank cap of the SAME size (isolates the letter mechanism from the cap itself).

Caveat carried from the review: the parquet is 94% tier-1 liquid names, so this
tests the MECHANICS at calibrated cap ratio, not the true-universe composition.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.universe_selection import select_ohlcv_tickers

CAP_RATIO = 0.38  # live: 1000 selected of ~2650 filtered


def rsi2(close: pd.Series) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def main() -> None:
    df = pd.read_parquet(
        "outputs/research/ohlcv_polygon_3y.parquet").rename(columns={"_ticker": "ticker"})
    df = df[df["ticker"] != "SPY"]
    tickers = sorted(df["ticker"].unique())
    n_cap = round(CAP_RATIO * len(tickers))
    print(f"{len(tickers)} tickers, daily cap={n_cap} ({CAP_RATIO:.0%})")

    # per-ticker frames + trigger days + forward EV
    frames = {}
    trig = {}  # ticker -> {date: fwd_ret}
    mcap_proxy = {}
    for t, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        frames[t] = g
        r = rsi2(g["close"].astype(float))
        fwd = {}
        for i in np.where(r <= 10)[0]:
            if i + 4 >= len(g) or i < 20:
                continue
            entry = g["open"].iloc[i + 1] * 1.001
            exitp = g["close"].iloc[i + 4] * 0.999
            fwd[g["date"].iloc[i]] = (exitp - entry) / entry * 100
        trig[t] = fwd
        mcap_proxy[t] = float((g["close"] * g["volume"]).median())

    all_dates = sorted(df["date"].unique())
    # daily selection under BOTH schemes
    sel_rr = defaultdict(set)   # date -> set(tickers) round-robin (actual code)
    sel_rank = defaultdict(set)  # date -> set: straight dollar-vol rank, same cap
    last = {t: g.set_index("date") for t, g in frames.items()}
    # Build a per-date lookup of (ticker, dv) once
    dv_by_date = defaultdict(dict)
    for t, g in frames.items():
        dvs = (g["close"] * g["volume"]).values
        for d, dv in zip(g["date"].values, dvs):
            dv_by_date[d][t] = float(dv)

    for d in all_dates:
        entries = [{"symbol": t, "marketCap": mcap_proxy[t], "price": 50.0,
                    "volume": dv_by_date[d][t] / 50.0}
                   for t in dv_by_date[d]]
        if len(entries) <= n_cap:
            for e in entries:
                sel_rr[d].add(e["symbol"]); sel_rank[d].add(e["symbol"])
            continue
        chosen = select_ohlcv_tickers(entries, max_tickers=n_cap)
        sel_rr[d] = set(chosen)
        ranked = sorted(entries, key=lambda e: -dv_by_date[d][e["symbol"]])[:n_cap]
        sel_rank[d] = {e["symbol"] for e in ranked}

    # EV accounting
    tot_ev = cap_rr = cap_rank = 0.0
    n_sig = n_rr = n_rank = 0
    missed_rr = defaultdict(float)  # letter -> missed EV under round-robin
    for t, fwd in trig.items():
        for d, ev in fwd.items():
            tot_ev += ev; n_sig += 1
            if t in sel_rr[d]:
                cap_rr += ev; n_rr += 1
            else:
                missed_rr[t[0]] += ev
            if t in sel_rank[d]:
                cap_rank += ev; n_rank += 1

    print(f"\nMR trigger-days total: {n_sig}, summed EV {tot_ev:+.1f}pp "
          f"(avg {tot_ev / max(n_sig, 1):+.3f}%/signal)")
    for name, cap, n in (("ROUND-ROBIN (live code)", cap_rr, n_rr),
                         ("DOLLAR-VOL RANK (same cap)", cap_rank, n_rank)):
        print(f"  {name:28s} captured {n:5d}/{n_sig} signals "
              f"({100 * n / max(n_sig, 1):.1f}%), EV {cap:+.1f}pp "
              f"({100 * cap / tot_ev if tot_ev else 0:.1f}% of total)")
    print(f"\n  round-robin FORFEITED: {n_sig - n_rr} signals, "
          f"{tot_ev - cap_rr:+.1f}pp EV = {100 * (tot_ev - cap_rr) / tot_ev if tot_ev else 0:.1f}% of total"
          f"  [kill criterion: <5% => cosmetic]")
    top = sorted(missed_rr.items(), key=lambda kv: -abs(kv[1]))[:8]
    print("  missed EV by first letter:", "  ".join(f"{k}:{v:+.1f}" for k, v in top))


if __name__ == "__main__":
    main()
