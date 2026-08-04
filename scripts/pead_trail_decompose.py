"""Decompose PEAD's live-vs-backtest gap: trail vs tiered-stop rescaling.

Live PEAD differs from the backtest that justified it in three ways:
  1. global trail 0.5/0.3 (backtest passed none)
  2. score-tiered stop rescaling — performance.py hardcodes MR's /0.75, turning
     PEAD's 3xATR stop into 2.0x / 3.4x / 5.0x by confidence tier
  3. 10bp vs 7.5bp slippage

Confidence = 0.9 * score, score = 60 + (surprise - min_surprise), so the tier a
PEAD signal lands in is an exact function of its EPS surprise.

Run on BOTH the raw EPS>=10% population and the live E1-gated one
(revenue beat >=2% AND day-1 reaction in [+2,+12]%).
"""
from __future__ import annotations

import asyncio
import statistics as st
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.pead_backtest import _parse_day, _surprise_pct, build_events
from src.data.earnings_cache import get_earnings
from src.research.signal_backtest import simulate_trade

MIN_SURPRISE = 10.0
STOP_ATR = 3.0
TARGET_ATR = 6.0
HOLD = 20


def live_stop_mult(surprise: float) -> float:
    """Replicate performance.py's tiered stop for a PEAD signal.

    tier_atr = |entry - stop| / 0.75 = 3xATR / 0.75 = 4xATR, then the tier
    multiple is applied to THAT — so the intended 3xATR becomes 2.0/3.4/5.0.
    """
    conf = 0.9 * min(60.0 + (surprise - MIN_SURPRISE), 100.0)
    tier = 1.25 if conf >= 85 else (0.85 if conf >= 70 else 0.50)
    return tier * (STOP_ATR / 0.75)


async def e1_flags(events):
    """Attach revenue surprise + day-1 reaction so the live E1 gate can be applied."""
    rev_cache: dict[str, list] = {}
    for ev in events:
        t = ev["ticker"]
        if t not in rev_cache:
            rev_cache[t] = await get_earnings(t)
        rs = None
        for row in rev_cache[t]:
            if _parse_day(row.get("date", "")) == ev["signal_date"]:
                a, e = row.get("revenueActual"), row.get("revenueEstimated")
                try:
                    a, e = float(a), float(e)
                    if abs(e) > 1e-6:
                        rs = (a - e) / abs(e) * 100
                except (TypeError, ValueError):
                    pass
                break
        ev["rev_surprise"] = rs
        # _report_day_reaction() has a live-freshness guard (report bar must be
        # among the last `max_age` bars), so it is always None for historical
        # events. Same formula, without the freshness gate.
        d = ev["df"]
        dates = [x.date() if hasattr(x, "date") else x for x in d["date"].tolist()]
        i = next((k for k, x in enumerate(dates) if x >= ev["signal_date"]), None)
        ev["reaction"] = None
        if i is not None and i >= 1:
            prev = float(d["close"].iloc[i - 1])
            if prev > 0:
                ev["reaction"] = (float(d["close"].iloc[i]) / prev - 1) * 100
    return events


def run(events, *, stop_mult, trail_act, trail_dist, cost):
    out = []
    for ev in events:
        atr, close = ev["atr"], ev["close"]
        sm = live_stop_mult(ev["surprise"]) if stop_mult == "live" else stop_mult
        r = simulate_trade(
            ev["df"], ev["signal_date"],
            stop_loss=round(close - sm * atr, 2),
            target=round(close + TARGET_ATR * atr, 2),
            max_hold=HOLD, slippage=cost, gap_through=True,
            trail_activate_pct=trail_act, trail_distance_pct=trail_dist,
        )
        if r:
            out.append((ev["signal_date"].year, r))
    return out


def rpt(name, rows):
    if not rows:
        print(f"  {name:46s} NO TRADES")
        return
    r = [x["pnl_pct"] for _, x in rows]
    hold = [x["holding_days"] for _, x in rows]
    sd = st.pstdev(r) or 1e-9
    se = sd / len(r) ** 0.5
    m = st.mean(r)
    yrs = sorted({y for y, _ in rows})
    ysplit = "  ".join(
        f"{y}:{st.mean([p['pnl_pct'] for yy, p in rows if yy == y]):+.2f}" for y in yrs)
    top = Counter(x["exit_reason"] for _, x in rows).most_common(1)[0]
    print(f"  {name:46s} n={len(r):5d} WR={100*sum(1 for x in r if x>0)/len(r):5.1f}% "
          f"avg={m:+.3f}% CI[{m-1.96*se:+.3f},{m+1.96*se:+.3f}] hold={st.median(hold):4.1f}d "
          f"{top[0]}{100*top[1]//len(r)}% | {ysplit}")


def main():
    df = pd.read_parquet("outputs/research/ohlcv_polygon_3y.parquet").rename(
        columns={"_ticker": "ticker"})
    price = {t: g.drop(columns=["ticker"]).reset_index(drop=True)
             for t, g in df.groupby("ticker")}
    events = asyncio.run(build_events(price, MIN_SURPRISE))
    events = asyncio.run(e1_flags(events))
    e1 = [e for e in events
          if e["rev_surprise"] is not None and e["rev_surprise"] >= 2.0
          and e["reaction"] is not None and 2.0 <= e["reaction"] <= 12.0]
    have_rev = sum(1 for e in events if e["rev_surprise"] is not None)
    have_rx = sum(1 for e in events if e["reaction"] is not None)
    pass_rev = sum(1 for e in events if (e["rev_surprise"] or -99) >= 2.0)
    pass_rx = sum(1 for e in events
                  if e["reaction"] is not None and 2.0 <= e["reaction"] <= 12.0)
    print(f"raw EPS>=10% events: {len(events)}   E1-gated: {len(e1)}")
    print(f"  gate attrition: revenue field present {have_rev}, reaction computable {have_rx}"
          f" | pass rev>=2% {pass_rev}, pass reaction[2,12]% {pass_rx}\n")

    tiers = Counter("2.0x" if live_stop_mult(e["surprise"]) < 2.5 else
                    ("3.4x" if live_stop_mult(e["surprise"]) < 4 else "5.0x")
                    for e in events)
    print(f"live tier distribution (designed 3.0x): {dict(tiers)}\n")

    configs = [
        ("A designed  (3xATR, no trail, 7.5bp)", dict(stop_mult=STOP_ATR, trail_act=0.0, trail_dist=0.0, cost=0.00075)),
        ("B LIVE EXACT (tiered, trail .5/.3, 10bp)", dict(stop_mult="live", trail_act=0.5, trail_dist=0.3, cost=0.001)),
        ("C live minus trail (tiered, no trail)", dict(stop_mult="live", trail_act=0.0, trail_dist=0.0, cost=0.001)),
        ("D live minus tier bug (3xATR, trail)", dict(stop_mult=STOP_ATR, trail_act=0.5, trail_dist=0.3, cost=0.001)),
        ("E PROPOSED FIX (3xATR, no trail, 10bp)", dict(stop_mult=STOP_ATR, trail_act=0.0, trail_dist=0.0, cost=0.001)),
    ]
    for pop_name, pop in (("RAW EPS>=10%", events), ("E1-GATED (live)", e1)):
        print("=" * 132)
        print(f"{pop_name}  n={len(pop)}      [year split = avg pnl/trade]")
        print("=" * 132)
        for name, cfg in configs:
            rpt(name, run(pop, **cfg))
        print()


if __name__ == "__main__":
    main()
