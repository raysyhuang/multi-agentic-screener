"""E2 — PEAD per-trade regime stamp (pre-registered in STRATEGY_REVIEW_2026-08).

Hypothesis: PEAD's +1.8-2.2%/trade backtest is a bull-tape estimate; the bear
cohort is unsampled and live bear PEAD is losing.
Decision rule: bear >= +0.5%/trade @ n>=100 -> no gate. Bear <= 0 -> bear-block
PEAD like sniper; promotion criteria change.

Market regime per DATE from SPY (SMA20/50 tally, same shape as
signal_backtest.classify_regime but computed per-day on the market index, which
is what LIVE gates on — not the per-ticker label the model backtests stamp).
"""
from __future__ import annotations

import asyncio
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.pead_trail_decompose import build_events, e1_flags, MIN_SURPRISE
from src.research.signal_backtest import simulate_trade


def spy_regime_series(spy: pd.DataFrame) -> dict:
    c = spy["close"].astype(float).reset_index(drop=True)
    s50 = c.rolling(50).mean()
    s20 = c.rolling(20).mean()
    out = {}
    for i in range(len(spy)):
        d = spy["date"].iloc[i]
        if pd.isna(s50.iloc[i]):
            out[d] = "unknown"
        elif c.iloc[i] > s50.iloc[i] and s20.iloc[i] > s50.iloc[i]:
            out[d] = "bull"
        elif c.iloc[i] < s50.iloc[i] and s20.iloc[i] < s50.iloc[i]:
            out[d] = "bear"
        else:
            out[d] = "choppy"
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - m), 100 * (c + m))


def rpt(label: str, rows: list) -> None:
    if not rows:
        print(f"  {label:22s} n=0")
        return
    r = [p for _, p in rows]
    w = sum(1 for x in r if x > 0)
    lo, hi = wilson(w, len(r))
    sd = st.pstdev(r) or 1e-9
    se = sd / len(r) ** 0.5
    m = st.mean(r)
    yrs = sorted({d.year for d, _ in rows})
    ys = "  ".join(f"{y}:{st.mean([p for d, p in rows if d.year == y]):+.2f}"
                   f"(n={sum(1 for d, _ in rows if d.year == y)})" for y in yrs)
    print(f"  {label:22s} n={len(r):4d} WR={100 * w / len(r):5.1f}% wilson[{lo:.0f},{hi:.0f}] "
          f"avg={m:+.3f}% CI[{m - 1.96 * se:+.3f},{m + 1.96 * se:+.3f}] | {ys}")


def main() -> None:
    df = pd.read_parquet(
        "outputs/research/ohlcv_polygon_3y.parquet").rename(columns={"_ticker": "ticker"})
    spy = df[df["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)
    regime = spy_regime_series(spy)
    price = {t: g.drop(columns=["ticker"]).reset_index(drop=True)
             for t, g in df.groupby("ticker") if t != "SPY"}

    events = asyncio.run(build_events(price, MIN_SURPRISE))
    events = asyncio.run(e1_flags(events))
    e1 = [e for e in events
          if e["rev_surprise"] is not None and e["rev_surprise"] >= 2.0
          and e["reaction"] is not None and 2.0 <= e["reaction"] <= 12.0]

    # sanity: regime coverage on event dates
    missing = sum(1 for e in events if regime.get(e["signal_date"], "unknown") == "unknown")
    print(f"events raw={len(events)} e1={len(e1)}; regime-unknown on {missing} event dates")
    from collections import Counter
    mix = Counter(regime.get(e["signal_date"], "unknown") for e in events)
    print(f"event-date regime mix (raw): {dict(mix)}\n")

    for name, pop in (("RAW EPS>=10%", events), ("E1-GATED (live config)", e1)):
        rows = {"bull": [], "choppy": [], "bear": [], "unknown": []}
        for e in pop:
            r = simulate_trade(
                e["df"], e["signal_date"],
                stop_loss=round(e["close"] - 3.0 * e["atr"], 2),
                target=round(e["close"] + 6.0 * e["atr"], 2),
                max_hold=20, slippage=0.001, gap_through=True,
                trail_activate_pct=0.0, trail_distance_pct=0.0,  # post-#43 live config
            )
            if r:
                rows[regime.get(e["signal_date"], "unknown")].append(
                    (e["signal_date"], r["pnl_pct"]))
        print(f"=== {name} — post-fix live config (3xATR, no trail, 10bp) ===")
        for reg in ("bull", "choppy", "bear", "unknown"):
            rpt(reg, rows[reg])
        print()


if __name__ == "__main__":
    main()
