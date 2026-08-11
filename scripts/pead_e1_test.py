"""E1 = PEAD, but only the high-quality beats. Does the drift-sleeve's E1 filter
sharpen PEAD and — the real question — slow its decay?

Raw PEAD fires on EPS surprise alone. E1 adds (all quant-computable here):
  - revenue beat too (revenueActual >= revenueEstimated * 1.02) — "both beats"
  - day-1 reaction in [+2%, +12%] — market noticed but didn't consume it
    (<+2% = rejected; >+12% = move already gone; both are E1 vetoes)

Everything else identical to pead_backtest (unified exit engine, Polygon prices,
stop 3xATR / target 6xATR / hold 20, cost 7.5bp/side). Reports each cohort's
edge + the sub-period thirds (PEAD decayed early +2.38% -> recent +1.05%; the
question is whether E1's recent third holds up better).
"""
from __future__ import annotations

import argparse
import asyncio

import numpy as np
import pandas as pd

from scripts.pead_backtest import _parse_day, _summ, run_config
from src.data.earnings_cache import get_earnings

MIN_HISTORY = 60


def _surprise(row, akey, ekey):
    a, e = row.get(akey), row.get(ekey)
    try:
        a, e = float(a), float(e)
    except (TypeError, ValueError):
        return None
    return (a - e) / abs(e) * 100 if abs(e) > 1e-9 else None


def _atr14(df, i, n=14):
    lo = max(1, i - n)
    h, low, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    trs = [max(h[j] - low[j], abs(h[j] - c[j - 1]), abs(low[j] - c[j - 1])) for j in range(lo, i + 1)]
    return float(np.mean(trs)) if trs else 0.0


async def build_events(price_data, min_eps):
    events = []
    for ticker, df in price_data.items():
        if ticker == "SPY" or len(df) < MIN_HISTORY:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
        days = list(df["date"])
        rows = await get_earnings(ticker)
        for row in rows:
            ed = _parse_day(row.get("date", ""))
            if ed is None:
                continue
            eps = _surprise(row, "epsActual", "epsEstimated")
            if eps is None or eps < min_eps:
                continue
            ts = pd.Timestamp(ed)
            e_idx = next((i for i, d in enumerate(days) if d >= ts), None)
            if e_idx is None or e_idx < 2 or e_idx >= len(df):
                continue
            atr = _atr14(df, e_idx)
            close = float(df["close"].iloc[e_idx])
            if atr <= 0 or close <= 0:
                continue
            prev = float(df["close"].iloc[e_idx - 1])
            reaction = (close / prev - 1) if prev else 0.0
            events.append({
                "ticker": ticker, "signal_date": days[e_idx], "surprise": eps,
                "rev": _surprise(row, "revenueActual", "revenueEstimated"),
                "reaction": reaction, "atr": atr, "close": close, "df": df,
            })
    return events


def _report(events, label):
    trades, eqt = run_config(events, stop_atr=3.0, target_atr=6.0, hold=20, cost=0.00075)
    print(_summ(trades, eqt, label))
    if trades:
        sd = sorted(t["signal_date"] for t in trades)
        e = (sd[len(sd) // 3], sd[2 * len(sd) // 3])
        for k, nm in enumerate(("  early", "  mid", "  recent")):
            st, se = [], []
            for t, x in zip(trades, eqt):
                third = 0 if t["signal_date"] <= e[0] else (1 if t["signal_date"] <= e[1] else 2)
                if third == k:
                    st.append(t)
                    se.append(x)
            print(_summ(st, se, nm))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default="outputs/research/ohlcv_polygon_3y.parquet")
    ap.add_argument("--min-eps", type=float, default=10.0)
    args = ap.parse_args()
    combined = pd.read_parquet(args.cache_file)
    price = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
             for t, g in combined.groupby("_ticker")}
    print(f"{len(price)} tickers; building events (cached earnings)...")
    ev = await build_events(price, args.min_eps)
    raw = ev
    both = [e for e in ev if (e["rev"] or -99) >= 2.0]
    e1 = [e for e in ev if (e["rev"] or -99) >= 2.0 and 0.02 <= e["reaction"] <= 0.12]
    react_only = [e for e in ev if 0.02 <= e["reaction"] <= 0.12]
    print(f"\nevents: raw(EPS>{args.min_eps:.0f}%)={len(raw)}  +rev_beat={len(both)}  "
          f"+reaction_band={len(react_only)}  E1(all)={len(e1)}\n")
    hdr = (f"{'cohort':<22}{'N':>6}{'WR':>8}{'avg%':>8}{'expect%':>9}{'PF':>7}"
           f"{'Sharpe':>7}{'equity×':>8}{'CAGR%':>8}{'eqDD%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for cohort, label in [(raw, "PEAD raw"), (both, "+ rev beat"),
                          (react_only, "+ reaction band"), (e1, "E1 (both)")]:
        _report(cohort, label)
        print()


if __name__ == "__main__":
    asyncio.run(main())
