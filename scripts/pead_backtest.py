"""PEAD backtest — long earnings beats, through the unified exit engine.

The PEAD probe found real, monotonic post-earnings drift on the beat side
(+148 bp/20d over base for big beats). This turns it into a proper backtest that
can KILL or confirm it as a tradeable strategy:

  * Signal: an earnings BEAT (or big beat) — EPS surprise above a threshold.
  * Entry: T+1 open after the announcement day (look-ahead-safe), via the shared
    simulate_trade (unified exit engine): realistic T+1 fill, gap rejection off,
    intraday-style stop/target on daily bars, gap-through fills, per-side cost.
  * Exit: ~20-day hold with a wide ATR stop (swing hold needs room — tight stops
    killed the intraday candidates); expiry at close otherwise.
  * Reported: per-trade net expectancy, concurrency-capped equity curve, cost
    sensitivity, and a sub-period split (guards against bull-window beta).

Runs entirely on cached daily bars + cached FMP earnings — no new API pulls.

Usage:
  python scripts/pead_backtest.py --cache-file <daily_parquet> \
      [--min-surprise 2] [--stop-atr 3] [--target-atr 6] [--hold 20]
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_metrics
from src.data.earnings_cache import get_earnings
from src.features.technical import compute_all_technical_features
from src.research.signal_backtest import simulate_trade
import scripts.sniper_equity_curve as eq

MIN_HISTORY = 60


def _parse_day(s: str) -> date | None:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _surprise_pct(row: dict) -> float | None:
    a, e = row.get("epsActual"), row.get("epsEstimated")
    try:
        a, e = float(a), float(e)
    except (TypeError, ValueError):
        return None
    if abs(e) < 1e-6:
        return None
    return (a - e) / abs(e) * 100


async def build_events(price_data, min_surprise):
    """Yield trades as dicts: (ticker, signal_date=announcement day, surprise, atr, close)."""
    events = []
    for ticker, df in price_data.items():
        if ticker == "SPY" or len(df) < MIN_HISTORY:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        enr = compute_all_technical_features(df).reset_index(drop=True)
        days = df["date"].tolist()
        win_start, win_end = days[0], days[-1]
        rows = await get_earnings(ticker)
        for row in rows:
            ed = _parse_day(row.get("date", ""))
            if ed is None or ed < win_start or ed > win_end:
                continue
            sp = _surprise_pct(row)
            if sp is None or sp < min_surprise:
                continue
            e_idx = next((i for i, d in enumerate(days) if d >= ed), None)
            if e_idx is None or e_idx >= len(df):
                continue
            atr = enr.iloc[e_idx].get("atr_14")
            close = float(df.iloc[e_idx]["close"])
            if atr is None or pd.isna(atr) or atr <= 0 or close <= 0:
                continue
            events.append({"ticker": ticker, "signal_date": days[e_idx],
                           "surprise": sp, "atr": float(atr), "close": close, "df": df})
    return events


def run_config(events, *, stop_atr, target_atr, hold, cost):
    trades = []
    eq_trades = []
    for ev in events:
        atr = ev["atr"]
        close = ev["close"]
        res = simulate_trade(
            ev["df"], ev["signal_date"],
            stop_loss=round(close - stop_atr * atr, 2),
            target=round(close + target_atr * atr, 2),
            max_hold=hold, slippage=cost, gap_through=True,
        )
        if res is None:
            continue
        trades.append({**res, "ticker": ev["ticker"], "surprise": ev["surprise"],
                       "signal_date": ev["signal_date"]})
        eq_trades.append(eq.Trade(ticker=ev["ticker"], entry=res["entry_date"],
                                  exit=res["exit_date"], pnl_pct=res["pnl_pct"],
                                  regime="", score=ev["surprise"]))
    return trades, eq_trades


def _summ(trades, eq_trades, label, *, max_concurrent=15, fraction=0.10):
    if not trades:
        return f"{label:<20}{'0 trades':>10}"
    r = np.array([t["pnl_pct"] for t in trades])
    m = compute_metrics(r.tolist())
    e = eq.simulate(eq_trades, mode="fixed_fraction", fraction=fraction,
                    max_concurrent=max_concurrent, start_capital=100_000.0) if eq_trades else None
    return (f"{label:<20}{len(r):>6}{m.win_rate:>8.1%}{m.avg_return_pct:>8.2f}"
            f"{m.expectancy:>9.3f}{m.profit_factor:>7.2f}{m.sharpe_ratio:>7.2f}"
            f"{(e.multiple if e else 0):>8.2f}{(e.cagr_pct if e else 0):>8.1f}"
            f"{(e.max_drawdown_pct if e else 0):>7.1f}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", required=True)
    ap.add_argument("--stop-atr", type=float, default=3.0)
    ap.add_argument("--target-atr", type=float, default=6.0)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--cost-bps", type=float, default=7.5, help="per-side cost in bps")
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}
    cost = args.cost_bps / 10000.0
    print(f"Building earnings events (cached)...")

    hdr = (f"{'config':<20}{'N':>6}{'WR':>8}{'avg%':>8}{'expect%':>9}{'PF':>7}"
           f"{'Sharpe':>7}{'equity×':>8}{'CAGR%':>8}{'eqDD%':>7}")
    print(f"stop {args.stop_atr}xATR, target {args.target_atr}xATR, hold {args.hold}d, "
          f"cost {args.cost_bps}bp/side\n")

    # Surprise-threshold sweep (beat vs big-beat).
    print("Long earnings beats — surprise-threshold sweep (net of cost):")
    print(hdr); print("-" * len(hdr))
    beat_events = None
    for thr, name in ((2.0, "beat (>2%)"), (10.0, "beat (>10%)"), (25.0, "big beat (>25%)")):
        events = await build_events(price_data, thr)
        trades, eqt = run_config(events, stop_atr=args.stop_atr, target_atr=args.target_atr,
                                 hold=args.hold, cost=cost)
        print(_summ(trades, eqt, name))
        if thr == 2.0:
            beat_events = events

    # Cost sensitivity + a control (inline/small events) at the beat threshold.
    print("\nCost sensitivity (beat >2%):")
    print(hdr); print("-" * len(hdr))
    for cb in (0.0, 7.5, 15.0):
        trades, eqt = run_config(beat_events, stop_atr=args.stop_atr,
                                 target_atr=args.target_atr, hold=args.hold, cost=cb / 10000.0)
        print(_summ(trades, eqt, f"cost {cb:.1f}bp/side"))

    # Sub-period split (regime robustness) at beat >2%.
    trades, eqt = run_config(beat_events, stop_atr=args.stop_atr,
                             target_atr=args.target_atr, hold=args.hold, cost=cost)
    if trades:
        sd = sorted(t["signal_date"] for t in trades)
        edges = (sd[len(sd) // 3], sd[2 * len(sd) // 3])
        print("\nSub-period stability (beat >2%):")
        print(hdr); print("-" * len(hdr))
        for k, nm in enumerate(("early third", "mid third", "late third")):
            sub_t, sub_e = [], []
            for t, e in zip(trades, eqt):
                third = 0 if t["signal_date"] <= edges[0] else (1 if t["signal_date"] <= edges[1] else 2)
                if third == k:
                    sub_t.append(t); sub_e.append(e)
            print(_summ(sub_t, sub_e, nm))
        from collections import Counter
        print("\nexit mix (beat >2%):", dict(Counter(t["exit_reason"] for t in trades)))


if __name__ == "__main__":
    asyncio.run(main())
