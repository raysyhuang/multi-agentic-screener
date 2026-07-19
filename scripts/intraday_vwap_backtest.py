"""Intraday VWAP-momentum backtest — through the unified exit engine, with costs.

The intraday probe found a monotonic VWAP-momentum edge (names strongly above
session VWAP keep rising into the close). This turns that probe into a real
backtest that can KILL or confirm it:

  * Signal: at a decision time, price is >= threshold above session VWAP.
  * Entry: that minute's close + per-side cost.
  * Exit: ride the remaining minutes through the UNIFIED exit engine (walk_exit)
    with a real intraday stop and expiry at the session close — minute bars fed
    as ExitBars, so fills are minute-resolution and costed on both sides.
  * Reported net of realistic per-side costs, with a cost-sensitivity sweep, a
    threshold sweep, a below-VWAP control (should be negative for longs), and a
    sub-period split (guards against a bull-regime / mega-cap artifact).

Honest constraint: a contiguous minute pull for a wide universe over 3Y is too
heavy for one session, so days are SAMPLED across the full window for regime
coverage. That gives clean per-trade expectancy and sub-period stability, but not
a single compounded equity curve — stated plainly so nobody reads a curve into it.

Usage:
  python scripts/intraday_vwap_backtest.py --cache-file <daily_parquet> \
      --tickers 40 --days 40 [--cost-bps 5] [--stop-pct 0.5]
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.exit_engine import ExitBar, ExitParams, walk_exit
from src.data.intraday_cache import get_intraday_day
from scripts.intraday_mr_probe import pick_liquid_tickers, sample_days

DECISION_TIME = "12:00"   # ET; VWAP well established, still time to run into close
RTH_OPEN, RTH_CLOSE = "09:30", "16:00"


@dataclass
class Trade:
    ticker: str
    day: object
    ret_pct: float          # net of both-side costs
    exit_reason: str


def _rth_with_vwap(bars: pd.DataFrame) -> pd.DataFrame | None:
    if bars.empty:
        return None
    b = bars.copy()
    et = pd.to_datetime(b["datetime"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    b["hm"] = et.dt.strftime("%H:%M")
    rth = b[(b["hm"] >= RTH_OPEN) & (b["hm"] <= RTH_CLOSE)].reset_index(drop=True)
    if len(rth) < 120:
        return None
    pv = (rth["vwap"].astype(float) * rth["volume"].astype(float)).cumsum()
    vol = rth["volume"].astype(float).cumsum().replace(0, np.nan)
    rth["session_vwap"] = pv / vol
    return rth


def simulate_ticker_day(
    rth: pd.DataFrame, *, threshold: float, stop_mode: str, stop_pct: float,
    cost: float, side_long: bool,
) -> tuple[float, str] | None:
    """One trade per qualifying ticker-day. Long side: enter if >= threshold ABOVE
    VWAP; the below-VWAP control uses the mirror condition.

    stop_mode:
      none  — ride to the close (through walk_exit; expiry exit). Matches the probe.
      fixed — fixed intraday stop at stop_pct below entry (through walk_exit).
      vwap  — thesis stop: exit the first minute the price closes back below the
              session VWAP (dynamic level, so computed inline, not via walk_exit).
    Returns (net_ret%, reason)."""
    at = rth[rth["hm"] <= DECISION_TIME]
    if at.empty:
        return None
    i = at.index[-1]
    px = float(rth.iloc[i]["close"])
    vwap = float(rth.iloc[i]["session_vwap"])
    if not np.isfinite(vwap) or vwap <= 0 or px <= 0:
        return None
    dev = (px - vwap) / vwap * 100
    fire = dev >= threshold if side_long else dev <= -threshold
    if not fire:
        return None

    remaining = rth.iloc[i + 1:]
    if remaining.empty:
        return None
    entry = px * (1 + cost)  # long pays the ask; cost applied to entry

    if stop_mode == "vwap":
        # Exit the first minute the close falls back below the session VWAP.
        for _, r in remaining.iterrows():
            v = float(r["session_vwap"])
            c = float(r["close"])
            if np.isfinite(v) and c < v:
                exit_px = c * (1 - cost)
                return (exit_px - entry) / entry * 100, "vwap_cross"
        last = float(remaining.iloc[-1]["close"]) * (1 - cost)
        return (last - entry) / entry * 100, "expiry"

    # none / fixed → unified exit engine, ride to close (expiry) with optional stop.
    stop = entry * (1 - stop_pct / 100) if stop_mode == "fixed" else entry * 0.5
    bars = [
        ExitBar(date=r["hm"], open=float(r["open"]), high=float(r["high"]),
                low=float(r["low"]), close=float(r["close"]))
        for _, r in remaining.iterrows()
    ]
    params = ExitParams(
        stop=stop,
        target=entry * 101,                 # effectively no target — ride to close/stop
        max_hold=len(bars),                 # expiry at the session close
        slippage=cost,                      # exit-side cost
        gap_through=True,
    )
    out = walk_exit(bars, entry, params)
    if out.exited and out.pnl_pct is not None:
        return out.pnl_pct, out.exit_reason
    last = float(remaining.iloc[-1]["close"]) * (1 - cost)
    return (last - entry) / entry * 100, "expiry"


async def run(price_data, tickers, days, *, threshold, stop_mode, stop_pct, cost,
              side_long, _cache):
    """Run one config. _cache maps (ticker, day) -> rth frame to avoid re-pulling
    across the many configs in a single invocation."""
    trades: list[Trade] = []
    failed = 0
    for ticker in tickers:
        for day in days:
            key = (ticker, day)
            if key not in _cache:
                try:
                    bars = await get_intraday_day(ticker, day)
                    _cache[key] = _rth_with_vwap(bars)
                except Exception:
                    failed += 1
                    _cache[key] = None
                    continue
            rth = _cache[key]
            if rth is None:
                continue
            res = simulate_ticker_day(rth, threshold=threshold, stop_mode=stop_mode,
                                      stop_pct=stop_pct, cost=cost, side_long=side_long)
            if res is not None:
                trades.append(Trade(ticker, day, res[0], res[1]))
    return trades, failed


def _summarize(trades: list[Trade], label: str) -> str:
    if not trades:
        return f"{label:<26}{'0 trades':>10}"
    r = np.array([t.ret_pct for t in trades])
    wr = (r > 0).mean() * 100
    return (f"{label:<26}{len(r):>6}{wr:>8.1f}%{r.mean():>10.3f}{np.median(r):>9.3f}"
            f"{r.std(ddof=1) if len(r) > 1 else 0:>8.2f}")


def _third(day, edges):
    return 0 if day <= edges[0] else (1 if day <= edges[1] else 2)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", required=True)
    ap.add_argument("--tickers", type=int, default=40)
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--cost-bps", type=float, default=5.0, help="per-side cost in bps")
    ap.add_argument("--stop-pct", type=float, default=0.5)
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}
    tickers = pick_liquid_tickers(price_data, args.tickers)
    days = sample_days(price_data, args.days)
    cost = args.cost_bps / 10000.0
    rc: dict = {}  # (ticker, day) -> rth frame, shared across configs
    print(f"Universe {len(tickers)} liquid tickers × {len(days)} sampled days; "
          f"decision {DECISION_TIME} ET, cost {args.cost_bps}bp/side")
    print("NOTE: days are SAMPLED across 3Y for regime coverage — per-trade "
          "expectancy, not a compounded equity curve.\n")

    hdr = f"{'config':<26}{'N':>6}{'WR':>9}{'avg%':>10}{'med%':>9}{'sd':>8}"

    async def cfg(threshold, stop_mode, c, side_long=True):
        t, _ = await run(price_data, tickers, days, threshold=threshold,
                         stop_mode=stop_mode, stop_pct=args.stop_pct, cost=c,
                         side_long=side_long, _cache=rc)
        return t

    # 1. Ride-to-close (matches probe), threshold sweep, net of cost.
    print("Long above-VWAP, RIDE TO CLOSE (no stop) — threshold sweep, net of cost:")
    print(hdr); print("-" * len(hdr))
    base_trades = None
    for thr in (0.5, 1.0, 1.5):
        trades = await cfg(thr, "none", cost)
        print(_summarize(trades, f"long dev>=+{thr}%"))
        if thr == 1.0:
            base_trades = trades

    # 2. Below-VWAP control — should be negative for longs if it's momentum.
    ctrl = await cfg(1.0, "none", cost, side_long=False)
    print("\nControl (expect negative if momentum):")
    print(_summarize(ctrl, "long dev<=-1.0%"))

    # 3. VWAP-cross stop variant (thesis risk management) at dev>=+1.0%.
    print("\nRisk-managed variant (exit on close back below VWAP), dev>=+1.0%:")
    print(hdr); print("-" * len(hdr))
    vwap_stop = await cfg(1.0, "vwap", cost)
    print(_summarize(vwap_stop, "vwap-cross stop"))

    # 4. Cost sensitivity, ride-to-close dev>=+1.0%.
    print("\nCost sensitivity (ride to close, dev>=+1.0%):")
    print(hdr); print("-" * len(hdr))
    for cb in (0.0, 5.0, 10.0):
        trades = await cfg(1.0, "none", cb / 10000.0)
        print(_summarize(trades, f"cost {cb:.0f}bp/side"))

    # 5. Sub-period split (regime robustness) of the ride-to-close dev>=+1.0% set.
    if base_trades:
        edges = (days[len(days) // 3], days[2 * len(days) // 3])
        print(f"\nSub-period stability (ride to close, dev>=+1.0%, {args.cost_bps}bp/side):")
        print(hdr); print("-" * len(hdr))
        for k, name in enumerate(("early third", "mid third", "late third")):
            sub = [t for t in base_trades if _third(t.day, edges) == k]
            print(_summarize(sub, name))


if __name__ == "__main__":
    asyncio.run(main())
