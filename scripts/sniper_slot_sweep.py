"""Would more concurrent slots deliver 2 sniper picks per day — and at what cost?

Ray's expectation is 2 sniper picks every trading day. Live throughput is lower:
mean hold is 1.48d and `sniper_max_positions` is 3, so 2/day implies ~3 open at
all times, sitting exactly on the cap. Some days it binds and the run produces
nothing.

PR #56 (`sniper_pick_count.py`) does NOT answer this. It varied the daily quota
`k` at a FIXED 3 slots and rejected widening. This varies the SLOTS at a fixed
k=2, which has a different mechanism: you still take the top 2, you just hold
more concurrently.

The methodological catch
------------------------
`simulate_book` sizes each position at `equity / max_concurrent`, so raising the
cap silently shrinks every position. Comparing returns across caps with that
model conflates two changes — more slots AND less capital per trade — and the
return decline it shows is mostly the sizing, not the slots. It cannot answer
"should I raise the cap".

So both capital models are reported:

  divided  slot count divides the account (the existing model). Raising the cap
           de-risks by shrinking positions; gross exposure is constant.
  fixed    position size stays at equity/3 — today's live sizing — and extra
           slots ADD gross exposure. This is the change actually under
           discussion, and the one where drawdown is the thing to watch.

Usage:
    python scripts/sniper_slot_sweep.py [--k 2] [--cohort PATH]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.portfolio import BookTrade

DEFAULT_COHORT = "outputs/research/sniper_truth_E_live_fixed_2026-07-26.csv"
LIVE_SLOTS = 3


@dataclass
class Result:
    taken: int
    skipped: int
    peak_concurrent: int
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float | None
    blocked_days: int


def simulate(
    trades: list[BookTrade],
    *,
    max_concurrent: int,
    size_divisor: int | None = None,
    start_capital: float = 100_000.0,
) -> Result:
    """Replay with slot count and position size decoupled.

    ``size_divisor`` is what each position is sized against. Passing
    ``max_concurrent`` reproduces the existing model exactly; passing a constant
    holds position size fixed while the cap varies, which is the comparison this
    script exists for.
    """
    divisor = size_divisor or max_concurrent
    trades = [t for t in trades if t.entry and t.exit and t.pnl_pct is not None]
    trades = sorted(trades, key=lambda t: (t.entry, t.exit))

    events: list[tuple[date, int, BookTrade]] = []
    for t in trades:
        events.append((t.exit, 0, t))
        events.append((t.entry, 1, t))
    events.sort(key=lambda e: (e[0], e[1]))

    cash = start_capital
    open_positions: dict[int, float] = {}
    taken = skipped = peak = 0
    blocked_days: set[date] = set()
    curve: list[tuple[date, float]] = [(events[0][0], start_capital)]

    def equity() -> float:
        return cash + sum(open_positions.values())

    for ev_date, kind, t in events:
        tid = id(t)
        if kind == 0:
            if tid in open_positions:
                notional = open_positions.pop(tid)
                cash += notional * (1.0 + t.pnl_pct / 100.0)
                curve.append((ev_date, equity()))
        else:
            if len(open_positions) >= max_concurrent:
                skipped += 1
                blocked_days.add(ev_date)
                continue
            notional = min(equity() / divisor, cash)
            if notional <= 0:
                skipped += 1
                blocked_days.add(ev_date)
                continue
            cash -= notional
            open_positions[tid] = notional
            taken += 1
            peak = max(peak, len(open_positions))

    final_eq = equity()
    running_peak, max_dd = -float("inf"), 0.0
    for _, eq in curve:
        running_peak = max(running_peak, eq)
        if running_peak > 0:
            max_dd = max(max_dd, (running_peak - eq) / running_peak)

    rets = []
    for i in range(1, len(curve)):
        prev = curve[i - 1][1]
        if prev > 0:
            rets.append((curve[i][1] - prev) / prev)
    sharpe = None
    if len(rets) > 2:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sd = var ** 0.5
        sharpe = (mean / sd) * (252 ** 0.5) if sd > 0 else None

    return Result(
        taken=taken,
        skipped=skipped,
        peak_concurrent=peak,
        total_return_pct=(final_eq / start_capital - 1.0) * 100.0,
        max_drawdown_pct=max_dd * 100.0,
        sharpe=sharpe,
        blocked_days=len(blocked_days),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=DEFAULT_COHORT)
    ap.add_argument("--k", type=int, default=2, help="daily quota, live is 2")
    args = ap.parse_args()

    df = pd.read_csv(args.cohort)
    df = df.sort_values(["entry_date", "score"], ascending=[True, False]).copy()
    df["rank"] = df.groupby("entry_date").cumcount() + 1
    sub = df[df["rank"] <= args.k]

    book: list[BookTrade] = []
    for r in sub.itertuples():
        try:
            book.append(BookTrade(
                entry=date.fromisoformat(str(r.entry_date)[:10]),
                exit=date.fromisoformat(str(r.exit_date)[:10]),
                pnl_pct=float(r.pnl_pct),
            ))
        except (ValueError, TypeError):
            continue

    days = sub["entry_date"].nunique()
    print(f"cohort: {len(sub)} signals at k<={args.k} over {days} entry days "
          f"({len(sub)/days:.2f}/day)\n")

    for label, divisor in (("divided (cap sizes the position)", None),
                           (f"fixed (position stays equity/{LIVE_SLOTS})", LIVE_SLOTS)):
        print("=" * 92)
        print(f"CAPITAL MODEL: {label}   |   k={args.k}")
        print("=" * 92)
        print(f"  {'slots':>6}{'taken':>7}{'skipped':>9}{'blocked days':>14}"
              f"{'peak':>6}{'return':>10}{'maxDD':>9}{'Sharpe':>8}")
        for cap in (3, 4, 5, 6, 8, 10):
            r = simulate(book, max_concurrent=cap, size_divisor=divisor)
            sh = "–" if r.sharpe is None else f"{r.sharpe:.2f}"
            live = "  <- live" if cap == LIVE_SLOTS else ""
            print(f"  {cap:>6}{r.taken:>7}{r.skipped:>9}{r.blocked_days:>14}"
                  f"{r.peak_concurrent:>6}{r.total_return_pct:>+9.1f}%"
                  f"{r.max_drawdown_pct:>8.1f}%{sh:>8}{live}")
        print()


if __name__ == "__main__":
    main()
