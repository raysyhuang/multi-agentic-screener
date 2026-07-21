"""Post-backfill sniper scorecard — confirms outcomes table reflects the corrected data."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import date, timedelta
from statistics import mean

from sqlalchemy import select

from src.db.models import Outcome, Signal
from src.db.session import close_db, get_session


async def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Live sniper scorecard from the outcomes DB")
    ap.add_argument("--days", type=int, default=30,
                    help="lookback window in days (sniper is sparse; widen to compare WR vs the ~54%% truth-matrix number)")
    args = ap.parse_args()
    cutoff = date.today() - timedelta(days=args.days)

    async with get_session() as session:
        # All sniper signals in last 30d
        result = await session.execute(
            select(Signal).where(
                Signal.signal_model == "sniper",
                Signal.run_date >= cutoff,
                Signal.signal_source == "mas_official",
            )
        )
        sigs = list(result.scalars().all())

        ids = [s.id for s in sigs]
        result = await session.execute(select(Outcome).where(Outcome.signal_id.in_(ids)))
        outs = list(result.scalars().all())

    closed = [o for o in outs if not o.still_open and o.pnl_pct is not None]
    open_ = [o for o in outs if o.still_open]

    print(f"=== Sniper {args.days}d scorecard (post-backfill) — cutoff {cutoff} ===\n")
    print(f"Total picks:     {len(sigs)}")
    print(f"Closed outcomes: {len(closed)}")
    print(f"Still open:      {len(open_)}\n")

    if closed:
        wins = sum(1 for o in closed if o.pnl_pct > 0)
        losses = len(closed) - wins
        wr = wins / len(closed)
        avg = mean(o.pnl_pct for o in closed)
        sum_pnl = sum(o.pnl_pct for o in closed)
        wins_avg = mean([o.pnl_pct for o in closed if o.pnl_pct > 0]) if wins else 0
        losses_avg = mean([o.pnl_pct for o in closed if o.pnl_pct <= 0]) if losses else 0
        payoff = abs(wins_avg / losses_avg) if losses_avg else float("inf")

        print(f"Win rate:        {wr:.1%}  ({wins}/{len(closed)})")
        print(f"Avg PnL:         {avg:+.2f}%")
        print(f"Sum PnL:         {sum_pnl:+.2f}%")
        print(f"Avg win:         {wins_avg:+.2f}%")
        print(f"Avg loss:        {losses_avg:+.2f}%")
        print(f"Payoff ratio:    {payoff:.2f}")

        print("\n=== Exit reason mix ===")
        reason_counter = Counter(o.exit_reason or "?" for o in closed)
        for r, n in reason_counter.most_common():
            print(f"  {r:<14} {n}")

        print("\n=== Hold duration buckets ===")
        for lo, hi, label in [(0, 1, "0-1d"), (2, 3, "2-3d"), (4, 5, "4-5d"), (6, 7, "6-7d"), (8, 99, "8d+")]:
            rows = [o for o in closed if o.exit_date and (o.exit_date - o.entry_date).days in range(lo, hi + 1)]
            if not rows:
                continue
            wins_b = sum(1 for o in rows if o.pnl_pct > 0)
            print(f"  {label:<8} n={len(rows):>2}  wins={wins_b}  losses={len(rows) - wins_b}  avg={mean(o.pnl_pct for o in rows):+.2f}%")

        print("\n=== All closed sniper trades (post-backfill) ===")
        for o in sorted(closed, key=lambda x: x.entry_date or date.min):
            held = (o.exit_date - o.entry_date).days if o.exit_date and o.entry_date else "?"
            print(f"  {o.ticker:<6} {o.entry_date}  exit={o.exit_date}  held={held}d  {o.exit_reason:<14}  pnl={o.pnl_pct:+.2f}%")

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
