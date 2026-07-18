"""One-off sniper diagnostic — answers the four questions from the openclaw critique.

1. Sniper losses bucketed by exit reason (stop / trail / time-expiry / target)
2. Win/loss by hold duration
3. Current PnL on open sniper trades
4. Score distribution across all sniper picks (last 30d) — test "everything gets 65" hypothesis
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from datetime import date, timedelta
from statistics import mean, median, stdev

from sqlalchemy import select

from src.db.models import Outcome, Signal
from src.db.session import close_db, get_session


LOOKBACK_DAYS = 60  # wider than 30 to capture more closed sniper trades


async def main() -> None:
    cutoff = date.today() - timedelta(days=LOOKBACK_DAYS)

    async with get_session() as session:
        # All sniper signals in lookback
        result = await session.execute(
            select(Signal).where(
                Signal.signal_model == "sniper",
                Signal.run_date >= cutoff,
                Signal.signal_source == "mas_official",
            )
        )
        sniper_signals = list(result.scalars().all())

        sig_ids = [s.id for s in sniper_signals]
        if not sig_ids:
            print("No sniper signals in lookback.")
            return

        result = await session.execute(
            select(Outcome).where(Outcome.signal_id.in_(sig_ids))
        )
        outcomes = list(result.scalars().all())

    out_by_sig = {o.signal_id: o for o in outcomes}

    # Helpers
    def hold_days(o: Outcome) -> int | None:
        if o.exit_date and o.entry_date:
            return (o.exit_date - o.entry_date).days
        return None

    closed = [o for o in outcomes if not o.still_open and o.pnl_pct is not None]
    open_ = [o for o in outcomes if o.still_open]

    print(f"=== SNIPER DIAGNOSTIC (lookback {LOOKBACK_DAYS}d, cutoff {cutoff}) ===\n")
    print(f"Total sniper signals: {len(sniper_signals)}")
    print(f"  closed outcomes: {len(closed)}")
    print(f"  still open:      {len(open_)}\n")

    # ---------- Q1 + Q2: exit reason × win/loss × hold duration ----------
    print("=== Q1: Exit reason breakdown (closed only) ===")
    reason_counter: Counter[str] = Counter()
    reason_pnl: defaultdict[str, list[float]] = defaultdict(list)
    reason_hold: defaultdict[str, list[int]] = defaultdict(list)
    for o in closed:
        r = o.exit_reason or "unknown"
        reason_counter[r] += 1
        reason_pnl[r].append(o.pnl_pct)
        h = hold_days(o)
        if h is not None:
            reason_hold[r].append(h)

    print(f"{'reason':<14}{'n':>4}{'wins':>6}{'losses':>8}{'avg_pnl%':>11}{'avg_hold_d':>13}")
    for r in sorted(reason_counter, key=lambda x: -reason_counter[x]):
        pnls = reason_pnl[r]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        ah = mean(reason_hold[r]) if reason_hold[r] else float("nan")
        print(
            f"{r:<14}{reason_counter[r]:>4}{wins:>6}{losses:>8}"
            f"{mean(pnls):>10.2f}%{ah:>12.1f}"
        )

    # Headline win-rate sanity check
    total_wins = sum(1 for o in closed if o.pnl_pct and o.pnl_pct > 0)
    total_pnl = sum(o.pnl_pct for o in closed if o.pnl_pct is not None)
    if closed:
        print(
            f"\nOverall closed: WR={total_wins / len(closed):.1%} "
            f"({total_wins}/{len(closed)}), "
            f"avg pnl={total_pnl / len(closed):+.2f}%, "
            f"sum pnl={total_pnl:+.2f}%"
        )

    # ---------- Q2: hold duration buckets ----------
    print("\n=== Q2: Win/loss by hold duration (closed only) ===")
    buckets = [(0, 1, "0-1d"), (2, 3, "2-3d"), (4, 5, "4-5d"), (6, 7, "6-7d"), (8, 99, "8d+")]
    print(f"{'bucket':<10}{'n':>4}{'wins':>6}{'losses':>8}{'WR':>8}{'avg_pnl%':>11}")
    for lo, hi, label in buckets:
        rows = [o for o in closed if (h := hold_days(o)) is not None and lo <= h <= hi]
        if not rows:
            print(f"{label:<10}{0:>4}{'-':>6}{'-':>8}{'-':>8}{'-':>11}")
            continue
        wins = sum(1 for o in rows if o.pnl_pct > 0)
        wr = wins / len(rows)
        avg = mean(o.pnl_pct for o in rows)
        print(
            f"{label:<10}{len(rows):>4}{wins:>6}{len(rows) - wins:>8}"
            f"{wr:>7.1%}{avg:>10.2f}%"
        )

    # ---------- Q3: open trade status ----------
    print("\n=== Q3: Open sniper trades (current state) ===")
    if not open_:
        print("None open.")
    else:
        sig_by_id = {s.id: s for s in sniper_signals}
        print(
            f"{'ticker':<8}{'entry_date':<12}{'entry':>9}{'stop':>9}{'target':>9}"
            f"{'mfe%':>8}{'mae%':>8}{'days_held':>11}"
        )
        today = date.today()
        for o in sorted(open_, key=lambda x: x.entry_date or date.min):
            sig = sig_by_id.get(o.signal_id)
            held = (today - o.entry_date).days if o.entry_date else None
            mfe = o.max_favorable if o.max_favorable is not None else float("nan")
            mae = o.max_adverse if o.max_adverse is not None else float("nan")
            stop = sig.stop_loss if sig else float("nan")
            tgt = sig.target_1 if sig else float("nan")
            print(
                f"{o.ticker:<8}{str(o.entry_date):<12}{o.entry_price:>9.2f}"
                f"{stop:>9.2f}{tgt:>9.2f}{mfe:>7.2f}%{mae:>7.2f}%"
                f"{(held if held is not None else -1):>11}"
            )

    # ---------- Q4: score distribution ----------
    print("\n=== Q4: Sniper score (confidence) distribution — last 30d picks ===")
    cutoff30 = date.today() - timedelta(days=30)
    scores_30 = [s.confidence for s in sniper_signals if s.run_date >= cutoff30]
    if scores_30:
        print(f"n={len(scores_30)}")
        print(f"  min  = {min(scores_30):.2f}")
        print(f"  p25  = {sorted(scores_30)[len(scores_30) // 4]:.2f}")
        print(f"  med  = {median(scores_30):.2f}")
        print(f"  p75  = {sorted(scores_30)[3 * len(scores_30) // 4]:.2f}")
        print(f"  max  = {max(scores_30):.2f}")
        print(f"  mean = {mean(scores_30):.2f}")
        if len(scores_30) > 1:
            print(f"  std  = {stdev(scores_30):.2f}")
        # Count of identical scores
        score_counter = Counter(round(s, 2) for s in scores_30)
        top = score_counter.most_common(5)
        print("\nTop score buckets (rounded):")
        for v, n in top:
            print(f"  {v:>6.2f}  ×{n}")
    else:
        print("No sniper picks in last 30d.")

    print("\n=== Full sniper score distribution (lookback) ===")
    all_scores = [s.confidence for s in sniper_signals]
    print(f"n={len(all_scores)}")
    if all_scores:
        print(
            f"min={min(all_scores):.2f}  med={median(all_scores):.2f}  "
            f"max={max(all_scores):.2f}  mean={mean(all_scores):.2f}  "
            f"std={stdev(all_scores) if len(all_scores) > 1 else 0:.2f}"
        )

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
