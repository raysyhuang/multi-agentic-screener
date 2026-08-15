"""One-off: run the backfill in dry-run mode and dump ALL diffs (no truncation)."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import date, timedelta

from sqlalchemy import text

from scripts.backfill_phantom_exits import (
    BAR_LOOKBACK_DAYS,
    build_candidates_query,
    diff_outcome,
    recompute_exit,
    row_to_context,
    summarize_diffs,
)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", type=lambda s: date.fromisoformat(s), required=True)
    p.add_argument("--end-date", type=lambda s: date.fromisoformat(s), required=True)
    p.add_argument("--exit-reason", default="all")
    args = p.parse_args()

    from src.data.aggregator import DataAggregator
    from src.db.session import close_db, get_session, init_db

    await init_db()
    aggregator = DataAggregator()

    sql, params = build_candidates_query(
        start_date=args.start_date,
        end_date=args.end_date,
        ticker=None,
        outcome_id=None,
        exit_reason=args.exit_reason,
        limit=None,
    )

    async with get_session() as session:
        result = await session.execute(text(sql), params)
        rows = [dict(r) for r in result.mappings()]

    print(f"Scanned {len(rows)} candidate outcome(s)")

    diffs = []
    skipped = 0
    for row in rows:
        ctx = row_to_context(row)
        lookback_start = ctx.entry_date - timedelta(days=3)
        lookback_end = ctx.entry_date + timedelta(days=ctx.holding_days + BAR_LOOKBACK_DAYS)
        df = await aggregator.get_ohlcv(ctx.ticker, lookback_start, lookback_end)
        if df is None or df.empty:
            skipped += 1
            continue
        bars_list = [
            {
                "date": r["date"].date() if hasattr(r["date"], "date") else r["date"],
                "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"],
            }
            for _, r in df.iterrows()
            if (r["date"].date() if hasattr(r["date"], "date") else r["date"]) >= ctx.entry_date
        ]
        new = recompute_exit(ctx, bars_list)
        if new is None:
            skipped += 1
            continue
        d = diff_outcome(ctx, new)
        if d is not None:
            d["signal_model"] = row.get("signal_model")
            diffs.append(d)

    summary = summarize_diffs(diffs)
    print(f"Skipped (inconclusive / no data): {skipped}")
    print(f"Changed: {summary['changed']}")
    print(
        f"P&L delta: {summary['pnl_delta']:+.2f}% "
        f"(old sum {summary['old_pnl_sum']:+.2f}% -> new {summary['new_pnl_sum']:+.2f}%)"
    )

    by_model = Counter(d["signal_model"] for d in diffs)
    print(f"\nChanged rows by signal_model: {dict(by_model)}")

    if summary["transitions"]:
        print("\nExit-reason transitions:")
        for k, v in sorted(summary["transitions"].items()):
            print(f"  {k}: {v}")

    print(f"\n=== ALL {len(diffs)} CHANGED ROWS (sorted by model, entry_date) ===")
    print(
        f"{'ticker':<6}{'entry_dt':<12}{'model':<14}"
        f"{'old_reason':<14}{'old_pnl':>9}  "
        f"{'new_reason':<14}{'new_pnl':>9}  {'delta':>8}"
    )
    diffs_sorted = sorted(diffs, key=lambda d: (str(d.get("signal_model")), d["entry_date"]))
    for d in diffs_sorted:
        old_pnl = d["old"]["pnl_pct"]
        new_pnl = d["new"]["pnl_pct"]
        delta = new_pnl - old_pnl
        marker = "  ↑" if delta > 0.05 else ("  ↓" if delta < -0.05 else "")
        model = d.get("signal_model") or "?"
        print(
            f"{d['ticker']:<6}{str(d['entry_date']):<12}{model:<14}"
            f"{d['old']['exit_reason']:<14}{old_pnl:>+8.2f}%  "
            f"{d['new']['exit_reason']:<14}{new_pnl:>+8.2f}%  {delta:>+7.2f}%{marker}"
        )

    worsened = [d for d in diffs if (d["new"]["pnl_pct"] - d["old"]["pnl_pct"]) < -0.1]
    if worsened:
        print(f"\n=== {len(worsened)} ROWS RECOMPUTE TO WORSE PnL (codex check) ===")
        for d in worsened:
            print(
                f"  {d['ticker']:<6} {d['entry_date']} {d.get('signal_model'):<14} "
                f"{d['old']['exit_reason']}@{d['old']['pnl_pct']:+.2f}% -> "
                f"{d['new']['exit_reason']}@{d['new']['pnl_pct']:+.2f}% "
                f"({d['new']['pnl_pct'] - d['old']['pnl_pct']:+.2f}%)"
            )

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
