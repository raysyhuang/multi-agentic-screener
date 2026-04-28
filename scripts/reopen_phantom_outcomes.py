"""One-off: reopen the 4 sniper outcomes whose recorded exit is a phantom
that the backfill couldn't recompute (insufficient post-entry OHLCV).

Strategy: revert each row to "still open with entry intact" so the live
afternoon health check (running on Heroku v212 with the fixed simulator)
manages the lifecycle from the next pass onward.

Tickers: QXO (2026-04-22), LEU (2026-04-24), NXT (2026-04-27), OII (2026-04-27).

Safeguards:
1. Snapshot each row to backups/phantom_backfill/reopen_snapshot_<ts>.json
   before any UPDATE.
2. Refuse if any matching SignalExitEvent rows exist (would leave
   contradictory exit-history state).
3. Run inside a single transaction; rollback on any error.

Run with --apply to actually write. Default is dry-run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import select, text

from src.db.models import Outcome, Signal, SignalExitEvent
from src.db.session import close_db, get_session


TARGETS = [
    ("QXO", "2026-04-22"),
    ("LEU", "2026-04-24"),
    ("NXT", "2026-04-27"),
    ("OII", "2026-04-27"),
]

SNAPSHOT_DIR = Path("backups/phantom_backfill")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    args = p.parse_args()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    async with get_session() as session:
        # Resolve target outcome rows.
        rows: list[tuple[Outcome, Signal]] = []
        for ticker, entry_date_str in TARGETS:
            entry_d = date.fromisoformat(entry_date_str)
            stmt = (
                select(Outcome, Signal)
                .join(Signal, Signal.id == Outcome.signal_id)
                .where(
                    Outcome.ticker == ticker,
                    Outcome.entry_date == entry_d,
                    Signal.signal_model == "sniper",
                )
            )
            result = await session.execute(stmt)
            matches = list(result.all())
            if not matches:
                print(f"WARN: no row found for {ticker} {entry_date_str}")
                continue
            if len(matches) > 1:
                print(f"WARN: multiple rows for {ticker} {entry_date_str} — refusing")
                return
            rows.append(matches[0])

        if not rows:
            print("No target rows resolved. Nothing to do.")
            return

        signal_ids = [s.id for _, s in rows]

        # Safety check: no SignalExitEvent rows for these signals.
        result = await session.execute(
            select(SignalExitEvent).where(SignalExitEvent.signal_id.in_(signal_ids))
        )
        exit_events = list(result.scalars().all())
        if exit_events:
            print(
                f"REFUSE: {len(exit_events)} SignalExitEvent row(s) exist for these "
                "signal_ids. Reopening would leave contradictory state. Aborting."
            )
            for ev in exit_events:
                print(f"  signal_id={ev.signal_id} event_type={ev.event_type} ts={ev.ts}")
            return

        # Build snapshot
        snapshot = []
        print(f"=== Pre-state for {len(rows)} target rows ===")
        for o, s in rows:
            snap = {
                "outcome_id": o.id,
                "signal_id": s.id,
                "ticker": o.ticker,
                "entry_date": o.entry_date.isoformat(),
                "entry_price": o.entry_price,
                "signal_model": s.signal_model,
                "old": {
                    "still_open": o.still_open,
                    "exit_date": o.exit_date.isoformat() if o.exit_date else None,
                    "exit_price": o.exit_price,
                    "exit_reason": o.exit_reason,
                    "pnl_pct": o.pnl_pct,
                    "pnl_dollars": o.pnl_dollars,
                    "leg2_exit_reason": o.leg2_exit_reason,
                    "max_favorable": o.max_favorable,
                    "max_adverse": o.max_adverse,
                    "daily_prices_keys": (
                        list(o.daily_prices.keys()) if o.daily_prices else None
                    ),
                },
            }
            snapshot.append(snap)
            print(
                f"  {o.ticker:<6} entry={o.entry_date}  open={o.still_open}  "
                f"exit={o.exit_date}@{o.exit_reason}  pnl={o.pnl_pct:+.2f}%"
            )

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snapshot_path = SNAPSHOT_DIR / f"reopen_snapshot_{ts}.json"
        snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str))
        print(f"\nSnapshot written: {snapshot_path}")

        if not args.apply:
            print("\nDry run. Pass --apply to reopen these rows.")
            return

        # Apply: clear exit-related fields, set still_open=True.
        for o, _s in rows:
            await session.execute(
                text(
                    """
                    UPDATE outcomes
                    SET still_open       = TRUE,
                        exit_date        = NULL,
                        exit_price       = NULL,
                        exit_reason      = NULL,
                        pnl_pct          = NULL,
                        pnl_dollars      = NULL,
                        leg2_exit_reason = NULL,
                        max_favorable    = NULL,
                        max_adverse      = NULL,
                        daily_prices     = NULL
                    WHERE id = :outcome_id
                    """
                ),
                {"outcome_id": o.id},
            )
        await session.commit()
        print(f"\nApplied: reopened {len(rows)} outcome(s).")

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
