"""Follow-up: investigate the 0-day exit anomaly + locate the real sniper score field."""

from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta

from sqlalchemy import select

from src.db.models import Outcome, Signal
from src.db.session import close_db, get_session


async def main() -> None:
    cutoff = date.today() - timedelta(days=60)

    async with get_session() as session:
        result = await session.execute(
            select(Signal)
            .where(
                Signal.signal_model == "sniper",
                Signal.run_date >= cutoff,
                Signal.signal_source == "mas_official",
            )
            .order_by(Signal.run_date.desc())
        )
        sigs = list(result.scalars().all())

        ids = [s.id for s in sigs]
        result = await session.execute(select(Outcome).where(Outcome.signal_id.in_(ids)))
        outs = {o.signal_id: o for o in result.scalars().all()}

    print("=== Signal × Outcome detail (sniper, last 60d) ===")
    print(
        f"{'run_date':<12}{'tkr':<6}{'sig_hold':>9}{'entry_dt':<12}{'exit_dt':<12}"
        f"{'days':>5}{'reason':<14}{'pnl%':>8}{'still_open':>11}"
    )
    for s in sigs:
        o = outs.get(s.id)
        if not o:
            print(f"{str(s.run_date):<12}{s.ticker:<6}  no outcome row")
            continue
        ed = str(o.entry_date) if o.entry_date else "-"
        xd = str(o.exit_date) if o.exit_date else "-"
        days = (
            (o.exit_date - o.entry_date).days if o.exit_date and o.entry_date else None
        )
        reason = o.exit_reason or "-"
        pnl = f"{o.pnl_pct:+.2f}" if o.pnl_pct is not None else "-"
        print(
            f"{str(s.run_date):<12}{s.ticker:<6}{s.holding_period_days:>9}{ed:<12}{xd:<12}"
            f"{(str(days) if days is not None else '-'):>5}{reason:<14}{pnl:>8}{str(o.still_open):>11}"
        )

    # Look at one Outcome.daily_prices to see what was happening intraday
    print("\n=== Sample Outcome.daily_prices for first 3 closed trades ===")
    closed = [
        s for s in sigs if (o := outs.get(s.id)) and not o.still_open and o.exit_date
    ][:3]
    for s in closed:
        o = outs[s.id]
        print(f"\n--- {s.ticker} run={s.run_date} entry={o.entry_date} exit={o.exit_date} reason={o.exit_reason} pnl={o.pnl_pct:+.2f}%")
        print(f"  entry_price={o.entry_price}  stop={s.stop_loss}  target={s.target_1}")
        print(f"  exit_price={o.exit_price}  partial_exit_price={o.partial_exit_price}  partial_exit_date={o.partial_exit_date}")
        print(f"  leg2_exit_reason={o.leg2_exit_reason}")
        if o.daily_prices:
            print("  daily_prices:")
            print("    " + json.dumps(o.daily_prices, indent=2, default=str)[:1500])

    # Look at Signal.features for the real sniper score
    print("\n=== Most recent sniper Signal.features keys ===")
    if sigs:
        recent = sigs[0]
        print(f"ticker={recent.ticker} run_date={recent.run_date} confidence={recent.confidence}")
        if recent.features:
            print("features keys:", list(recent.features.keys()))
            print(json.dumps(recent.features, indent=2, default=str)[:2000])

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
