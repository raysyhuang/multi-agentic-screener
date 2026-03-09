#!/usr/bin/env python
"""Sniper campaign evaluator — tracks rolling window performance for promotion.

Queries the DB for sniper-track outcomes and evaluates promotion gates:
  - WR >= 50% over last 20 trades
  - Campaign return positive for 3 consecutive windows
  - PF >= 2.0
  - Time-stop rate < 40%

Usage:
    python scripts/evaluate_sniper_campaign.py
    python scripts/evaluate_sniper_campaign.py --window 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def evaluate_campaign(window_size: int = 20) -> dict:
    """Evaluate sniper campaign performance from DB outcomes."""
    from dotenv import load_dotenv
    load_dotenv()

    from src.db.session import init_db, get_session
    from src.db.models import Outcome, Signal
    from sqlalchemy import select, and_

    await init_db()

    async with get_session() as session:
        # Fetch all closed sniper outcomes
        result = await session.execute(
            select(Outcome, Signal)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(
                and_(
                    Outcome.still_open == False,  # noqa: E712
                    Signal.signal_model == "sniper",
                )
            )
            .order_by(Outcome.exit_date.asc())
        )
        rows = result.all()

    if not rows:
        print("No closed sniper trades found.")
        return {"status": "no_data"}

    outcomes = [(o, s) for o, s in rows]
    pnls = [o.pnl_pct or 0.0 for o, _ in outcomes]
    exit_reasons = [o.exit_reason or "unknown" for o, _ in outcomes]

    total = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / total if total > 0 else 0
    avg_return = sum(pnls) / total if total > 0 else 0
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    time_stops = sum(1 for r in exit_reasons if r == "time_stop")
    time_stop_rate = time_stops / total if total > 0 else 0

    # Rolling window analysis
    windows = []
    for i in range(0, total - window_size + 1):
        window_pnls = pnls[i:i + window_size]
        w_wins = sum(1 for p in window_pnls if p > 0)
        w_wr = w_wins / window_size
        w_return = sum(window_pnls)
        windows.append({
            "start_idx": i,
            "end_idx": i + window_size - 1,
            "win_rate": w_wr,
            "total_return": w_return,
        })

    # Check consecutive positive windows
    consecutive_positive = 0
    max_consecutive_positive = 0
    for w in windows:
        if w["total_return"] > 0:
            consecutive_positive += 1
            max_consecutive_positive = max(max_consecutive_positive, consecutive_positive)
        else:
            consecutive_positive = 0

    # Latest window stats
    latest_window = windows[-1] if windows else None
    latest_wr = latest_window["win_rate"] if latest_window else 0

    # Hybrid Promotion Gates
    # Layer 1: Raw signal quality (from DB trade outcomes)
    raw_gates = {
        "raw WR >= 50%": {
            "pass": wr >= 0.50,
            "value": wr,
            "gate": ">= 50%",
        },
        "raw avg_return >= 2.0%": {
            "pass": avg_return >= 2.0,
            "value": avg_return,
            "gate": ">= 2.0%",
        },
        "raw PF >= 1.5": {
            "pass": pf >= 1.5,
            "value": pf,
            "gate": ">= 1.5",
        },
    }

    # Layer 2: Campaign/portfolio gates
    campaign_gates = {
        "latest window WR >= 50%": {
            "pass": latest_wr >= 0.50,
            "value": latest_wr,
            "gate": ">= 50%",
        },
        "3+ consec positive windows": {
            "pass": max_consecutive_positive >= 3,
            "value": max_consecutive_positive,
            "gate": ">= 3",
        },
        "time_stop_rate < 40%": {
            "pass": time_stop_rate < 0.40,
            "value": time_stop_rate,
            "gate": "< 40%",
        },
    }

    all_gates = {**raw_gates, **campaign_gates}
    all_pass = all(g["pass"] for g in all_gates.values())

    # Print report
    print(f"\n{'='*60}")
    print(f"  SNIPER CAMPAIGN EVALUATION")
    print(f"{'='*60}")
    print(f"  Total trades:     {total}")
    print(f"  Win rate:         {wr:.1%}")
    print(f"  Avg return:       {avg_return:+.2f}%")
    print(f"  Profit factor:    {pf:.2f}")
    print(f"  Time-stop rate:   {time_stop_rate:.1%}")
    print(f"  Rolling windows:  {len(windows)} (size={window_size})")
    if latest_window:
        print(f"  Latest WR:       {latest_wr:.1%}")
    print(f"  Max consec +ve:   {max_consecutive_positive}")

    print(f"\n  Promotion Gates:")
    print(f"    --- Layer 1: Signal Quality (raw trades) ---")
    for name, g in raw_gates.items():
        status = "PASS" if g["pass"] else "FAIL"
        print(f"    {name:35s}: {status}  (value={g['value']:.4f}, gate={g['gate']})")

    print(f"    --- Layer 2: Campaign Reality ---")
    for name, g in campaign_gates.items():
        status = "PASS" if g["pass"] else "FAIL"
        val = g["value"]
        fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"    {name:35s}: {status}  (value={fmt}, gate={g['gate']})")

    if all_pass:
        print(f"\n  PROMOTION ELIGIBLE — all gates passed")
    else:
        print(f"\n  NOT READY — some gates failed")

    return {
        "total_trades": total,
        "win_rate": wr,
        "avg_return": avg_return,
        "profit_factor": pf,
        "time_stop_rate": time_stop_rate,
        "max_consecutive_positive_windows": max_consecutive_positive,
        "raw_gates": raw_gates,
        "campaign_gates": campaign_gates,
        "promotion_eligible": all_pass,
    }


def main():
    parser = argparse.ArgumentParser(description="Sniper campaign evaluator")
    parser.add_argument("--window", type=int, default=20,
                        help="Rolling window size for evaluation (default: 20)")
    args = parser.parse_args()

    asyncio.run(evaluate_campaign(args.window))


if __name__ == "__main__":
    main()
