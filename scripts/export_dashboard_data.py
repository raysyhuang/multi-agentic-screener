"""Export the MAS dashboard data snapshot from the DB to JSON.

Runs as the final step of the GitHub Actions pipeline jobs (where DATABASE_URL
points at the production Postgres) and writes a single self-contained
`data.json` consumed by the static dashboard (dashboard/). No secrets reach the
page — this bakes a read-only snapshot.

Streams are ALWAYS kept separate (official vs manual sleeve — never blended;
see CLAUDE.md). Baseline expectation bands are the honest, truth-matrix /
reconciliation numbers, not the retired optimistic labels.

Usage:
  python scripts/export_dashboard_data.py [--out dashboard/data.json] [--days 90]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select

from src.db.models import DailyRun, Outcome, Signal
from src.db.session import get_session

# Honest expectation bands (per-trade), from the 2026-07 truth work:
#  - sniper: truth-matrix Run E (live-faithful fills): ~54.3% WR / +0.54%/trade
#  - MR official: reconciled 90d live +0.46%/trade (n=23, provisional)
#  - MR sleeve: reconciled ~breakeven (-0.01%)
BASELINES = {
    "sniper|mas_official": {"label": "Sniper (official)", "wr": 0.543, "avg": 0.54,
                            "source": "truth-matrix Run E (2026-07-19)"},
    "mean_reversion|mas_official": {"label": "MR (official)", "wr": 0.522, "avg": 0.46,
                                    "source": "90d reconciliation (provisional, n=23)"},
    "mean_reversion|mr_manual_sleeve": {"label": "MR (manual sleeve)", "wr": 0.493, "avg": -0.01,
                                        "source": "90d reconciliation"},
}


def _iso(d) -> str | None:
    return d.isoformat() if d is not None else None


def _stream_key(model: str | None, source: str | None) -> str:
    return f"{model or 'unknown'}|{source or 'unknown'}"


async def build_snapshot(days: int = 90) -> dict:
    cutoff = date.today() - timedelta(days=days)

    async with get_session() as session:
        runs = (await session.execute(
            select(DailyRun).where(DailyRun.run_date >= cutoff).order_by(DailyRun.run_date)
        )).scalars().all()

        signals = (await session.execute(
            select(Signal).where(Signal.run_date >= cutoff)
        )).scalars().all()

        sig_ids = [s.id for s in signals]
        outcomes = (await session.execute(
            select(Outcome).where(Outcome.signal_id.in_(sig_ids))
        )).scalars().all() if sig_ids else []

    sig_by_id = {s.id: s for s in signals}
    latest_run = runs[-1] if runs else None

    # --- Today: picks of the latest run date ---
    today_picks = []
    if latest_run:
        for s in signals:
            if s.run_date != latest_run.run_date:
                continue
            feats = s.features or {}
            today_picks.append({
                "ticker": s.ticker,
                "model": s.signal_model,
                "source": s.signal_source,
                "confidence": s.confidence,
                "raw_score": feats.get("model_raw_score"),
                "components": feats.get("model_components") or {},
                "entry": s.entry_price,
                "stop": s.stop_loss,
                "target1": s.target_1,
                "hold_days": s.holding_period_days,
                "regime": s.regime,
            })

    # --- Trades: closed, skipped, open — per stream, never blended ---
    trades: dict[str, list[dict]] = {}
    open_positions = []
    skip_counts: dict[str, int] = {}
    for o in outcomes:
        s = sig_by_id.get(o.signal_id)
        if s is None:
            continue
        key = _stream_key(s.signal_model, s.signal_source)
        if o.skip_reason:
            skip_counts[key] = skip_counts.get(key, 0) + 1
            continue
        if o.still_open:
            open_positions.append({
                "ticker": o.ticker, "stream": key,
                "entry_date": _iso(o.entry_date),
                "days_held": (date.today() - o.entry_date).days if o.entry_date else None,
                "unrealized_pnl_pct": o.pnl_pct,
            })
            continue
        if o.pnl_pct is None:
            continue
        trades.setdefault(key, []).append({
            "ticker": o.ticker,
            "signal_date": _iso(s.run_date),
            "entry_date": _iso(o.entry_date),
            "exit_date": _iso(o.exit_date),
            "exit_reason": o.exit_reason,
            "pnl_pct": round(o.pnl_pct, 4),
            "mfe": o.max_favorable,
            "mae": o.max_adverse,
            "hold_days": (o.exit_date - o.entry_date).days
                         if (o.exit_date and o.entry_date) else None,
        })
    for key in trades:
        trades[key].sort(key=lambda t: (t["exit_date"] or "", t["ticker"]))

    # --- System: run history + health ---
    run_history = [{
        "date": _iso(r.run_date),
        "regime": r.regime,
        "universe": r.universe_size,
        "candidates": r.candidates_scored,
        "duration_s": r.pipeline_duration_s,
        "mode": r.execution_mode,
        "health": (r.pipeline_health or {}).get("status")
                  if isinstance(r.pipeline_health, dict) else None,
        "warnings": (r.pipeline_health or {}).get("warnings")
                    if isinstance(r.pipeline_health, dict) else None,
    } for r in runs]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "latest_run": {
            "date": _iso(latest_run.run_date) if latest_run else None,
            "regime": latest_run.regime if latest_run else None,
            "universe": latest_run.universe_size if latest_run else None,
            "candidates": latest_run.candidates_scored if latest_run else None,
        },
        "today_picks": today_picks,
        "trades": trades,
        "open_positions": open_positions,
        "skip_counts": skip_counts,
        "run_history": run_history,
        "baselines": BASELINES,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard/data.json")
    ap.add_argument("--days", type=int, default=90)
    args = ap.parse_args()

    snap = await build_snapshot(days=args.days)
    with open(args.out, "w") as f:
        json.dump(snap, f, default=str)
    n_trades = sum(len(v) for v in snap["trades"].values())
    print(f"Wrote {args.out}: {len(snap['today_picks'])} picks today, "
          f"{n_trades} closed trades ({args.days}d), {len(snap['open_positions'])} open, "
          f"{len(snap['run_history'])} runs")


if __name__ == "__main__":
    asyncio.run(main())
