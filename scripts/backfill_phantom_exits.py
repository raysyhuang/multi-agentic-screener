"""Backfill historical phantom exits in the `outcomes` table.

Purpose
-------
Recompute closed outcomes that were recorded under the buggy pre-fix
simulator (commit 58648fa or earlier) so the trade log reflects the
post-fix (c75e921) semantics. Defaults to dry-run; `--apply` required
to mutate rows. Every apply writes a timestamped JSON snapshot of the
affected rows BEFORE issuing any UPDATE.

Scope
-----
Targets the same narrow issues the forward fix addressed:
  1. Same-bar `trail_stop` phantoms from newly-armed trails
  2. Same-bar `stop` phantoms from newly-filled leg1 breakeven pivots

The recompute logic mirrors `_evaluate_position` in
`src/output/performance.py` but is duplicated here deliberately — we want
a frozen snapshot of the corrected logic that won't drift if the
simulator is refactored later, and we avoid the DB/daily_prices
side-effects of the live evaluator.

Usage
-----
Dry run (default):
    python scripts/backfill_phantom_exits.py \
        --start-date 2026-01-01 --end-date 2026-04-11

Apply changes (writes snapshot first):
    python scripts/backfill_phantom_exits.py \
        --start-date 2026-03-26 --end-date 2026-04-11 --apply

Notes
-----
- Candidate selection defaults to `exit_reason='trail_stop'`. To include
  other reasons (e.g. `stop` for leg1 phantoms), pass `--exit-reason all`.
- Only closed outcomes (still_open=False) are considered.
- Skipped outcomes (skip_reason not null) are never touched.
- A run with no `--start-date` / `--end-date` refuses to proceed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


# --- Constants frozen at fix time (c75e921) ---
# If production config changes these, bump this file with a note.
TRAIL_ACTIVATE_MR = 0.5
TRAIL_ACTIVATE_SNIPER = 1.0
TRAIL_DISTANCE_MR = 0.3
TRAIL_DISTANCE_SNIPER = 0.5
SLIPPAGE = 0.001
SCORE_TIERED_STOPS = True
SNIPER_TIME_STOP_DAYS = 7
BAR_LOOKBACK_DAYS = 5


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TradeContext:
    """Minimal per-trade state needed to replay a closed outcome."""

    outcome_id: int
    ticker: str
    signal_model: str
    confidence: float | None
    entry_date: date
    entry_price: float           # outcome.entry_price (actual fill)
    planned_entry: float         # signal.entry_price
    planned_stop: float          # signal.stop_loss
    target: float                # signal.target_1
    holding_days: int            # signal.holding_period_days
    old_exit_reason: str
    old_exit_price: float | None
    old_exit_date: date | None
    old_pnl_pct: float | None


@dataclass
class RecomputeResult:
    exit_reason: str
    exit_price: float
    exit_date: date
    pnl_pct: float
    mfe: float
    mae: float


# ---------------------------------------------------------------------------
# Pure logic — no DB access
# ---------------------------------------------------------------------------


def compute_base_stop(ctx: TradeContext) -> float:
    """Replicate the score-tiered base-stop logic from _evaluate_position."""
    if not SCORE_TIERED_STOPS or ctx.confidence is None:
        return ctx.planned_stop
    tier_atr = abs(ctx.planned_entry - ctx.planned_stop) / 0.75 if ctx.planned_stop else 0.0
    if tier_atr <= 0:
        return ctx.planned_stop
    if ctx.confidence >= 85:
        return round(ctx.entry_price - 1.25 * tier_atr, 2)
    if ctx.confidence >= 70:
        return round(ctx.entry_price - 0.85 * tier_atr, 2)
    return round(ctx.entry_price - 0.50 * tier_atr, 2)


def recompute_exit(ctx: TradeContext, bars: list[dict[str, Any]]) -> RecomputeResult | None:
    """Replay the bar walk using the FIXED simulator logic.

    `bars` must be a list of dicts with keys {date, open, high, low, close}
    starting at or after ctx.entry_date, sorted ascending.

    Returns None if there are no bars to walk or the trade never exited
    within the holding window (inconclusive — caller should skip these).
    """
    if not bars:
        return None

    base_stop = compute_base_stop(ctx)
    target = ctx.target
    trail_activate = (
        TRAIL_ACTIVATE_SNIPER if ctx.signal_model == "sniper" else TRAIL_ACTIVATE_MR
    )
    trail_distance = (
        TRAIL_DISTANCE_SNIPER if ctx.signal_model == "sniper" else TRAIL_DISTANCE_MR
    )

    high_watermark = ctx.entry_price
    trailing_active = False
    mfe = 0.0
    mae = 0.0
    days_traded = 0

    for bar in bars:
        bar_date = bar["date"]
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])
        days_traded += 1

        high_watermark = max(high_watermark, bar_high)
        mfe = max(mfe, (bar_high - ctx.entry_price) / ctx.entry_price * 100)
        mae = min(mae, (bar_low - ctx.entry_price) / ctx.entry_price * 100)

        # Narrow trail fix: defer newly-armed trails to the next bar
        trail_just_activated = False
        if not trailing_active:
            gain_pct = (high_watermark - ctx.entry_price) / ctx.entry_price * 100
            if gain_pct >= trail_activate:
                trailing_active = True
                trail_just_activated = True

        effective_stop = base_stop
        if trailing_active and not trail_just_activated:
            trail_stop = high_watermark * (1 - trail_distance / 100)
            effective_stop = max(base_stop, trail_stop)

        # Leg1 partial-TP is disabled in production (partial_tp_enabled=False),
        # so the breakeven pivot path is intentionally omitted here. If leg1
        # is ever re-enabled, this script needs an updated version.

        # Gap-through stop
        if bar_open <= effective_stop:
            exit_price = round(bar_open * (1 - SLIPPAGE), 4)
            reason = (
                "trail_stop"
                if trailing_active and not trail_just_activated and effective_stop > base_stop
                else "stop"
            )
            pnl = (exit_price - ctx.entry_price) / ctx.entry_price * 100
            return RecomputeResult(reason, exit_price, bar_date, round(pnl, 4), round(mfe, 4), round(mae, 4))

        # Intraday stop
        if bar_low <= effective_stop:
            exit_price = round(effective_stop * (1 - SLIPPAGE), 4)
            reason = (
                "trail_stop"
                if trailing_active and not trail_just_activated and effective_stop > base_stop
                else "stop"
            )
            pnl = (exit_price - ctx.entry_price) / ctx.entry_price * 100
            return RecomputeResult(reason, exit_price, bar_date, round(pnl, 4), round(mfe, 4), round(mae, 4))

        # Gap-through target
        if bar_open >= target:
            exit_price = round(bar_open * (1 - SLIPPAGE), 4)
            pnl = (exit_price - ctx.entry_price) / ctx.entry_price * 100
            return RecomputeResult("target", exit_price, bar_date, round(pnl, 4), round(mfe, 4), round(mae, 4))

        # Intraday target
        if bar_high >= target:
            exit_price = round(target * (1 - SLIPPAGE), 4)
            pnl = (exit_price - ctx.entry_price) / ctx.entry_price * 100
            return RecomputeResult("target", exit_price, bar_date, round(pnl, 4), round(mfe, 4), round(mae, 4))

        # Sniper time stop
        if (
            ctx.signal_model == "sniper"
            and days_traded >= SNIPER_TIME_STOP_DAYS
            and bar_close <= ctx.entry_price
        ):
            exit_price = round(bar_close * (1 - SLIPPAGE), 4)
            pnl = (exit_price - ctx.entry_price) / ctx.entry_price * 100
            return RecomputeResult("time_stop", exit_price, bar_date, round(pnl, 4), round(mfe, 4), round(mae, 4))

        # Expiry
        if days_traded >= ctx.holding_days:
            exit_price = round(bar_close * (1 - SLIPPAGE), 4)
            pnl = (exit_price - ctx.entry_price) / ctx.entry_price * 100
            return RecomputeResult("expiry", exit_price, bar_date, round(pnl, 4), round(mfe, 4), round(mae, 4))

    # Out of bars before any exit triggered. Inconclusive — caller skips.
    return None


def diff_outcome(ctx: TradeContext, new: RecomputeResult) -> dict[str, Any] | None:
    """Return a diff dict if any material field differs, else None."""
    def _close(a: float | None, b: float | None, eps: float) -> bool:
        if a is None or b is None:
            return a == b
        return abs(a - b) < eps

    changed = (
        ctx.old_exit_reason != new.exit_reason
        or not _close(ctx.old_exit_price, new.exit_price, 0.005)
        or ctx.old_exit_date != new.exit_date
        or not _close(ctx.old_pnl_pct, new.pnl_pct, 0.005)
    )
    if not changed:
        return None
    return {
        "outcome_id": ctx.outcome_id,
        "ticker": ctx.ticker,
        "entry_date": ctx.entry_date.isoformat(),
        "old": {
            "exit_reason": ctx.old_exit_reason,
            "exit_price": ctx.old_exit_price,
            "exit_date": ctx.old_exit_date.isoformat() if ctx.old_exit_date else None,
            "pnl_pct": ctx.old_pnl_pct,
        },
        "new": {
            "exit_reason": new.exit_reason,
            "exit_price": new.exit_price,
            "exit_date": new.exit_date.isoformat(),
            "pnl_pct": new.pnl_pct,
        },
    }


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


def build_candidates_query(
    *,
    start_date: date | None,
    end_date: date | None,
    ticker: str | None,
    outcome_id: int | None,
    exit_reason: str,
    limit: int | None,
) -> tuple[str, dict[str, Any]]:
    """Build the SELECT for candidate outcomes as (sql, params).

    Exposed for unit testing — the production code just passes this into
    `session.execute(text(sql), params)`.
    """
    where: list[str] = [
        "dr.execution_mode = 'quant_only'",
        "o.still_open = false",
        "o.skip_reason IS NULL",
    ]
    params: dict[str, Any] = {}

    if exit_reason != "all":
        where.append("o.exit_reason = :exit_reason")
        params["exit_reason"] = exit_reason
    if start_date is not None:
        where.append("s.run_date >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        where.append("s.run_date <= :end_date")
        params["end_date"] = end_date
    if ticker is not None:
        where.append("s.ticker = :ticker")
        params["ticker"] = ticker
    if outcome_id is not None:
        where.append("o.id = :outcome_id")
        params["outcome_id"] = outcome_id

    sql = f"""
        SELECT
            o.id AS outcome_id,
            s.ticker,
            s.signal_model,
            s.confidence,
            o.entry_date,
            o.entry_price,
            s.entry_price AS planned_entry,
            s.stop_loss   AS planned_stop,
            s.target_1    AS target,
            s.holding_period_days AS holding_days,
            o.exit_reason AS old_exit_reason,
            o.exit_price  AS old_exit_price,
            o.exit_date   AS old_exit_date,
            o.pnl_pct     AS old_pnl_pct
        FROM outcomes o
        JOIN signals s      ON o.signal_id = s.id
        JOIN daily_runs dr  ON s.run_date  = dr.run_date
        WHERE {' AND '.join(where)}
        ORDER BY s.run_date, s.ticker
    """.strip()
    if limit is not None:
        sql += f"\n        LIMIT {int(limit)}"
    return sql, params


def row_to_context(row: dict[str, Any]) -> TradeContext:
    return TradeContext(
        outcome_id=row["outcome_id"],
        ticker=row["ticker"],
        signal_model=row["signal_model"],
        confidence=row["confidence"],
        entry_date=row["entry_date"],
        entry_price=row["entry_price"],
        planned_entry=row["planned_entry"],
        planned_stop=row["planned_stop"],
        target=row["target"],
        holding_days=row["holding_days"],
        old_exit_reason=row["old_exit_reason"],
        old_exit_price=row["old_exit_price"],
        old_exit_date=row["old_exit_date"],
        old_pnl_pct=row["old_pnl_pct"],
    )


# ---------------------------------------------------------------------------
# Snapshot / apply
# ---------------------------------------------------------------------------


def write_snapshot(diffs: list[dict[str, Any]], out_dir: Path) -> Path:
    """Dump the pre-update state for every row that would change."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"phantom_backfill_snapshot_{timestamp}.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "count": len(diffs),
        "diffs": diffs,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


async def apply_updates(session, diffs: list[dict[str, Any]], *, snapshot_path: Path) -> int:
    """Apply UPDATEs to the outcomes table.

    Precondition: `snapshot_path` must already exist. This is a defense-
    in-depth check — the caller is expected to have written the snapshot
    first, and this function refuses to run if the artifact is missing.
    """
    if not snapshot_path.exists():
        raise RuntimeError(
            f"Refusing to apply updates: snapshot artifact missing at {snapshot_path}"
        )

    from sqlalchemy import text

    count = 0
    for d in diffs:
        await session.execute(
            text(
                """
                UPDATE outcomes
                SET exit_reason = :exit_reason,
                    exit_price  = :exit_price,
                    exit_date   = :exit_date,
                    pnl_pct     = :pnl_pct
                WHERE id = :outcome_id
                """
            ),
            {
                "exit_reason": d["new"]["exit_reason"],
                "exit_price": d["new"]["exit_price"],
                "exit_date": date.fromisoformat(d["new"]["exit_date"]),
                "pnl_pct": d["new"]["pnl_pct"],
                "outcome_id": d["outcome_id"],
            },
        )
        count += 1
    await session.commit()
    return count


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def summarize_diffs(diffs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary stats for the CLI report."""
    old_sum = sum((d["old"]["pnl_pct"] or 0.0) for d in diffs)
    new_sum = sum(d["new"]["pnl_pct"] for d in diffs)
    transitions: dict[str, int] = {}
    for d in diffs:
        key = f"{d['old']['exit_reason']} -> {d['new']['exit_reason']}"
        transitions[key] = transitions.get(key, 0) + 1
    return {
        "changed": len(diffs),
        "old_pnl_sum": round(old_sum, 4),
        "new_pnl_sum": round(new_sum, 4),
        "pnl_delta": round(new_sum - old_sum, 4),
        "transitions": transitions,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    if args.start_date is None and args.end_date is None and args.outcome_id is None:
        print("ERROR: must specify at least one of --start-date/--end-date/--outcome-id")
        return 2

    from sqlalchemy import text

    from src.data.aggregator import DataAggregator
    from src.db.session import get_session, init_db

    await init_db()
    aggregator = DataAggregator()

    sql, params = build_candidates_query(
        start_date=args.start_date,
        end_date=args.end_date,
        ticker=args.ticker,
        outcome_id=args.outcome_id,
        exit_reason=args.exit_reason,
        limit=args.limit,
    )

    async with get_session() as session:
        result = await session.execute(text(sql), params)
        rows = [dict(r) for r in result.mappings()]

    print(f"Scanned {len(rows)} candidate outcome(s)")
    if not rows:
        return 0

    diffs: list[dict[str, Any]] = []
    skipped_inconclusive = 0

    for row in rows:
        ctx = row_to_context(row)
        lookback_start = ctx.entry_date - timedelta(days=3)
        lookback_end = ctx.entry_date + timedelta(days=ctx.holding_days + BAR_LOOKBACK_DAYS)
        df = await aggregator.get_ohlcv(ctx.ticker, lookback_start, lookback_end)
        if df is None or df.empty:
            skipped_inconclusive += 1
            continue

        bars_list = [
            {
                "date": r["date"].date() if hasattr(r["date"], "date") else r["date"],
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
            }
            for _, r in df.iterrows()
            if (r["date"].date() if hasattr(r["date"], "date") else r["date"]) >= ctx.entry_date
        ]
        new = recompute_exit(ctx, bars_list)
        if new is None:
            skipped_inconclusive += 1
            continue

        d = diff_outcome(ctx, new)
        if d is not None:
            diffs.append(d)

    summary = summarize_diffs(diffs)
    print(f"Skipped (inconclusive / no data): {skipped_inconclusive}")
    print(f"Changed: {summary['changed']}")
    print(f"P&L delta: {summary['pnl_delta']:+.2f}% (old sum {summary['old_pnl_sum']:+.2f}% -> new {summary['new_pnl_sum']:+.2f}%)")
    if summary["transitions"]:
        print("Exit-reason transitions:")
        for k, v in sorted(summary["transitions"].items()):
            print(f"  {k}: {v}")

    if not diffs:
        print("No changes.")
        return 0

    if not args.apply:
        print("\nDry run. Pass --apply to write changes. Sample diffs:")
        for d in diffs[:5]:
            print(
                f"  {d['ticker']:<6} {d['entry_date']}  "
                f"{d['old']['exit_reason']}@{d['old']['pnl_pct']:+.2f}% "
                f"-> {d['new']['exit_reason']}@{d['new']['pnl_pct']:+.2f}%"
            )
        return 0

    snapshot_path = write_snapshot(diffs, Path(args.snapshot_dir))
    print(f"\nSnapshot written: {snapshot_path}")
    async with get_session() as session:
        n = await apply_updates(session, diffs, snapshot_path=snapshot_path)
    print(f"Applied {n} updates.")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-date", type=lambda s: date.fromisoformat(s))
    p.add_argument("--end-date", type=lambda s: date.fromisoformat(s))
    p.add_argument("--ticker")
    p.add_argument("--outcome-id", type=int)
    p.add_argument(
        "--exit-reason",
        default="trail_stop",
        help="Targeted exit_reason, or 'all' to scan every closed outcome in scope",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--apply", action="store_true", help="Write changes to the DB")
    p.add_argument(
        "--snapshot-dir",
        default="backups/phantom_backfill",
        help="Directory for pre-update snapshot artifacts",
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(_parse_args())))
