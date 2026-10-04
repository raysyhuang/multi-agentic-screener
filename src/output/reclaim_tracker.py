"""The only code that evaluates RECLAIM clones (RECLAIM_MAS_SPEC_v1.1 §5A, §8.4, §8.5).

The generic ``check_open_positions()`` never sees these rows: they carry a
reclaim ``skip_reason`` and the generic loop also filters their sources.

Per trigger, once bar e = k+1 exists:
1. one ``reclaim_open_decisions`` row from the RAW open (no slippage), with every
   applicable reason and the fail-closed earnings status;
2. PASS -> three clone Outcomes filled at ``open_e * 1.001``, walked with the
   structural stop, no target, no trail, no partial, no time stop;
   non-PASS -> each clone gets a closed non-trade Outcome and no return.

The economic exit is booked on the Outcome. Statistical completeness (uniform
censoring) belongs to the Reclaim reporter, never to the Outcome table.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.db.models import (
    Outcome,
    ReclaimClone,
    ReclaimEarningsSnapshot,
    ReclaimOpenDecision,
    ReclaimRiskSnapshot,
    ReclaimTrigger,
    Signal,
)
from src.signals import reclaim as rc
from src.streams import RECLAIM_NONTRADE_SKIP_REASON, RECLAIM_SKIP_REASON
from src.utils.trading_calendar import is_trading_day, previous_trading_day, trading_sessions

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
# Outcome.exit_reason is String(20); the full status lives in reclaim_open_decisions.
NONTRADE_EXIT_REASON = {
    rc.UNFILLABLE_GAP_THROUGH: "unfillable_gap",
    rc.SKIP_EXTENDED: "skip_extended",
    rc.SKIP_RISK: "skip_risk",
}


def last_complete_session(now_utc: datetime) -> date:
    """The latest XNYS session whose daily bar is final (after 16:15 ET)."""
    now_et = now_utc.astimezone(_ET)
    d = now_et.date()
    if is_trading_day(d) and now_et.time() >= time(16, 15):
        return d
    return previous_trading_day(d)


class ReclaimTargetError(RuntimeError):
    """A reclaim row reached the tracker without its no-target marker."""


async def _default_bars(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    from src.data.aggregator import DataAggregator
    agg = DataAggregator()
    try:
        return await agg.get_bulk_ohlcv(symbols, start, end)
    finally:
        agg.close()


async def run_reclaim_tracker(
    *,
    session_factory=None,
    bars_fetch=None,
    now_utc: datetime | None = None,
) -> dict:
    """Decide new opens, then walk every open clone. Idempotent."""
    if session_factory is None:
        from src.db.session import get_session as session_factory
    bars_fetch = bars_fetch or _default_bars
    now_utc = now_utc or datetime.now(timezone.utc)
    last = last_complete_session(now_utc)
    stats = {"decisions": 0, "opened": 0, "nontrade": 0, "closed": 0, "updated": 0}

    async with session_factory() as session:
        triggers = (await session.execute(select(ReclaimTrigger))).scalars().all()
        decided = {(eid, d) for eid, d in (await session.execute(
            select(ReclaimOpenDecision.episode_id, ReclaimOpenDecision.trigger_date))).all()}
        clones = (await session.execute(select(ReclaimClone))).scalars().all()
        open_outcome_ids = {oid for (oid,) in (await session.execute(
            select(Outcome.id).where(Outcome.still_open == True))).all()}  # noqa: E712

    pending = [t for t in triggers if (t.episode_id, t.trigger_date) not in decided
               and t.entry_session <= last]
    trig_by_key = {(t.episode_id, t.trigger_date): t for t in triggers}
    active_clones = [c for c in clones if c.outcome_id is None or c.outcome_id in open_outcome_ids]
    active_keys = {(c.episode_id, c.trigger_date) for c in active_clones}
    control_symbols: set[str] = set()
    if pending:
        async with session_factory() as session:
            pending_ids = [t.episode_id for t in pending]
            control_symbols = {sym for (sym,) in (await session.execute(
                select(ReclaimRiskSnapshot.control_symbol).where(
                    ReclaimRiskSnapshot.trigger_episode_id.in_(pending_ids)))).all()}
    symbols = sorted({t.symbol for t in pending} | control_symbols
                     | {trig_by_key[k].symbol for k in active_keys if k in trig_by_key})
    if not symbols and not pending:
        return stats

    earliest = min([t.trigger_date for t in pending]
                   + [trig_by_key[k].trigger_date for k in active_keys if k in trig_by_key]
                   or [last])
    start = earliest - timedelta(days=10)
    sessions = trading_sessions(start, last)
    raw = await bars_fetch(symbols, sessions[0], last) if symbols else {}
    bars = {s: rc.align_to_sessions(raw.get(s), sessions) for s in symbols}
    idx_of = {d: i for i, d in enumerate(sessions)}
    last_idx = len(sessions) - 1

    # 1. Actual-open decisions.
    for t in pending:
        e = idx_of.get(t.entry_session)
        b = bars.get(t.symbol)
        open_e = None
        if e is not None and b is not None and np.isfinite(b["open"].iloc[e]):
            open_e = float(b["open"].iloc[e])
        dec = rc.open_decision(open_e, t.setup_low, t.sma50_k)
        if dec.mech_status == rc.CENSORED_NO_ENTRY_BAR:
            continue                    # re-evaluated on the next run
        async with session_factory() as session:
            snap = (await session.execute(select(ReclaimEarningsSnapshot).where(
                ReclaimEarningsSnapshot.symbol == t.symbol,
                ReclaimEarningsSnapshot.trigger_date == t.trigger_date))).scalar_one_or_none()
            if snap is None:
                earnings = rc.INACTIVE_FAILED
            else:
                dates = rc.parse_earnings_dates(snap.payload if isinstance(snap.payload, list) else None)
                earnings = rc.earnings_status(snap.ok, snap.captured_at_utc, dates,
                                              t.entry_session, is_trading_day)
            from src.reclaim_shadow import insert_ignore
            stats["decisions"] += await insert_ignore(session, ReclaimOpenDecision, {
                "episode_id": t.episode_id, "trigger_date": t.trigger_date,
                "open_e": dec.open_e, "stop": dec.stop, "sma50_k": dec.sma50_k,
                "ext_ratio": dec.ext_ratio, "risk_pct": dec.risk_pct,
                "reasons": dec.reasons, "mech_status": dec.mech_status,
                "earnings_status": earnings, "decided_at_utc": now_utc,
            })
        await _record_control_entry_bars(session_factory, t, bars, idx_of)

    # 2. Clones.
    async with session_factory() as session:
        decisions = {(d.episode_id, d.trigger_date): d for d in (await session.execute(
            select(ReclaimOpenDecision))).scalars().all()}
    for clone in active_clones:
        key = (clone.episode_id, clone.trigger_date)
        dec = decisions.get(key)
        trig = trig_by_key.get(key)
        if dec is None or trig is None or clone.signal_id is None:
            continue
        async with session_factory() as session:
            sig = await session.get(Signal, clone.signal_id)
            if sig is None:
                continue
            feats = sig.features or {}
            if feats.get("no_target") is not True:
                raise ReclaimTargetError(
                    f"reclaim signal {sig.id} has no no_target marker; refusing to walk it")
            row = await session.get(ReclaimClone, clone.id)
            outcome = await session.get(Outcome, row.outcome_id) if row.outcome_id else None

            if dec.mech_status != rc.PASS:
                if outcome is None:
                    outcome = Outcome(
                        signal_id=sig.id, ticker=sig.ticker, entry_date=trig.entry_session,
                        entry_price=dec.open_e if dec.open_e is not None else trig.close_k,
                        exit_date=trig.entry_session, exit_reason=NONTRADE_EXIT_REASON.get(
                            dec.mech_status, dec.mech_status.lower()[:20]),
                        still_open=False, skip_reason=RECLAIM_NONTRADE_SKIP_REASON,
                        pnl_pct=None,
                    )
                    session.add(outcome)
                    await session.flush()
                    row.outcome_id = outcome.id
                    stats["nontrade"] += 1
                continue

            entry = dec.open_e * (1.0 + rc.COST)
            if outcome is None:
                outcome = Outcome(
                    signal_id=sig.id, ticker=sig.ticker, entry_date=trig.entry_session,
                    entry_price=round(entry, 4), still_open=True,
                    skip_reason=RECLAIM_SKIP_REASON, entry_slippage_pct=rc.COST * 100,
                )
                session.add(outcome)
                await session.flush()
                row.outcome_id = outcome.id
                stats["opened"] += 1
            if not outcome.still_open:
                continue
            e = idx_of.get(trig.entry_session)
            b = bars.get(trig.symbol)
            if e is None or b is None:
                continue
            res = rc.clone_exit(b, sessions, e, dec.open_e, trig.setup_low, clone.horizon, last_idx)
            if res.exited:
                outcome.exit_date = res.exit_date
                outcome.exit_price = round(res.exit_price, 4)
                outcome.exit_reason = res.exit_reason
                outcome.pnl_pct = _clean(round(res.net_return * 100, 4))
                outcome.max_favorable = _clean(round(res.mfe_pct, 4))
                outcome.max_adverse = _clean(round(res.mae_pct, 4))
                outcome.exit_slippage_pct = rc.COST * 100
                outcome.still_open = False
                stats["closed"] += 1
            else:
                close_now = b["close"].iloc[last_idx]
                if np.isfinite(close_now):
                    outcome.pnl_pct = _clean(round((float(close_now) / entry - 1.0) * 100, 4))
                outcome.max_favorable = _clean(round(res.mfe_pct, 4))
                outcome.max_adverse = _clean(round(res.mae_pct, 4))
                stats["updated"] += 1
    logger.info("Reclaim tracker (%s): %s", last, stats)
    return stats


async def _record_control_entry_bars(session_factory, trig, bars, idx_of) -> None:
    """Persist, once, whether each control has an entry bar at e (§7).

    Made when bar e arrives (the same pass as the trigger's open decision) and
    never revisited: only rows still NULL are written.
    """
    e = idx_of.get(trig.entry_session)
    if e is None:
        return
    async with session_factory() as session:
        rows = (await session.execute(select(ReclaimRiskSnapshot).where(
            ReclaimRiskSnapshot.trigger_episode_id == trig.episode_id,
            ReclaimRiskSnapshot.trigger_date == trig.trigger_date,
            ReclaimRiskSnapshot.has_entry_bar.is_(None)))).scalars().all()
        for r in rows:
            b = bars.get(r.control_symbol)
            if b is None:
                continue
            o = b["open"].iloc[e]
            r.has_entry_bar = bool(np.isfinite(o))
            r.control_open_e = float(o) if np.isfinite(o) else None


def _clean(v: float) -> float | None:
    return None if (v is None or math.isnan(v) or math.isinf(v)) else v
