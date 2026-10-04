"""RECLAIM native shadow lanes — the daily loop (RECLAIM_MAS_SPEC_v1.1 §8.1, §10).

Governance, by construction:

- Its own function, called once from the morning pipeline behind
  ``settings.reclaim_shadow_enabled`` (default False). With the flag off it is
  never called: no fetch, no DB write, one skip line.
- It shares nothing mutable with the official path. It never touches
  ``all_signals``, ranking, confluence, cooldown, the correlation filter, top-N,
  position caps, validation, or Cat/Telegram pick rendering. It reads the
  run's filtered universe rows and regime label; it writes only its own tables
  plus clone ``Signal`` rows under ``reclaim_shadow_h*`` sources.
- Its own data fetch (``reclaim_history_days`` of history) with its own
  provenance; the shared 300-day fetch and its cache keys are untouched.
- Any exception is the caller's to catch and log as ``RECLAIM_SHADOW_FAILED``
  (WARN). Official outputs are already built when it runs.

Nothing produced here is evidence until the §7 gate passes on clean forward
triggers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.db.models import (
    ReclaimClone,
    ReclaimEarningsSnapshot,
    ReclaimEpisode,
    ReclaimEvent,
    ReclaimLaneRegistry,
    ReclaimPoolMember,
    ReclaimPoolRun,
    ReclaimRiskSnapshot,
    ReclaimTrigger,
    Signal,
)
from src.research import reclaim_replay as rr
from src.research.reclaim_replay import select_reclaim_universe
from src.signals import reclaim as rc
from src.streams import reclaim_source
from src.utils.trading_calendar import previous_trading_day, trading_sessions

logger = logging.getLogger(__name__)

IDENTITY_METHOD = "TICKER_ONLY"
MIN_SPY_SESSIONS = 400
# A run whose bars do not reach C for most symbols is stale data, not a pool.
MIN_FRACTION_WITH_C_BAR = 0.50


# ── Insert-ignore (append-only, earliest row wins) ───────────────────────────

def _insert_stmt(session, model):
    dialect = session.bind.dialect.name if session.bind is not None else "postgresql"
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:
        from sqlalchemy.dialects.postgresql import insert
    return insert(model)


def _unique_cols(model) -> list[str]:
    from sqlalchemy import UniqueConstraint
    for con in model.__table__.constraints:
        if isinstance(con, UniqueConstraint):
            return [c.name for c in con.columns]
    raise ValueError(f"{model.__tablename__} has no unique key")


async def insert_ignore(session, model, rows: list[dict] | dict) -> int:
    """INSERT ... ON CONFLICT (unique key) DO NOTHING. Returns rows inserted."""
    rows = [rows] if isinstance(rows, dict) else rows
    if not rows:
        return 0
    inserted = 0
    for i in range(0, len(rows), 500):
        stmt = _insert_stmt(session, model).values(rows[i:i + 500])
        stmt = stmt.on_conflict_do_nothing(index_elements=_unique_cols(model))
        result = await session.execute(stmt)
        inserted += max(result.rowcount or 0, 0)
    return inserted


async def insert_ignore_returning_id(session, model, row: dict) -> int | None:
    stmt = (_insert_stmt(session, model).values(row)
            .on_conflict_do_nothing(index_elements=_unique_cols(model))
            .returning(model.id))
    result = await session.execute(stmt)
    got = result.first()
    return int(got[0]) if got else None


# ── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class ReclaimData:
    sessions: list[date]
    bars: dict[str, pd.DataFrame]                 # session-aligned
    meta: dict[str, dict]                         # symbol -> {mcap, sector}
    benchmarks: dict[str, pd.DataFrame]           # session-aligned SPY + sector SPDRs
    complete: bool
    universe_count: int
    cap_hits: int
    provenance: dict = field(default_factory=dict)
    failure: str | None = None


def signal_date(today: date) -> date:
    """C = the last completed XNYS session before the morning run's date."""
    return previous_trading_day(today)


async def fetch_reclaim_data(universe_rows: list[dict], C: date, settings) -> ReclaimData:
    """Own fetch, own provenance, fail-closed completeness (§8.7)."""
    from src.data.aggregator import DataAggregator

    sessions = trading_sessions(C - timedelta(days=settings.reclaim_history_days), C)
    symbols, meta, cap_hits = select_reclaim_universe(universe_rows, settings.max_ohlcv_tickers)
    agg = DataAggregator()
    agg.reset_data_provenance()
    try:
        raw = await agg.get_bulk_ohlcv(symbols + list(rc.BENCHMARK_TICKERS), sessions[0], C)
        provenance = agg.get_data_provenance()
    finally:
        agg.close()
    return assemble_reclaim_data(raw, symbols, meta, sessions, cap_hits, provenance)


def assemble_reclaim_data(
    raw: dict[str, pd.DataFrame],
    symbols: list[str],
    meta: dict[str, dict],
    sessions: list[date],
    cap_hits: int,
    provenance: dict | None = None,
) -> ReclaimData:
    C = sessions[-1]
    bars = {s: rc.align_to_sessions(raw.get(s), sessions) for s in symbols}
    benchmarks = {b: rc.align_to_sessions(raw.get(b), sessions) for b in rc.BENCHMARK_TICKERS}
    spy_n = int(benchmarks["SPY"]["close"].notna().sum())
    with_c = sum(1 for b in bars.values() if np.isfinite(b["close"].iloc[-1]))
    failure = None
    if spy_n < MIN_SPY_SESSIONS:
        failure = f"SPY history {spy_n} < {MIN_SPY_SESSIONS} sessions"
    elif not np.isfinite(benchmarks["SPY"]["close"].iloc[-1]):
        failure = f"SPY has no bar on {C}"
    elif symbols and with_c / len(symbols) < MIN_FRACTION_WITH_C_BAR:
        failure = f"only {with_c}/{len(symbols)} symbols have a bar on {C}"
    per_symbol = {s: int(b["close"].notna().sum()) for s, b in bars.items()}
    prov = dict(provenance or {})
    prov.update({
        "sessions": len(sessions),
        "from": sessions[0].isoformat(),
        "to": C.isoformat(),
        "spy_sessions": spy_n,
        "symbols_requested": len(symbols),
        "symbols_with_c_bar": with_c,
        "symbols_empty": sorted(s for s, n in per_symbol.items() if n == 0),
        "market_cap_basis": "current_vendor_value",
    })
    return ReclaimData(
        sessions=sessions, bars=bars, meta=meta, benchmarks=benchmarks,
        complete=failure is None, universe_count=len(symbols), cap_hits=cap_hits,
        provenance=prov, failure=failure,
    )


async def capture_earnings(fmp_fetch, symbol: str, now_utc: datetime) -> dict:
    """FMP earnings snapshot for one triggered symbol. Failure is recorded, not raised."""
    try:
        payload = await fmp_fetch(symbol)
        ok = isinstance(payload, list)
    except Exception as exc:
        logger.warning("Reclaim earnings capture failed for %s: %s", symbol, exc)
        payload, ok = {"error": f"{type(exc).__name__}: {exc}"}, False
    dates = rc.parse_earnings_dates(payload if isinstance(payload, list) else None)
    coverage = {"min": dates[0].isoformat(), "max": dates[-1].isoformat(), "n": len(dates)} if dates else {"n": 0}
    return {"payload": _json_safe(payload), "ok": ok, "captured_at_utc": now_utc, "coverage": coverage}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return obj


def _f(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


# ── The daily run ────────────────────────────────────────────────────────────

@dataclass
class ReclaimRunSummary:
    signal_date: date
    pool_status: dict[str, str] = field(default_factory=dict)
    members: dict[str, int] = field(default_factory=dict)
    episodes: dict[str, int] = field(default_factory=dict)
    triggers: list[str] = field(default_factory=list)
    clones_created: int = 0
    forward_start: dict[str, str | None] = field(default_factory=dict)


async def run_reclaim_shadow(
    today: date,
    settings,
    *,
    universe_rows: list[dict],
    mas_regime: str | None,
    session_factory=None,
    data: ReclaimData | None = None,
    earnings_fetch=None,
    now_utc: datetime | None = None,
    code_sha: str | None = None,
) -> ReclaimRunSummary | None:
    """One production run of both native lanes for signal date C (§10)."""
    if not settings.reclaim_shadow_enabled:
        logger.info("Reclaim shadow: disabled (reclaim_shadow_enabled=False) — skipped")
        return None
    if session_factory is None:
        from src.db.session import get_session as session_factory
    now_utc = now_utc or datetime.now(timezone.utc)
    C = signal_date(today)
    if data is None:
        data = await fetch_reclaim_data(universe_rows, C, settings)
    if earnings_fetch is None:
        from src.data.fmp_client import FMPClient
        earnings_fetch = FMPClient().get_earnings_surprise
    sessions = data.sessions
    if sessions[-1] != C:
        raise ValueError(f"reclaim data ends {sessions[-1]}, expected signal date {C}")
    c_idx = len(sessions) - 1
    summary = ReclaimRunSummary(signal_date=C)

    # 1. Production snapshots, one per lane. An incomplete fetch leaves C uncovered.
    pools: dict[str, rr.LanePool] = {}
    async with session_factory() as session:
        for lane in rc.LANES:
            if not data.complete:
                await insert_ignore(session, ReclaimPoolRun, {
                    "lane": lane, "snapshot_date": C, "status": "FAILED",
                    "universe_count": data.universe_count, "members_count": 0,
                    "ohlcv_cap_hits": data.cap_hits, "identity_method": IDENTITY_METHOD,
                    "provenance": _json_safe({**data.provenance, "failure": data.failure}),
                })
                summary.pool_status[lane] = "FAILED"
                continue
            pool = rr.compute_lane_pool(lane, data.bars, data.meta)
            pools[lane] = pool
            members = [s for s in pool.member.columns if bool(pool.member[s].iloc[c_idx])]
            basis = "close_x_volume" if lane == rc.RANGE else "vwap_x_volume_else_close"
            await insert_ignore(session, ReclaimPoolRun, {
                "lane": lane, "snapshot_date": C, "status": "OK",
                "universe_count": data.universe_count, "members_count": len(members),
                "ohlcv_cap_hits": data.cap_hits, "dollar_volume_basis": basis,
                "identity_method": IDENTITY_METHOD, "provenance": _json_safe(data.provenance),
            })
            await insert_ignore(session, ReclaimPoolMember, [
                _member_row(lane, C, sym, pool, c_idx, data.meta) for sym in members
            ])
            summary.pool_status[lane] = "OK"
            summary.members[lane] = len(members)

            # Forward start: the first covered production run; immutable once set.
            await insert_ignore(session, ReclaimLaneRegistry, {
                "lane": lane, "forward_start_date": C,
                "spec_sha": "RECLAIM_MAS_SPEC_v1.1", "code_sha": code_sha,
            })
    if not data.complete:
        logger.warning("Reclaim shadow: data incomplete (%s) — %s uncovered, no triggers",
                       data.failure, C)
        return summary

    # 2. Episodes per lane: production inclusions from the forward start on,
    #    reconstructed (RECONSTRUCTED_DESCRIPTIVE) before it.
    async with session_factory() as session:
        fwd = await _forward_starts(session)
        covered = await _covered_dates(session)
        prod_members = await _production_members(session)
    summary.forward_start = {lane: (fwd.get(lane).isoformat() if fwd.get(lane) else None)
                             for lane in rc.LANES}
    idx_of = {d: i for i, d in enumerate(sessions)}
    lane_eps: dict[str, rr.LaneEpisodes] = {}
    for lane in rc.LANES:
        pool = pools[lane]
        fwd_date = fwd.get(lane)
        fwd_idx = idx_of.get(fwd_date) if fwd_date else None
        if fwd_date is not None and fwd_idx is None:
            fwd_idx = next((i for i, d in enumerate(sessions) if d >= fwd_date), len(sessions))
        inclusions = rr.reconstructed_inclusions(pool, before_idx=fwd_idx)
        floor_at: dict[str, dict[int, float]] = {}
        for sym in pool.member.columns:
            floor_at[sym] = {int(i): float(pool.box_low[sym].iloc[int(i)])
                             for i in inclusions.get(sym, set())}
        for (sym, d), row in prod_members.get(lane, {}).items():
            i = idx_of.get(d)
            if i is None or (fwd_idx is not None and i < fwd_idx):
                continue
            inclusions.setdefault(sym, set()).add(i)
            if row.get("box_low") is not None:
                floor_at.setdefault(sym, {})[i] = row["box_low"]
        lane_covered = {d for d in covered.get(lane, set()) if fwd_date and d >= fwd_date}

        def prehistory_fn(start: int, _cov=lane_covered, _fwd=fwd_date, _lane=lane):
            return rc.prehistory_native(start, sessions, _cov, _fwd, rc.LANE_WARMUP[_lane])

        lane_eps[lane] = rr.replay_lane(lane, data.bars, sessions, c_idx, inclusions,
                                        floor_at, prehistory_fn)
        summary.episodes[lane] = len(lane_eps[lane].results)

    # 3. Persist episodes/events that are live in the forward window; detect
    #    replay divergence against what was persisted before.
    async with session_factory() as session:
        for lane in rc.LANES:
            fwd_date = fwd.get(lane) or C
            keep = [r for r in lane_eps[lane].results
                    if r.end is None or sessions[r.end] >= fwd_date]
            await _persist_episodes(session, keep, sessions, C)

    # 4. Triggers from the forward start on that are not yet recorded.
    async with session_factory() as session:
        persisted_triggers = {
            eid: d for eid, d in (await session.execute(
                select(ReclaimTrigger.episode_id, ReclaimTrigger.trigger_date))).all()
        }
    spy = data.benchmarks["SPY"]
    for lane in rc.LANES:
        fwd_date = fwd.get(lane)
        if fwd_date is None:
            continue
        other = lane_eps[rc.DRIFT if lane == rc.RANGE else rc.RANGE]
        for res in lane_eps[lane].results:
            trig = res.trigger
            if trig is None or trig.trigger_date < fwd_date:
                continue
            prior = persisted_triggers.get(trig.episode_id)
            if prior is not None and prior != trig.trigger_date:
                # One trigger per episode and the persisted one stands; the
                # recomputed disagreement is already logged as REPLAY_DIVERGENCE.
                continue
            # A known trigger is re-entered on purpose: every write below is
            # insert-ignore, so a run that died between the trigger row and its
            # clones completes them now instead of never.
            await _record_trigger(
                session_factory, today, trig, res.episode, lane_eps[lane], other,
                pools[lane], data, spy, mas_regime, earnings_fetch, now_utc, c_idx, summary,
            )
    logger.info(
        "Reclaim shadow %s: pools=%s members=%s episodes=%s triggers=%d clones=%d fwd=%s",
        C, summary.pool_status, summary.members, summary.episodes,
        len(summary.triggers), summary.clones_created, summary.forward_start,
    )
    return summary


def _member_row(lane: str, C: date, sym: str, pool: rr.LanePool, c_idx: int, meta: dict) -> dict:
    s = pool.series[sym].iloc[c_idx]
    m = meta.get(sym) or {}
    gate = {k: _f(v) for k, v in s.items() if k not in ("member",)}
    return {
        "lane": lane, "snapshot_date": C, "symbol": sym,
        "score": _f(s.get("score")),
        "score_pct": _f(pool.score_pct[sym].iloc[c_idx]) if lane == rc.RANGE else None,
        "box_low": _f(s.get("box_low")), "box_high": _f(s.get("box_high")),
        "atr14": _f(s.get("atr14")), "mcap": _f(m.get("mcap")), "sector": m.get("sector"),
        "gate_inputs": gate,
    }


async def _forward_starts(session) -> dict[str, date]:
    rows = (await session.execute(
        select(ReclaimLaneRegistry.lane, ReclaimLaneRegistry.forward_start_date))).all()
    return {lane: d for lane, d in rows}


async def _covered_dates(session) -> dict[str, set[date]]:
    rows = (await session.execute(
        select(ReclaimPoolRun.lane, ReclaimPoolRun.snapshot_date)
        .where(ReclaimPoolRun.status == "OK"))).all()
    out: dict[str, set[date]] = {}
    for lane, d in rows:
        out.setdefault(lane, set()).add(d)
    return out


async def _production_members(session) -> dict[str, dict[tuple[str, date], dict]]:
    rows = (await session.execute(
        select(ReclaimPoolMember.lane, ReclaimPoolMember.symbol, ReclaimPoolMember.snapshot_date,
               ReclaimPoolMember.box_low, ReclaimPoolMember.score_pct)
        .join(ReclaimPoolRun, (ReclaimPoolRun.lane == ReclaimPoolMember.lane)
              & (ReclaimPoolRun.snapshot_date == ReclaimPoolMember.snapshot_date))
        .where(ReclaimPoolRun.status == "OK"))).all()
    out: dict[str, dict[tuple[str, date], dict]] = {}
    for lane, sym, d, box_low, pct in rows:
        out.setdefault(lane, {})[(sym, d)] = {"box_low": box_low, "score_pct": pct}
    return out


async def _persist_episodes(session, results: list[rc.EpisodeResult], sessions: list[date], C: date) -> None:
    if not results:
        return
    ids = [r.episode.episode_id for r in results]
    persisted: dict[str, set[tuple[date, str]]] = {}
    for i in range(0, len(ids), 500):
        rows = (await session.execute(
            select(ReclaimEvent.episode_id, ReclaimEvent.event_date, ReclaimEvent.event)
            .where(ReclaimEvent.episode_id.in_(ids[i:i + 500])))).all()
        for eid, d, ev in rows:
            persisted.setdefault(eid, set()).add((d, ev))

    episode_rows, event_rows = [], []
    for r in results:
        ep = r.episode
        episode_rows.append({
            "episode_id": ep.episode_id, "lane": ep.lane, "symbol": ep.symbol,
            "start_date": ep.start_date, "floor": _f(ep.floor), "floor_type": ep.floor_type,
            "prehistory": ep.prehistory, "prehistory_reason": ep.prehistory_reason,
        })
        recomputed = [(sessions[i], ev, payload) for i, ev, payload in r.events]
        old = persisted.get(ep.episode_id, set())
        last_old = max((d for d, _ in old), default=None)
        if old:
            new_pairs = {(d, ev) for d, ev, _ in recomputed}
            missing = sorted(old - new_pairs - {(d, ev) for d, ev in old if ev == "REPLAY_DIVERGENCE"})
            added_retro = sorted((d, ev) for d, ev in new_pairs - old
                                 if last_old is not None and d <= last_old)
            if missing or added_retro:
                # Persisted rows stand; the disagreement is recorded (R30).
                event_rows.append({
                    "episode_id": ep.episode_id, "event_date": C, "event": "REPLAY_DIVERGENCE",
                    "payload": _json_safe({"persisted_not_reproduced": missing,
                                           "reproduced_not_persisted": added_retro}),
                })
        for d, ev, payload in recomputed:
            if last_old is not None and d <= last_old:
                continue
            event_rows.append({"episode_id": ep.episode_id, "event_date": d, "event": ev,
                               "payload": _json_safe(payload)})
    await insert_ignore(session, ReclaimEpisode, episode_rows)
    await insert_ignore(session, ReclaimEvent, event_rows)


async def _record_trigger(
    session_factory, today: date, trig: rc.Trigger, ep: rc.Episode,
    lane_eps: rr.LaneEpisodes, other: rr.LaneEpisodes, pool: rr.LanePool,
    data: ReclaimData, spy: pd.DataFrame, mas_regime: str | None,
    earnings_fetch, now_utc: datetime, c_idx: int, summary: ReclaimRunSummary,
) -> None:
    entry_session = trig.entry_session
    if entry_session is None:
        from src.utils.trading_calendar import next_trading_day
        entry_session = next_trading_day(trig.trigger_date)
    m = data.meta.get(trig.symbol) or {}
    overlap = rr.overlap_desc(trig, other, c_idx)
    pct = rr.latest_score_pct(pool, trig.symbol, trig.k)
    risk = rr.risk_rows(trig, lane_eps, pool, data.meta)
    async with session_factory() as session:
        have_snapshot = (await session.execute(
            select(ReclaimEarningsSnapshot.id).where(
                ReclaimEarningsSnapshot.symbol == trig.symbol,
                ReclaimEarningsSnapshot.trigger_date == trig.trigger_date,
            ))).first() is not None
    # The earliest snapshot wins (§5A); a rerun never re-captures over it.
    snapshot = None if have_snapshot else await capture_earnings(earnings_fetch, trig.symbol, now_utc)

    async with session_factory() as session:
        inserted = await insert_ignore(session, ReclaimTrigger, {
            "episode_id": trig.episode_id, "trigger_date": trig.trigger_date,
            "lane": trig.lane, "symbol": trig.symbol, "entry_session": entry_session,
            "reclaim_date": trig.reclaim_date, "retest_date": trig.retest_date,
            "setup_low": trig.setup_low, "high_retest": trig.high_retest,
            "close_k": trig.close_k, "sma50_k": trig.sma50_k, "slope5": _f(trig.slope5),
            "floor": _f(ep.floor), "prehistory": ep.prehistory,
            "prehistory_reason": ep.prehistory_reason, "overlap_desc": overlap,
            "score_pct": _f(pct), "sector": m.get("sector"), "mcap": _f(m.get("mcap")),
            "mas_regime": mas_regime, "spy_regime": rc.spy_regime_label(spy, trig.k + 1),
            "detected_on": today, "forward_recorded": True,
        })
        await insert_ignore(session, ReclaimRiskSnapshot, [{
            "trigger_episode_id": trig.episode_id, "trigger_date": trig.trigger_date,
            "control_episode_id": r.control_episode_id, "control_symbol": r.control_symbol,
            "sector": r.sector, "mcap": _f(r.mcap), "score_pct": _f(r.score_pct),
            "sma_dist": _f(r.sma_dist), "age": r.age, "prehistory": r.prehistory,
            "retest_pending": r.retest_pending, "confirm_pending": r.confirm_pending,
            "triggered_on_k": r.triggered_on_k,
        } for r in risk])
        if snapshot is not None:
            await insert_ignore(session, ReclaimEarningsSnapshot, {
                "symbol": trig.symbol, "trigger_date": trig.trigger_date, **snapshot,
            })
    if inserted:
        summary.triggers.append(trig.episode_id)

    for h in rc.HORIZONS:
        sig = rc.ReclaimSignal.from_trigger(trig, ep, h, overlap_desc=overlap, score_pct_value=pct)
        async with session_factory() as session:
            clone_id = await insert_ignore_returning_id(session, ReclaimClone, {
                "episode_id": trig.episode_id, "trigger_date": trig.trigger_date, "horizon": h,
            })
            if clone_id is None:
                continue        # durable key exists: no second Signal, ever
            row = Signal(
                run_date=today, ticker=trig.symbol, direction="LONG",
                signal_model="reclaim", signal_source=reclaim_source(h),
                entry_price=trig.close_k, stop_loss=trig.setup_low,
                target_1=rc.NO_TARGET_SENTINEL, target_2=rc.NO_TARGET_SENTINEL,
                holding_period_days=h, confidence=sig.score,
                risk_gate_decision="SHADOW", regime=mas_regime or "unknown",
                features=sig.persisted_features(), max_entry_price=_f(sig.max_entry_price),
            )
            session.add(row)
            await session.flush()
            clone = await session.get(ReclaimClone, clone_id)
            clone.signal_id = row.id
            summary.clones_created += 1

