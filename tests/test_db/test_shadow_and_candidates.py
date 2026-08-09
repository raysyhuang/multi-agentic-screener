"""Behavioural tests for shadow-booking and the candidate audit — real DB rows.

These replace source-inspection tests that grepped function bodies for
`SHADOW_SKIP_REASON` and `or_(`. Those passed whether or not the SQL was
correct, which is precisely the failure mode they were supposed to guard.

Every assertion here is against actual query output over persisted rows:
official, manual-sleeve, PEAD, gap-rejected and shadow-booked.
"""
from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Reuse the JSONB->JSON sqlite shim registered by the persistence test module.
from tests.test_db import test_exit_config_persistence  # noqa: F401

from src.db.models import Base, Candidate, DailyRun, Outcome, Signal
from src.output.performance import SHADOW_SKIP_REASON

RUN = date(2026, 8, 1)


def _sig(sid, ticker, model, source):
    return Signal(
        id=sid, run_date=RUN, ticker=ticker, direction="LONG",
        signal_model=model, signal_source=source,
        entry_price=100.0, stop_loss=97.0, target_1=106.0, target_2=None,
        holding_period_days=3, confidence=80.0, risk_gate_decision="APPROVE",
        regime="bull", features={},
    )


def _out(sid, ticker, *, still_open=True, skip=None, pnl=None):
    return Outcome(signal_id=sid, ticker=ticker, entry_date=RUN,
                   entry_price=100.0, still_open=still_open,
                   skip_reason=skip, pnl_pct=pnl, exit_date=None if still_open else RUN)


async def _db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _seed(factory):
    """One run with every row type the queries must distinguish."""
    async with factory() as s:
        s.add(DailyRun(run_date=RUN, regime="bull", universe_size=2000,
                       candidates_scored=5, execution_mode="quant_only"))
        # official taken, official SHADOW-BLOCKED, official gap-rejected,
        # manual sleeve, PEAD paper — all on distinct tickers except AAA/EEE.
        s.add_all([
            _sig(1, "AAA", "sniper", "mas_official"),
            _sig(2, "BBB", "sniper", "mas_official"),
            _sig(3, "CCC", "mean_reversion", "mas_official"),
            _sig(4, "DDD", "mean_reversion", "mr_manual_sleeve"),
            _sig(5, "EEE", "pead", "pead_paper"),
        ])
        s.add_all([
            _out(1, "AAA"),                                  # taken, open
            _out(2, "BBB", skip=SHADOW_SKIP_REASON),         # gate-blocked shadow
            _out(3, "CCC", skip="gap_above_limit", still_open=False),
            _out(4, "DDD"),                                  # manual sleeve
            _out(5, "EEE"),                                  # PEAD paper
        ])
        # Candidates are OFFICIAL ranked rows. DDD/EEE share tickers with the
        # non-official signals above — the trap the old identity fell into.
        for i, (tkr, model) in enumerate(
                [("AAA", "sniper"), ("BBB", "sniper"), ("CCC", "mean_reversion"),
                 ("DDD", "mean_reversion"), ("EEE", "mean_reversion")]):
            s.add(Candidate(run_date=RUN, ticker=tkr, close_price=100.0,
                            avg_daily_volume=1e6, composite_score=90 - i,
                            signal_model=model, features={}))
        await s.commit()


# ── Candidate audit identity ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_only_official_non_skipped_signals_count_as_picked():
    """The audit's whole question is picked-vs-passed. A blocked pick marked
    'picked' inverts it; a manual/PEAD signal marked picked pollutes it."""
    import scripts.export_dashboard_data as ex

    engine, factory = await _db()
    await _seed(factory)
    async with factory() as s:
        signals = (await s.execute(select(Signal))).scalars().all()
        outcomes = (await s.execute(select(Outcome))).scalars().all()
        cands = (await s.execute(select(Candidate))).scalars().all()

    payload = ex._candidates_payload(cands, signals, {o.signal_id: o for o in outcomes})
    by = {(r["ticker"], r["model"]): r["picked"] for r in payload}

    assert by[("AAA", "sniper")] is True            # official + taken
    assert by[("BBB", "sniper")] is False           # SHADOW-blocked: never taken
    assert by[("CCC", "mean_reversion")] is False   # gap-rejected: never entered
    assert by[("DDD", "mean_reversion")] is False   # manual sleeve is a separate book
    assert by[("EEE", "mean_reversion")] is False   # PEAD paper is a separate book
    await engine.dispose()


@pytest.mark.asyncio
async def test_pick_identity_includes_model_not_just_ticker():
    """Two models can propose the same ticker on the same day; only the one that
    was actually picked may be tagged."""
    import scripts.export_dashboard_data as ex

    engine, factory = await _db()
    async with factory() as s:
        s.add(DailyRun(run_date=RUN, regime="bull", universe_size=10,
                       candidates_scored=2, execution_mode="quant_only"))
        s.add(_sig(1, "ZZZ", "sniper", "mas_official"))
        s.add(_out(1, "ZZZ"))
        s.add(Candidate(run_date=RUN, ticker="ZZZ", close_price=1.0, avg_daily_volume=1.0,
                        composite_score=90, signal_model="sniper", features={}))
        s.add(Candidate(run_date=RUN, ticker="ZZZ", close_price=1.0, avg_daily_volume=1.0,
                        composite_score=80, signal_model="mean_reversion", features={}))
        await s.commit()
        signals = (await s.execute(select(Signal))).scalars().all()
        outcomes = (await s.execute(select(Outcome))).scalars().all()
        cands = (await s.execute(select(Candidate))).scalars().all()

    by = {(r["ticker"], r["model"]): r["picked"]
          for r in ex._candidates_payload(cands, signals, {o.signal_id: o for o in outcomes})}
    assert by[("ZZZ", "sniper")] is True
    assert by[("ZZZ", "mean_reversion")] is False
    await engine.dispose()


# ── Shadow rows: tracked by the walker, invisible to stats ─────────────────

@pytest.mark.asyncio
async def test_tracker_admits_shadow_rows_but_not_other_skips():
    """check_open_positions must evaluate shadow rows (else they measure
    nothing) while still ignoring gap-rejected ones."""
    from sqlalchemy import or_

    engine, factory = await _db()
    await _seed(factory)
    async with factory() as s:
        rows = (await s.execute(
            select(Outcome).where(
                Outcome.still_open == True,  # noqa: E712
                or_(Outcome.skip_reason.is_(None),
                    Outcome.skip_reason == SHADOW_SKIP_REASON),
            )
        )).scalars().all()
    tickers = {r.ticker for r in rows}
    assert "BBB" in tickers, "shadow-booked row must be tracked"
    assert "AAA" in tickers
    assert "CCC" not in tickers, "gap-rejected row must stay excluded"
    await engine.dispose()


@pytest.mark.asyncio
async def test_shadow_rows_are_invisible_to_the_stats_filter():
    """Every stats query filters skip_reason.is_(None). Shadow rows must not
    survive it — that is what keeps them out of the official record."""
    engine, factory = await _db()
    await _seed(factory)
    async with factory() as s:
        rows = (await s.execute(
            select(Outcome).where(Outcome.skip_reason.is_(None))
        )).scalars().all()
    assert {r.ticker for r in rows} == {"AAA", "DDD", "EEE"}
    await engine.dispose()


# ── Drift monitor reads live streams, grouped correctly ────────────────────

@pytest.mark.asyncio
async def test_drift_groups_closed_live_trades_by_stream_and_skips_shadows(monkeypatch):
    """compute_drift must join Signal, group by model|source, and exclude every
    skipped row — including shadow-booked ones."""
    from contextlib import asynccontextmanager

    import src.research.drift_check as dc

    engine, factory = await _db()
    async with factory() as s:
        s.add(DailyRun(run_date=RUN, regime="bull", universe_size=10,
                       candidates_scored=2, execution_mode="quant_only"))
        s.add_all([_sig(1, "AAA", "sniper", "mas_official"),
                   _sig(2, "BBB", "sniper", "mas_official"),
                   _sig(3, "CCC", "mean_reversion", "mas_official")])
        s.add_all([
            _out(1, "AAA", still_open=False, pnl=2.0),
            _out(2, "BBB", still_open=False, pnl=-99.0, skip=SHADOW_SKIP_REASON),
            _out(3, "CCC", still_open=False, pnl=1.0),
        ])
        await s.commit()

    @asynccontextmanager
    async def _fake():
        async with factory() as s:
            yield s

    monkeypatch.setattr(dc, "get_session", _fake)
    report = await dc.compute_drift(lookback_days=3650)
    streams = {s.stream: s for s in report.streams}

    assert "sniper|mas_official" in streams
    assert streams["sniper|mas_official"].n == 1, "shadow row must not enter drift"
    assert streams["sniper|mas_official"].live_avg == pytest.approx(2.0)
    assert streams["mean_reversion|mas_official"].n == 1
    await engine.dispose()
