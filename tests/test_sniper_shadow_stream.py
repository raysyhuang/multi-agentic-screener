"""Sniper shadow stream (2026-09-18): sniper left the official book and keeps
recording under signal_source="sniper_shadow", mirroring the PEAD quarantine.

What these pin:
  * the default is OUT of the book (`sniper_in_book=False`);
  * the concurrency cap counts BOTH the legacy official and the shadow sources —
    a shadow-only filter would have read zero on the day the stream moved and
    never bound again (the PEAD per-run-vs-concurrent cap bug, in a new coat);
  * the alert renders shadow picks in their own labeled section, after the
    official picks, and omits the section entirely when the stream is absent.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import date, timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Reuse the JSONB->JSON sqlite shim registered by the persistence test module.
from tests.test_db import test_exit_config_persistence  # noqa: F401

from src.config import Settings
from src.db.models import Base, Outcome, Signal
from src.output.telegram import format_daily_alert

RUN = date(2026, 9, 18)


def test_sniper_is_out_of_the_book_by_default():
    assert Settings(_env_file=None).sniper_in_book is False


def _sig(sid, ticker, model, source):
    return Signal(
        id=sid, run_date=RUN, ticker=ticker, direction="LONG",
        signal_model=model, signal_source=source,
        entry_price=100.0, stop_loss=97.0, target_1=106.0, target_2=None,
        holding_period_days=7, confidence=80.0, risk_gate_decision="APPROVE",
        regime="bull", features={},
    )


def _open(sid, ticker, still_open=True):
    return Outcome(signal_id=sid, ticker=ticker, entry_date=RUN, entry_price=100.0,
                   still_open=still_open, exit_date=None if still_open else RUN)


async def _db_with(rows):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        s.add_all(rows)
        await s.commit()

    @asynccontextmanager
    async def _fake_session():
        async with factory() as session:
            yield session

    return engine, _fake_session


@pytest.mark.asyncio
async def test_open_sniper_count_spans_official_and_shadow_sources(monkeypatch):
    """One legacy official sniper + one shadow sniper open = 2 slots used. A
    PEAD position, a closed sniper, a manual-sleeve MR row AND an open sniper
    under some OTHER source never count — the last one is what pins that the
    predicate is a source allow-list, not "any sniper"."""
    engine, fake = await _db_with([
        _sig(1, "AAA", "sniper", "mas_official"), _open(1, "AAA"),
        _sig(2, "BBB", "sniper", "sniper_shadow"), _open(2, "BBB"),
        _sig(3, "CCC", "sniper", "sniper_shadow"), _open(3, "CCC", still_open=False),
        _sig(4, "DDD", "pead", "pead_neglected"), _open(4, "DDD"),
        _sig(5, "EEE", "mean_reversion", "mr_manual_sleeve"), _open(5, "EEE"),
        _sig(6, "FFF", "sniper", "cursor_quality_veto"), _open(6, "FFF"),
    ])
    from src import main as m

    monkeypatch.setattr(m, "get_session", fake)
    assert set(m.SNIPER_CAP_SOURCES) == {"mas_official", "sniper_shadow"}
    assert await m._count_open_sniper_positions() == 2
    await engine.dispose()


@pytest.mark.asyncio
async def test_official_cooldown_ignores_shadow_history_but_shadow_sees_all(monkeypatch):
    """Yesterday's SHADOW sniper pick of AAA must not suppress today's official
    MR AAA (a quarantined stream may not shape the book); the shadow stream's
    own cooldown still sees every row, including official history."""
    from types import SimpleNamespace

    from src import main as m
    from src.signals.ranker import apply_cooldown

    # Rows are dated relative to the real clock: `_get_recent_signals` and
    # `apply_cooldown` each call `date.today()` from their own module, so
    # freezing one clock but not the other would make this test expire.
    yesterday = date.today() - timedelta(days=1)
    a, b = _sig(1, "AAA", "sniper", "sniper_shadow"), _sig(2, "BBB", "mean_reversion", "mas_official")
    a.run_date = b.run_date = yesterday
    engine, fake = await _db_with([a, b])
    monkeypatch.setattr(m, "get_session", fake)
    recent = await m._get_recent_signals(days=7)
    await engine.dispose()

    assert {r["signal_source"] for r in recent} == {"sniper_shadow", "mas_official"}
    official_recent = [r for r in recent if r.get("signal_source") not in m.SHADOW_SOURCES]
    today = [SimpleNamespace(ticker="AAA", signal_date=date.today()),
             SimpleNamespace(ticker="BBB", signal_date=date.today())]
    kept_official = {s.ticker for s in apply_cooldown(today, official_recent)}
    kept_shadow = {s.ticker for s in apply_cooldown(today, recent)}
    assert kept_official == {"AAA"}        # shadow AAA history ignored; official BBB suppressed
    assert kept_shadow == set()            # shadow cooldown sees both


def test_count_open_sniper_positions_is_failsafe_on_db_error(monkeypatch):
    """A DB failure widens the cap (returns 0) rather than blocking the run."""
    from src import main as m

    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(m, "get_session", _boom)
    assert asyncio.run(m._count_open_sniper_positions()) == 0


_OFFICIAL = [{
    "ticker": "XOM", "direction": "LONG", "entry_price": 100.0,
    "stop_loss": 97.0, "target_1": 106.0, "confidence": 80,
    "signal_model": "mean_reversion", "holding_period": 3,
}]
_SHADOW = [{
    "ticker": "SMCI", "direction": "LONG", "entry_price": 50.0,
    "stop_loss": 46.0, "target_1": 62.0, "confidence": 74,
    "holding_period": 7, "also_in_mas": False,
}]


def test_shadow_section_labeled_and_after_official_picks():
    msg = format_daily_alert(_OFFICIAL, "bull", "2026-09-18", sniper_shadow_picks=_SHADOW)
    assert "Sniper — Shadow" in msg
    assert "SMCI" in msg
    assert "not traded and not in the book" in msg
    assert msg.index("XOM") < msg.index("SMCI")


def test_shadow_section_absent_when_stream_not_passed():
    msg = format_daily_alert(_OFFICIAL, "bull", "2026-09-18")
    assert "Shadow" not in msg and "SMCI" not in msg


@pytest.mark.parametrize("kwargs", [
    {},                                   # no official picks
    {"validation_failed": True, "failed_checks": ["mean_reversion: x"]},
])
def test_shadow_section_renders_on_every_alert_branch(kwargs):
    """Empty-pick and validation-failed bodies still carry the shadow section,
    exactly as the PEAD paper section does."""
    msg = format_daily_alert([], "bull", "2026-09-18", sniper_shadow_picks=_SHADOW, **kwargs)
    assert "Sniper — Shadow" in msg and "SMCI" in msg


def test_shadow_section_with_no_setups_says_so():
    msg = format_daily_alert(_OFFICIAL, "bull", "2026-09-18", sniper_shadow_picks=[])
    assert "Sniper — Shadow" in msg and "No sniper setups today" in msg


def test_afternoon_outcomes_keep_shadow_out_of_the_book_totals():
    """The afternoon alert's headline count/wins/net is the OFFICIAL book only;
    shadow and paper rows are listed under their own caption with a label."""
    from src.output.telegram import format_outcome_alert

    msg = format_outcome_alert([
        {"ticker": "XOM", "pnl_pct": 2.0, "exit_reason": "target", "signal_source": "mas_official"},
        {"ticker": "SMCI", "pnl_pct": -6.0, "exit_reason": "time_stop", "signal_source": "sniper_shadow"},
        {"ticker": "RBRK", "pnl_pct": 16.0, "exit_reason": "open", "signal_source": "pead_neglected"},
        {"ticker": "LEGACY", "pnl_pct": -1.0, "exit_reason": "stop"},   # no source = official (legacy)
    ])
    assert "Positions: <b>2</b>" in msg and "Wins: <b>1/2</b>" in msg and "Net: <b>+1.00%</b>" in msg
    assert "Paper / shadow" in msg and "not in the book" in msg
    assert "Sniper shadow" in msg and "PEAD neglected-beat" in msg
    assert msg.index("XOM") < msg.index("Paper / shadow") < msg.index("SMCI")


def test_book_streams_follow_the_admission_flag():
    """Dashboard book composition is derived from the same setting the pipeline
    admits on, so restoring sniper cannot leave execution and the book disagreeing."""
    import scripts.export_dashboard_data as exp

    assert exp.book_streams(False) == ["mean_reversion|mas_official"]
    assert exp.book_streams(True) == ["sniper|mas_official", "mean_reversion|mas_official"]
    assert [k for k, _, _ in exp.portfolio_specs(True)] == ["sniper", "mr", "book"]
    assert exp.portfolio_specs(True)[2][2] == exp.book_streams(True)
    assert "retired" in exp.portfolio_specs(False)[0][1]
    assert "retired" not in exp.portfolio_specs(True)[0][1]
    assert exp.BOOK_STREAMS == exp.book_streams(Settings(_env_file=None).sniper_in_book)
