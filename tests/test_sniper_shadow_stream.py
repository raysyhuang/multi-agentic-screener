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

from pathlib import Path

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


_MR_SHADOW = [{
    "ticker": "PBR", "direction": "LONG", "entry_price": 12.0,
    "stop_loss": 11.6, "target_1": 12.8, "confidence": 71,
    "holding_period": 3, "also_in_mas": False,
}]


def test_mr_shadow_entries_are_visible_wherever_its_exits_are():
    """Entries and outcomes must both show, or neither.

    `format_outcome_alert` labels mr_shadow closures, so an alert with no MR
    shadow entry section reports positions closing that the reader never saw
    open. That asymmetry is the defect this pins.
    """
    from src.output.telegram import format_outcome_alert

    msg = format_daily_alert(_OFFICIAL, "choppy", "2026-09-21", mr_shadow_picks=_MR_SHADOW)
    assert "Mean Reversion — Shadow" in msg
    assert "PBR" in msg and "not traded and not in the book" in msg
    assert msg.index("XOM") < msg.index("PBR")

    outcome = format_outcome_alert([
        {"ticker": "PBR", "pnl_pct": -1.0, "exit_reason": "time_stop",
         "signal_source": "mr_shadow"},
    ])
    assert "MR shadow" in outcome


@pytest.mark.parametrize("kwargs", [
    {},
    {"validation_failed": True, "failed_checks": ["mean_reversion: x"]},
])
def test_mr_shadow_section_renders_on_every_alert_branch(kwargs):
    """With the book empty, the empty-picks body is the ONLY body in production."""
    msg = format_daily_alert([], "choppy", "2026-09-21", mr_shadow_picks=_MR_SHADOW, **kwargs)
    assert "Mean Reversion — Shadow" in msg and "PBR" in msg


def test_mr_shadow_section_with_no_setups_says_so():
    msg = format_daily_alert(_OFFICIAL, "choppy", "2026-09-21", mr_shadow_picks=[])
    assert "Mean Reversion — Shadow" in msg and "No mean-reversion setups today" in msg


def test_mr_shadow_section_absent_when_stream_not_passed():
    msg = format_daily_alert(_OFFICIAL, "choppy", "2026-09-21")
    assert "Mean Reversion — Shadow" not in msg and "PBR" not in msg


def test_every_labeled_non_book_source_can_be_seen_entering():
    """A source labeled in outcome alerts needs an entry section somewhere.

    Keeps the next retirement from reinstating the asymmetry by wiring only
    the closure half. pead_60d_shadow is exempt by construction: it is paired
    to an already-alerted entry and is suppressed from outcome alerts too.
    """
    import inspect
    from src.output import telegram as tg

    sig = inspect.signature(tg.format_daily_alert).parameters
    alerted = set(tg._NON_BOOK_SOURCE_LABELS) - {"pead_60d_shadow"}
    entry_params = {
        "mr_manual_sleeve": "manual_sleeve_picks",
        "pead_paper": "pead_paper_picks",
        "pead_neglected": "pead_paper_picks",
        "sniper_shadow": "sniper_shadow_picks",
        "mr_shadow": "mr_shadow_picks",
    }
    assert alerted <= set(entry_params), f"unmapped labeled sources: {alerted - set(entry_params)}"
    for source in alerted:
        assert entry_params[source] in sig, f"{source} has no entry section parameter"


def test_quarantined_pead_cannot_suppress_an_official_pick():
    """Official cooldown is defined from what the book IS, not "not shadow".

    pead_paper and pead_neglected are quarantined but were never in
    SHADOW_SOURCES, so the complement let a paper PEAD row knock out an
    eligible official pick — the reverse of the quarantine.
    """
    from src import main as m
    from src.streams import BOOK_SOURCES

    recent = [
        {"ticker": "AAA", "signal_source": "mas_official"},
        {"ticker": "BBB", "signal_source": "pead_paper"},
        {"ticker": "CCC", "signal_source": "pead_neglected"},
        {"ticker": "DDD", "signal_source": "sniper_shadow"},
        {"ticker": "EEE", "signal_source": "mr_shadow"},
        {"ticker": "LEG", "signal_source": None},        # legacy row = official
    ]
    official = [
        r for r in recent
        if (r.get("signal_source") or "mas_official") in BOOK_SOURCES
    ]

    assert [r["ticker"] for r in official] == ["AAA", "LEG"]
    assert m.BOOK_SOURCES is BOOK_SOURCES
    source = (Path(__file__).parents[1] / "src" / "main.py").read_text()
    assert 'in BOOK_SOURCES' in source
    assert 'not in SHADOW_SOURCES]' not in source


def test_stream_classification_has_one_definition():
    """Re-exports must be the same objects, or the sets can drift apart."""
    from src import main as m
    from src import streams

    assert m.SHADOW_SOURCES is streams.SHADOW_SOURCES
    assert m.PEAD_POSITION_SOURCES is streams.PEAD_POSITION_SOURCES
    assert m.SNIPER_CAP_SOURCES is streams.SNIPER_CAP_SOURCES
    assert m.PAIRED_OBSERVATION_SOURCES is streams.PAIRED_OBSERVATION_SOURCES

    # Production code CLASSIFIES by the shared sets. Writing the label once
    # where the stream is created, and using it as a display-table key, are
    # fine; a comparison against the literal is the drift that finding 8 was.
    allowed_prefixes = (
        'pick.signal_source = ',        # the one site that creates the stream
        '"pead|pead_60d_shadow": ',     # baseline/label table keys
    )
    for path in ("src/main.py", "src/output/performance.py",
                 "src/research/drift_check.py", "scripts/export_dashboard_data.py"):
        text = (Path(__file__).parents[1] / path).read_text()
        for ln in text.splitlines():
            stripped = ln.strip()
            # Only the quoted SOURCE literal; `settings.pead_60d_shadow_*` is
            # a config attribute name, not a classification.
            if '"pead_60d_shadow"' not in stripped and '"pead|pead_60d_shadow"' not in stripped:
                continue
            if stripped.startswith("#"):
                continue
            assert stripped.startswith(allowed_prefixes), (
                f"{path} classifies by literal: {stripped}"
            )


def test_book_streams_follow_the_admission_flag():
    """Dashboard book composition is derived from the same setting the pipeline
    admits on, so restoring sniper cannot leave execution and the book disagreeing."""
    import scripts.export_dashboard_data as exp

    assert exp.book_streams(False) == ["mean_reversion|mas_official"]
    assert exp.book_streams(True) == ["sniper|mas_official", "mean_reversion|mas_official"]
    assert exp.book_streams(False, False) == []
    assert exp.book_streams(True, False) == ["sniper|mas_official"]
    assert [k for k, _, _ in exp.portfolio_specs(True)] == ["sniper", "mr", "book"]
    assert exp.portfolio_specs(True)[2][2] == exp.book_streams(True)
    assert "retired" in exp.portfolio_specs(False)[0][1]
    assert "retired" not in exp.portfolio_specs(True)[0][1]
    settings = Settings(_env_file=None)
    assert exp.BOOK_STREAMS == exp.book_streams(
        settings.sniper_in_book, settings.mean_reversion_in_book,
    )


def test_both_retired_models_default_to_shadow():
    from src import main as m

    settings = Settings(_env_file=None)
    assert settings.sniper_in_book is False
    assert settings.mean_reversion_in_book is False
    assert {"sniper_shadow", "mr_shadow", "pead_60d_shadow"} <= m.SHADOW_SOURCES
