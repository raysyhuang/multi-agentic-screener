"""Tests for the dashboard data exporter — in-memory DB, no network."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Reuse the JSONB->JSON sqlite shim registered by the persistence test module.
from tests.test_db import test_exit_config_persistence  # noqa: F401

from src.db.models import Base, DailyRun, Outcome, Signal


def _signal(run_date, ticker, model, source, sid=None):
    return Signal(
        id=sid, run_date=run_date, ticker=ticker, direction="LONG",
        signal_model=model, signal_source=source,
        entry_price=100.0, stop_loss=97.0, target_1=106.0, target_2=None,
        holding_period_days=3, confidence=80.0, risk_gate_decision="APPROVE",
        regime="bull",
        features={"model_raw_score": 76.0, "model_components": {"bb_squeeze": 50}},
    )


@pytest.mark.asyncio
async def test_snapshot_shape_and_stream_separation(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    d1, d2 = date(2026, 7, 20), date(2026, 7, 21)
    async with factory() as s:
        s.add(DailyRun(run_date=d1, regime="bull", universe_size=1800,
                       candidates_scored=70, execution_mode="quant_only",
                       pipeline_health={"status": "OK", "warnings": []}))
        s.add(DailyRun(run_date=d2, regime="bull", universe_size=1816,
                       candidates_scored=74, execution_mode="quant_only",
                       pipeline_health={"status": "WARN", "warnings": ["fmp degraded"]}))
        s.add(_signal(d1, "AAA", "sniper", "mas_official", sid=1))
        s.add(_signal(d1, "BBB", "mean_reversion", "mr_manual_sleeve", sid=2))
        s.add(_signal(d1, "CCC", "mean_reversion", "mr_manual_sleeve", sid=3))
        s.add(_signal(d2, "DDD", "sniper", "mas_official", sid=4))
        # closed win (official sniper)
        s.add(Outcome(signal_id=1, ticker="AAA", entry_date=d1, entry_price=100.0,
                      exit_date=d2, exit_price=102.0, exit_reason="trail_stop",
                      pnl_pct=2.0, max_favorable=2.5, max_adverse=-0.5, still_open=False))
        # closed loss (sleeve MR)
        s.add(Outcome(signal_id=2, ticker="BBB", entry_date=d1, entry_price=100.0,
                      exit_date=d2, exit_price=98.0, exit_reason="stop",
                      pnl_pct=-2.0, max_favorable=0.3, max_adverse=-2.2, still_open=False))
        # unfilled skip (sleeve MR) — must be counted, never a trade
        s.add(Outcome(signal_id=3, ticker="CCC", entry_date=d1, entry_price=100.0,
                      still_open=False, skip_reason="gap_above_limit"))
        # open position (official sniper)
        s.add(Outcome(signal_id=4, ticker="DDD", entry_date=d2, entry_price=100.0,
                      pnl_pct=0.5, still_open=True))
        await s.commit()

    # Point the exporter's session at the fixture DB.
    from contextlib import asynccontextmanager
    import scripts.export_dashboard_data as exp

    @asynccontextmanager
    async def _fake_session():
        async with factory() as session:
            yield session

    monkeypatch.setattr(exp, "get_session", _fake_session)

    snap = await exp.build_snapshot(days=90)

    assert snap["latest_run"]["date"] == "2026-07-21"
    # Today = latest run's picks only.
    assert [p["ticker"] for p in snap["today_picks"]] == ["DDD"]
    assert snap["today_picks"][0]["raw_score"] == 76.0
    # Streams never blended.
    assert set(snap["trades"]) == {"sniper|mas_official", "mean_reversion|mr_manual_sleeve"}
    assert [t["ticker"] for t in snap["trades"]["sniper|mas_official"]] == ["AAA"]
    assert [t["ticker"] for t in snap["trades"]["mean_reversion|mr_manual_sleeve"]] == ["BBB"]
    # Skips counted, not simulated as trades.
    assert snap["skip_counts"] == {"mean_reversion|mr_manual_sleeve": 1}
    # Open positions listed separately.
    assert [o["ticker"] for o in snap["open_positions"]] == ["DDD"]
    # Run history includes health status.
    assert [r["health"] for r in snap["run_history"]] == ["OK", "WARN"]
    # Baselines present for the expectation bands.
    assert "sniper|mas_official" in snap["baselines"]
    await engine.dispose()
