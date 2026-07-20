"""End-to-end persistence test: the exit_config snapshot must reach Signal.features.

Closes Neo's coverage refinement on the 2026-07 reconciliation work: the helper
test proves the mapping, but not that it survives the _json_safe -> Signal ORM ->
DB round-trip main.py performs at Step 8. This inserts a Signal built with the
exact production construction pattern into an in-memory SQLite DB and reads it
back.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles

from src.db.models import Base, Signal
from src.main import _json_safe, build_exit_config_snapshot


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    """Render Postgres JSONB as JSON on SQLite so create_all works in tests."""
    return "JSON"


def _settings_stub() -> SimpleNamespace:
    return SimpleNamespace(
        slippage_pct=0.001, trail_activate_pct=0.5, trail_distance_pct=0.3,
        score_tiered_stops_enabled=True, partial_tp_enabled=False,
        sniper_time_stop_days=1, entry_gap_max_atr=0.2,
    )


@pytest.mark.asyncio
async def test_exit_config_snapshot_round_trips_to_signal_features():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    settings = _settings_stub()
    pick_features = {"rsi_2": 4.2, "model_raw_score": 81.0}

    # Exact production construction pattern from main.py Step 8.
    async with session_factory() as session:
        session.add(Signal(
            run_date=date(2026, 7, 21),
            ticker="TEST",
            direction="LONG",
            signal_model="mean_reversion",
            signal_source="mas_official",
            entry_price=100.0,
            stop_loss=98.0,
            target_1=103.0,
            target_2=None,
            holding_period_days=3,
            confidence=80.0,
            risk_gate_decision="APPROVE",
            regime="bull",
            features=_json_safe({**pick_features, "exit_config": build_exit_config_snapshot(settings)}),
        ))
        await session.commit()

    # Fresh session: what actually landed in the DB.
    async with session_factory() as session:
        stored = (await session.execute(select(Signal))).scalar_one()

    assert stored.features is not None
    assert stored.features["rsi_2"] == 4.2                    # pick features preserved
    snap = stored.features.get("exit_config")
    assert snap == {
        "slippage_pct": 0.001,
        "trail_activate_pct": 0.5,
        "trail_distance_pct": 0.3,
        "score_tiered_stops_enabled": True,
        "partial_tp_enabled": False,
        "sniper_time_stop_days": 1,
        "entry_gap_max_atr": 0.2,
    }
    await engine.dispose()
