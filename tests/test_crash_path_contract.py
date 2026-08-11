"""A crashed run must leave exactly one governance record, whenever it crashes.

`run_morning_pipeline` is the fail-closed boundary: nothing may escape it
without leaving evidence. Three ways that guarantee was incomplete, all found in
review of the crash-path work:

1. The crash handler only persisted governance if the core had already created
   the context. `_check_paper_gate()` and the database work ahead of it are
   fallible, so an early crash produced a `final_output` artifact and no
   governance record — the contract held only for late failures.
2. A crash AFTER the core wrote its `governance: success` artifact appended a
   second `governance: failed` row for the same run. Nothing enforces
   uniqueness on `(run_id, stage)`, so a consumer asking "did this run succeed"
   got whichever row it happened to select.
3. `_trading_date_et()` and `get_settings()` ran outside the try, so a bad
   timezone database or unparseable `.env` escaped fail-closed entirely: no
   record, no alert, the run simply vanished. The failure most likely to affect
   every run at once was the one that left no trace.

These are unit-level and mock the database, because the point is the control
flow of the handler. The end-to-end version lives in the integration smoke test.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import main as main_mod


class _Session:
    """Captures what the failure handler writes, and what it mutates."""

    def __init__(self, existing: list | None = None):
        self.added: list = []
        self._existing = existing or []

    def add(self, obj) -> None:
        self.added.append(obj)

    async def execute(self, stmt):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        result.scalars.return_value.all.return_value = self._existing
        return result

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass


@pytest.fixture
def captured(monkeypatch):
    """Route the handler's DB writes into a capture object."""
    from contextlib import asynccontextmanager

    session = _Session()

    @asynccontextmanager
    async def fake_get_session():
        yield session

    monkeypatch.setattr(main_mod, "get_session", fake_get_session)
    monkeypatch.setattr(main_mod, "send_alert", AsyncMock(return_value=False))
    return session


def _stages(session: _Session) -> list[str]:
    return [getattr(a, "stage", None) for a in session.added]


@pytest.mark.asyncio
async def test_a_crash_before_governance_exists_still_records_one(
    captured, monkeypatch
) -> None:
    """The early-crash hole: fail inside the core before the context is made."""

    async def boom(*a, **k):
        raise RuntimeError("paper gate exploded")

    monkeypatch.setattr(main_mod, "_run_pipeline_core", boom)

    await main_mod.run_morning_pipeline()  # must not raise

    assert "governance" in _stages(captured), (
        "an early crash left no governance record; the guarantee only held for "
        f"late failures. Wrote: {_stages(captured)}"
    )
    gov_artifact = next(
        a for a in captured.added if getattr(a, "stage", None) == "governance"
    )
    assert gov_artifact.status == "failed"
    flags = gov_artifact.payload.get("governance_flags", [])
    assert any("synthesized" in f for f in flags), (
        f"a synthesized record should say so; flags={flags}"
    )
    assert any("paper gate exploded" in str(e) for e in gov_artifact.errors)


@pytest.mark.asyncio
async def test_a_late_crash_replaces_the_success_record_rather_than_duplicating(
    monkeypatch
) -> None:
    """Two rows with opposite statuses and no tiebreak is worse than one wrong."""
    from contextlib import asynccontextmanager

    existing = MagicMock()
    existing.stage = "governance"
    existing.status = "success"
    session = _Session(existing=[existing])

    @asynccontextmanager
    async def fake_get_session():
        yield session

    monkeypatch.setattr(main_mod, "get_session", fake_get_session)
    monkeypatch.setattr(main_mod, "send_alert", AsyncMock(return_value=False))

    async def boom(*a, **k):
        raise RuntimeError("crashed after governance was written")

    monkeypatch.setattr(main_mod, "_run_pipeline_core", boom)

    await main_mod.run_morning_pipeline()

    assert "governance" not in _stages(session), (
        "a second governance row was appended alongside the existing one"
    )
    assert existing.status == "failed", "the existing record must be corrected"
    assert any("crashed after governance" in str(e) for e in existing.errors)


@pytest.mark.asyncio
async def test_a_config_failure_does_not_escape_fail_closed(
    captured, monkeypatch
) -> None:
    """get_settings() ran outside the try — the run vanished without a trace."""

    def boom():
        raise RuntimeError("unparseable .env")

    monkeypatch.setattr(main_mod, "get_settings", boom)

    await main_mod.run_morning_pipeline()  # must not raise

    stages = _stages(captured)
    assert "final_output" in stages, f"no fail-closed record written; got {stages}"
    assert "governance" in stages


@pytest.mark.asyncio
async def test_a_trading_date_failure_does_not_escape_either(
    captured, monkeypatch
) -> None:
    def boom():
        raise RuntimeError("tz database missing")

    monkeypatch.setattr(main_mod, "_trading_date_et", boom)

    await main_mod.run_morning_pipeline()

    assert "final_output" in _stages(captured)
    # It still has to be filed under some date.
    run_dates = {getattr(a, "run_date", None) for a in captured.added}
    assert date.today() in run_dates
