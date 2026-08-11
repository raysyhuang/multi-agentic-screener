"""A fail-closed run must be visible from outside the process.

Fail-closed is the right behaviour: catch, write the NoTrade record, alert,
return. But returning quietly meant the process exited 0, so the GitHub workflow
went green — and `pipeline-failure-alert.yml` fires only on a failed conclusion,
so the standing safety net stayed silent on the one day it was needed.

On 2026-08-11 the run that took the book to NoTrade reported
`conclusion: success`. The only reason it was noticed was the pipeline's own
Telegram alert; a crash inside the DB-write or alert section would have produced
silence and a green tick.

This is the third instance of the same shape: a green run that does not mean the
pipeline ran. The others were the DST no-op (a 48s skip reading as success) and
the fail-closed NoTrade here.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import main as main_mod


@pytest.fixture
def worker_without_key_validation(monkeypatch):
    """Stub the worker's startup key check.

    `start_worker()` calls `settings.validate_keys_for_mode()`, which passes on
    a developer machine holding a populated `.env` and raises in CI where none
    exists — so these tests would pass locally and fail on the runner. Same
    `.env`-precedence trap that made "TELEGRAM_BOT_TOKEN unset" an unreliable
    guarantee on the VPS mirror.
    """
    import src.worker as worker_mod

    settings = MagicMock()
    settings.validate_keys_for_mode.return_value = None
    monkeypatch.setattr(worker_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(worker_mod, "init_db", AsyncMock())
    return worker_mod


@pytest.mark.asyncio
async def test_a_successful_run_reports_true(monkeypatch) -> None:
    async def fine(*a, **k):
        return None

    monkeypatch.setattr(main_mod, "_run_pipeline_core", fine)

    assert await main_mod.run_morning_pipeline() is True


@pytest.mark.asyncio
async def test_a_fail_closed_run_reports_false(monkeypatch) -> None:
    """The record and the alert still happen — only the silence is removed."""
    from contextlib import asynccontextmanager

    async def boom(*a, **k):
        raise RuntimeError("induced")

    @asynccontextmanager
    async def fake_session():
        class _S:
            async def execute(self, *a, **k):
                from unittest.mock import MagicMock

                r = MagicMock()
                r.scalar_one_or_none.return_value = None
                r.scalars.return_value.all.return_value = []
                return r

            def add(self, *a, **k):
                pass

            async def commit(self):
                pass

            async def rollback(self):
                pass

        yield _S()

    alert = AsyncMock(return_value=False)
    monkeypatch.setattr(main_mod, "_run_pipeline_core", boom)
    monkeypatch.setattr(main_mod, "get_session", fake_session)
    monkeypatch.setattr(main_mod, "send_alert", alert)

    result = await main_mod.run_morning_pipeline()

    assert result is False, "a fail-closed run must not report success"
    assert alert.await_count == 1, "the alert must still be sent"


@pytest.mark.asyncio
async def test_the_one_off_worker_exits_non_zero_on_fail_closed(
    monkeypatch, worker_without_key_validation
) -> None:
    """This is what turns the workflow red and fires the failure alert."""
    worker_mod = worker_without_key_validation

    monkeypatch.setattr(worker_mod.sys, "argv", ["worker", "--run-now"])
    monkeypatch.setattr(worker_mod, "run_morning_pipeline", AsyncMock(return_value=False))

    with pytest.raises(SystemExit) as exc:
        await worker_mod.start_worker()

    assert exc.value.code == 3, (
        "the documented fail-closed code is 3; asserting merely non-zero would "
        "let it drift and silently break anything keying on the value"
    )


@pytest.mark.asyncio
async def test_the_one_off_worker_exits_zero_on_success(
    monkeypatch, worker_without_key_validation
) -> None:
    """The happy path must stay quiet — no false alarms on healthy runs."""
    worker_mod = worker_without_key_validation

    monkeypatch.setattr(worker_mod.sys, "argv", ["worker", "--run-now"])
    monkeypatch.setattr(worker_mod, "run_morning_pipeline", AsyncMock(return_value=True))

    await worker_mod.start_worker()  # returns normally, no SystemExit


@pytest.fixture
def direct_entrypoint(monkeypatch):
    """`main()` opens the database before dispatching flags — stub that out.

    Without it these tests reach for a real Postgres, which passes or fails
    depending on whatever DATABASE_URL the machine happens to carry.
    """
    monkeypatch.setattr(main_mod, "init_db", AsyncMock())
    monkeypatch.setattr(main_mod, "_setup_logging", lambda *a, **k: None)
    return main_mod


@pytest.mark.asyncio
async def test_the_direct_module_entrypoint_also_exits_three(
    monkeypatch, direct_entrypoint
) -> None:
    """`python -m src.main --run-now` is documented and must honour the contract.

    Fixing only src.worker would make the guarantee true for the Actions route
    and false for anyone invoking the module directly — including a human
    re-running by hand after an incident, which is exactly when a misleading
    exit code does the most damage.
    """
    monkeypatch.setattr(sys, "argv", ["main", "--run-now"])
    monkeypatch.setattr(
        main_mod, "run_morning_pipeline", AsyncMock(return_value=False)
    )

    with pytest.raises(SystemExit) as exc:
        await main_mod.main()

    assert exc.value.code == 3


@pytest.mark.asyncio
async def test_the_direct_entrypoint_exits_zero_when_clean(
    monkeypatch, direct_entrypoint
) -> None:
    monkeypatch.setattr(sys, "argv", ["main", "--run-now"])
    monkeypatch.setattr(
        main_mod, "run_morning_pipeline", AsyncMock(return_value=True)
    )

    await main_mod.main()  # returns normally


@pytest.mark.asyncio
async def test_the_direct_afternoon_entrypoint_is_unaffected(
    monkeypatch, direct_entrypoint
) -> None:
    """It returns None and raises on error — already surfaces, leave it be."""
    monkeypatch.setattr(sys, "argv", ["main", "--check-now"])
    monkeypatch.setattr(
        main_mod, "run_afternoon_check", AsyncMock(return_value=None)
    )

    await main_mod.main()  # must not raise SystemExit


@pytest.mark.asyncio
async def test_a_job_returning_none_is_not_treated_as_failure(
    monkeypatch, worker_without_key_validation
) -> None:
    """`run_afternoon_check` returns None and raises on error — leave it alone.

    Testing `if ok is False` rather than `if not ok` matters: the afternoon
    check has no handler of its own, so an exception there already propagates
    and exits non-zero. Treating its `None` as failure would make every healthy
    afternoon run red.
    """
    worker_mod = worker_without_key_validation

    monkeypatch.setattr(worker_mod.sys, "argv", ["worker", "--check-now"])
    monkeypatch.setattr(worker_mod, "run_afternoon_check", AsyncMock(return_value=None))

    await worker_mod.start_worker()  # must not raise SystemExit
