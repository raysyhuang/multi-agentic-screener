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
    """Neutralize everything `main()` does before it dispatches the flag.

    It opens the database and calls `settings.validate_keys_for_mode()` first,
    so without stubbing both, these tests depend on whatever DATABASE_URL and
    API keys the machine happens to carry — passing against a populated local
    `.env` and failing on a bare CI runner. That is the third time in this
    session the same `.env`-precedence trap has produced a green local run and a
    red remote one; the environment a test needs has to be stated, not
    inherited.
    """
    settings = MagicMock()
    settings.validate_keys_for_mode.return_value = None
    settings.execution_mode = "quant_only"
    monkeypatch.setattr(main_mod, "init_db", AsyncMock())
    monkeypatch.setattr(main_mod, "get_settings", lambda: settings)
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
async def test_the_scheduler_wrapper_raises_on_fail_closed(monkeypatch) -> None:
    """APScheduler counts a normal return as success.

    Registering `run_morning_pipeline` directly meant a fail-closed run never
    reached the EVENT_JOB_ERROR listener, so the scheduler's own alert stayed
    silent — the long-running equivalent of exiting 0.
    """
    monkeypatch.setattr(
        main_mod, "run_morning_pipeline", AsyncMock(return_value=False)
    )

    with pytest.raises(RuntimeError, match="fail-closed"):
        await main_mod.scheduled_morning_pipeline()


@pytest.mark.parametrize("returned", [True, None, 0, "", []])
@pytest.mark.asyncio
async def test_only_a_literal_false_raises(monkeypatch, returned) -> None:
    """`is False`, not truthiness.

    Falsey-but-not-False values must pass through: coercing them would turn
    healthy runs into scheduler alerts, and `run_afternoon_check` returns None
    on every successful execution.
    """
    monkeypatch.setattr(
        main_mod, "run_morning_pipeline", AsyncMock(return_value=returned)
    )

    await main_mod.scheduled_morning_pipeline()  # must not raise


def test_both_schedulers_register_the_wrapper() -> None:
    """Two registrations exist; fixing one and missing the other is the bug."""
    from pathlib import Path

    for module in ("src/main.py", "src/worker.py"):
        source = Path(module).read_text()
        assert "scheduled_morning_pipeline,\n" in source, (
            f"{module} does not register the wrapper — a fail-closed run there "
            "would still be recorded by APScheduler as a success"
        )
        assert "        run_morning_pipeline,\n" not in source, (
            f"{module} still registers run_morning_pipeline directly"
        )


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
