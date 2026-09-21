"""The integration suite must refuse any database that is not local.

`tests/integration/test_pipeline_smoke.py` runs the real morning pipeline and
persists today's `DailyRun`, `Signal` and `Outcome` rows. pytest reads the
developer's `.env`, whose `DATABASE_URL` is a remote managed Postgres, so
before this guard `pytest -m integration` on a laptop would overwrite the real
run record for the day and insert positions in synthetic tickers.

These run in the UNIT suite deliberately: the guard has to be verified without
a database, and it protects the file that would otherwise need one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.db_guard import dsn_rejection_reason, is_local_dsn

CI_DSN = "postgresql://postgres:postgres@localhost:5432/mas_ci"


@pytest.mark.parametrize("url", [
    CI_DSN,
    "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/mas_ci",
    "postgresql://u:p@LOCALHOST:5432/mas_ci",   # host comparison is case-insensitive
])
def test_local_postgres_is_allowed(url: str) -> None:
    assert is_local_dsn(url), dsn_rejection_reason(url)


@pytest.mark.parametrize("url", [
    # The exact shape of the DSN in the developer .env that made this necessary.
    "postgresql://u:p@c55vaqijj0vpoi.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/db",
    "postgresql://u:p@ep-example-123456.us-east-2.aws.neon.tech/neondb",
    "postgres://u:p@db.internal:5432/mas",
])
def test_remote_databases_are_refused(url: str) -> None:
    reason = dsn_rejection_reason(url)
    assert reason and "not a local database" in reason
    # The message has to say why, or the next person just deletes the guard.
    assert "WRITE" in reason


@pytest.mark.parametrize("url", ["", "sqlite+aiosqlite:///:memory:"])
def test_missing_or_non_postgres_is_refused(url: str) -> None:
    """Not a safe fallback: these tests need real Postgres (JSONB models)."""
    assert dsn_rejection_reason(url) is not None


def test_the_guard_is_actually_wired_into_collection() -> None:
    """A guard nothing calls is decoration.

    Asserted on the conftest source rather than by running pytest-in-pytest:
    the guard aborts the session, which a nested run cannot observe cleanly.
    """
    conftest = (Path(__file__).parent / "integration" / "conftest.py").read_text()

    assert "dsn_rejection_reason" in conftest
    assert "pytest.exit" in conftest
    # Session-scoped and autouse: set up before any connection is opened, and
    # confined to this directory so the default unit run is unaffected.
    assert 'scope="session", autouse=True' in conftest


def test_no_environment_variable_can_switch_the_guard_off() -> None:
    """An opt-out is set once and then inherited forever; there must be none."""
    guard = (Path(__file__).parent / "integration" / "db_guard.py").read_text()

    assert "os.environ" not in guard
    assert "getenv" not in guard
