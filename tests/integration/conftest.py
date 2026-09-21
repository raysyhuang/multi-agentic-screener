"""Shared fixtures for integration tests.

Every test here talks to a real Postgres, and `get_engine()` caches a
module-level singleton while pytest-asyncio gives each test its own event loop.
From the second test onwards, connections belong to a closed loop and writes
fail with `RuntimeError: Event loop is closed`.

This lived inside test_pipeline_smoke.py, so the first integration test file
added afterwards inherited the bug — a fix applied at one site rather than to
the shared cause, which is the same shape as several defects found in review of
this workstream. Centralised so every integration test gets a live engine.
"""

from __future__ import annotations

import pytest

from tests.integration.db_guard import dsn_rejection_reason


@pytest.fixture(scope="session", autouse=True)
def _refuse_a_database_that_is_not_local():
    """Stop the session before the first integration test if the DB is remote.

    These tests write: they run the real morning pipeline and persist today's
    run. Session-scoped and declared in THIS directory's conftest, so it is set
    up before any connection is opened and applies to the integration tests
    only — the default unit run (`-m 'not integration'`) never reaches it.
    """
    from src.config import get_settings

    reason = dsn_rejection_reason(get_settings().database_url)
    if reason:
        pytest.exit(f"Refusing to run integration tests: {reason}", returncode=3)
    yield


@pytest.fixture(autouse=True)
async def _engine_per_test():
    """Dispose the database engine after each test, while its loop is alive."""
    yield
    from src.db.session import close_db

    await close_db()
