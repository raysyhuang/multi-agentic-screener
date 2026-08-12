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


@pytest.fixture(autouse=True)
async def _engine_per_test():
    """Dispose the database engine after each test, while its loop is alive."""
    yield
    from src.db.session import close_db

    await close_db()
