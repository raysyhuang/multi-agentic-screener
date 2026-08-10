"""`init_db` must refuse to bootstrap an unmigrated database.

Silently running `create_all` against an empty database is what left the schema
with no migration provenance — the tables existed, alembic was pointed at them
afterwards, and the chain was never able to build them again. The failure has to
be loud at the point of bootstrap, because every later symptom (a mirror that
cannot be rebuilt, a chain that only works on a pre-existing database) is
indistinguishable from a healthy system until someone tries to rebuild.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from src.db import session as session_mod


@pytest.fixture
def virgin_db(monkeypatch):
    """An empty database — no alembic_version, no tables."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    monkeypatch.setattr(session_mod, "get_engine", lambda: engine)
    return engine


@pytest.mark.asyncio
async def test_init_db_raises_on_unmigrated_database(virgin_db) -> None:
    with pytest.raises(RuntimeError) as exc:
        await session_mod.init_db()

    message = str(exc.value)
    assert "alembic upgrade head" in message, "the error must state the fix"
    assert "never been migrated" in message


@pytest.mark.asyncio
async def test_init_db_is_a_noop_on_a_migrated_database(virgin_db) -> None:
    """Present alembic_version => schema is migration-managed => nothing to do."""
    from sqlalchemy import text

    async with virgin_db.begin() as conn:
        await conn.execute(text("CREATE TABLE alembic_version (version_num varchar(32))"))

    await session_mod.init_db()  # must not raise, must not create anything


@pytest.mark.asyncio
async def test_escape_hatch_is_opt_in_only(virgin_db, monkeypatch) -> None:
    """The throwaway-database path exists but never engages by accident."""
    monkeypatch.delenv("MAS_DB_ALLOW_CREATE_ALL", raising=False)
    with pytest.raises(RuntimeError):
        await session_mod.init_db()

    # Set to anything other than exactly "1" and it stays closed.
    monkeypatch.setenv("MAS_DB_ALLOW_CREATE_ALL", "true")
    with pytest.raises(RuntimeError):
        await session_mod.init_db()
