"""Async database session management."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import get_settings
from src.db.models import Base

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


# libpq/psycopg-style query params that managed Postgres providers (Neon,
# Supabase, Heroku, ...) append to their connection strings but asyncpg's
# connect() rejects. We drop them and negotiate SSL via connect_args instead.
_ASYNCPG_INCOMPATIBLE_PARAMS = frozenset(
    {"sslmode", "channel_binding", "gssencmode", "sslrootcert", "sslcert", "sslkey"}
)
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", ""})


def _get_async_url() -> str:
    """Normalize DATABASE_URL to the asyncpg driver and strip libpq-only params.

    Converts ``postgres://`` / ``postgresql://`` to ``postgresql+asyncpg://`` and
    removes query params like ``sslmode``/``channel_binding`` that asyncpg does
    not accept (SSL is handled in ``get_engine`` via ``connect_args``).
    """
    url = get_settings().database_url
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    split = urlsplit(url)
    kept = [
        (k, v)
        for k, v in parse_qsl(split.query, keep_blank_values=True)
        if k not in _ASYNCPG_INCOMPATIBLE_PARAMS
    ]
    return urlunsplit(split._replace(query=urlencode(kept)))


def _needs_ssl(url: str) -> bool:
    """True for remote managed Postgres; False for a local dev database."""
    return (urlsplit(url).hostname or "") not in _LOCAL_HOSTS


def get_engine():
    global _engine
    if _engine is None:
        url = _get_async_url()
        connect_args: dict = {}
        # asyncpg ignores sslmode in the URL and defaults to "prefer" (silent
        # plaintext fallback); managed providers (Heroku/RDS, Neon, Supabase)
        # require SSL, so force it for any non-local host.
        if _needs_ssl(url):
            connect_args["ssl"] = "require"
        _engine = create_async_engine(
            url,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Verify the schema exists and is migration-managed.

    This used to call ``Base.metadata.create_all``, which is precisely how the
    schema came to exist without any migration having built it: the first run
    against an empty database created every table from the ORM, alembic was
    pointed at the result months later, and from then on the migration chain
    only ever ran against a database it had not built. `alembic upgrade head`
    on an empty database failed, and no environment could be rebuilt from
    source. Two schema authorities is one too many.

    Alembic is now the only one. A database with no ``alembic_version`` table
    has not been migrated, and silently creating tables here would hide that
    again — so it raises instead. Set ``MAS_DB_ALLOW_CREATE_ALL=1`` for a
    throwaway local database you do not intend to keep; anything whose schema
    matters should be built by ``alembic upgrade head``.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        migrated = await conn.run_sync(
            lambda sync_conn: sa_inspect(sync_conn).has_table("alembic_version")
        )
        if migrated:
            return

        if os.getenv("MAS_DB_ALLOW_CREATE_ALL") == "1":
            logger.warning(
                "MAS_DB_ALLOW_CREATE_ALL=1 — building the schema from ORM metadata "
                "instead of migrations. This database will have no migration "
                "provenance and must not be promoted to anything durable."
            )
            await conn.run_sync(Base.metadata.create_all)
            return

        raise RuntimeError(
            "Database has no alembic_version table — it has never been migrated. "
            "Run `alembic upgrade head` against it first. (For a throwaway local "
            "database, set MAS_DB_ALLOW_CREATE_ALL=1 to build from ORM metadata "
            "instead, accepting that it will have no migration provenance.)"
        )


async def close_db() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
