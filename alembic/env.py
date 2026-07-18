"""Alembic env — async-aware, reads DATABASE_URL from .env."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from src.db.models import Base
from src.db.session import _get_async_url, _needs_ssl

# Alembic Config object
config = context.config

# Set up Python logging from the ini file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Point autogenerate at our metadata
target_metadata = Base.metadata

# Legacy tables in the DB that are NOT managed by the ORM models.
# Autogenerate will ignore these instead of emitting DROP TABLE.
_LEGACY_TABLES = frozenset({
    "daily_market_data",
    "paper_trades",
    "tickers",
    "screener_signals",
    "reversion_signals",
})


def include_name(name, type_, parent_names):
    if type_ == "table" and name in _LEGACY_TABLES:
        return False
    return True


def _get_url() -> str:
    """Resolve the database URL, normalized for asyncpg.

    Delegates to ``src.db.session._get_async_url`` so alembic and the app share
    identical URL handling — asyncpg driver + stripping libpq-only query params
    (``sslmode``/``channel_binding``) that managed Postgres providers append and
    asyncpg rejects. SSL itself is negotiated via connect_args (see below).
    """
    return _get_async_url()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a live connection)."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_name=include_name,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with an async engine."""
    configuration = config.get_section(config.config_ini_section, {})
    url = _get_url()
    configuration["sqlalchemy.url"] = url

    # asyncpg ignores sslmode in the URL; force SSL for remote managed Postgres
    # (Neon/Supabase/Heroku), matching src.db.session.get_engine.
    connect_args = {"ssl": "require"} if _needs_ssl(url) else {}

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations — delegates to async runner."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
