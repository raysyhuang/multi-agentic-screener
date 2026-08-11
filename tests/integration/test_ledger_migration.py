"""A row written before the ledger migration must survive it as NULL.

`selection_stage_reached` and `selected` were first written NOT NULL DEFAULT
false, which would have rewritten every legacy candidate as "reached selection
and was not picked" — an observation nobody made. The first query against this
table would then have counted fabricated negatives as data.

Same family as the empty-vs-missing attestation and `{}`-vs-`None` benchmark
provenance: a falsey default standing in for absent data.

**This must run the actual upgrade sequence.** An earlier version of this file
inserted its row into a table CI had already migrated to head, which proves
nothing — a future migration adding a default or a backfill would pass it
unchanged. So the test provisions its own database, migrates it to the
pre-ledger revision, writes a row as pre-ledger code would have, and only then
upgrades.

Alembic runs as a subprocess: `env.py` resolves the URL through the settings
singleton and drives its own event loop, so invoking it in-process from an async
test entangles both.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

REPO = Path(__file__).resolve().parents[2]
PRE_LEDGER_REVISION = "1c2d3e4f5a6b"
PROBE_DB = "mas_ledger_migration_probe"

# Every ledger column added by 0002. A legacy row must carry NULL in all of them.
LEDGER_COLUMNS = [
    "selection_stage_reached", "selected", "strategy_rank",
    "rejection_stage", "rejection_reason", "slots_total",
    "slots_occupied", "slots_available", "correlated_with", "correlation",
]


def _sync_url(url: str, database: str) -> str:
    """Point a URL at a different database, using the sync psycopg driver."""
    split = urlsplit(url.replace("postgresql+asyncpg://", "postgresql://"))
    return urlunsplit(split._replace(path=f"/{database}", query=""))


def _async_url(url: str, database: str) -> str:
    split = urlsplit(url.replace("postgresql://", "postgresql+asyncpg://"))
    return urlunsplit(split._replace(path=f"/{database}", query=""))


def _alembic(revision: str, database_url: str) -> None:
    result = subprocess.run(
        ["alembic", "upgrade", revision],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "DATABASE_URL": database_url},
    )
    assert result.returncode == 0, (
        f"alembic upgrade {revision} failed:\n{result.stdout}\n{result.stderr}"
    )


@pytest.fixture
async def probe_database():
    """An isolated database, dropped afterwards whatever happens."""
    base = os.environ.get("DATABASE_URL")
    if not base:
        pytest.skip("DATABASE_URL not set")

    admin = create_async_engine(_async_url(base, "postgres"), isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f'DROP DATABASE IF EXISTS "{PROBE_DB}" WITH (FORCE)'))
        await conn.execute(text(f'CREATE DATABASE "{PROBE_DB}"'))
    await admin.dispose()

    try:
        yield _sync_url(base, PROBE_DB)
    finally:
        admin = create_async_engine(
            _async_url(base, "postgres"), isolation_level="AUTOCOMMIT"
        )
        async with admin.connect() as conn:
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{PROBE_DB}" WITH (FORCE)'))
        await admin.dispose()


async def test_a_pre_ledger_row_survives_the_upgrade_as_null(probe_database) -> None:
    """Migrate to pre-ledger, write a row, upgrade, assert nothing was invented."""
    _alembic(PRE_LEDGER_REVISION, probe_database)

    engine = create_async_engine(_async_url(probe_database, PROBE_DB))
    try:
        # The ledger columns must not exist yet — otherwise the test is not
        # starting where it claims to.
        async with engine.connect() as conn:
            existing = {
                r[0] for r in (await conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'candidates'"
                ))).all()
            }
        assert not (set(LEDGER_COLUMNS) & existing), (
            f"ledger columns already present at {PRE_LEDGER_REVISION}: "
            f"{sorted(set(LEDGER_COLUMNS) & existing)}"
        )

        # A row exactly as pre-ledger code would have written it.
        async with engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO candidates
                    (run_date, ticker, close_price, avg_daily_volume,
                     composite_score, signal_model)
                VALUES (DATE '2020-01-02', 'LEGACY', 10.0, 1000000, 50.0, 'sniper')
            """))
    finally:
        await engine.dispose()

    _alembic("head", probe_database)

    engine = create_async_engine(_async_url(probe_database, PROBE_DB))
    try:
        async with engine.connect() as conn:
            row = (await conn.execute(text(
                f"SELECT {', '.join(LEDGER_COLUMNS)} FROM candidates "
                "WHERE ticker = 'LEGACY'"
            ))).one()
    finally:
        await engine.dispose()

    fabricated = {c: v for c, v in zip(LEDGER_COLUMNS, row) if v is not None}
    assert not fabricated, (
        "the migration invented observations for a row that predates the "
        f"ledger: {fabricated}. A default here would make every historical "
        "candidate read as 'reached selection and was not picked'."
    )


async def test_new_rows_can_still_record_an_explicit_negative(probe_database) -> None:
    """Nullable must not prevent recording a genuine observed False."""
    _alembic("head", probe_database)

    engine = create_async_engine(_async_url(probe_database, PROBE_DB))
    try:
        async with engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO candidates
                    (run_date, ticker, close_price, avg_daily_volume,
                     composite_score, signal_model,
                     selection_stage_reached, selected, rejection_stage,
                     rejection_reason, strategy_rank, slots_total,
                     slots_occupied, slots_available)
                VALUES (DATE '2026-08-11', 'OBSERVED', 10.0, 1000000, 50.0,
                        'sniper', true, false, 'capacity', 'capacity_censored',
                        3, 3, 3, 0)
            """))
        async with engine.connect() as conn:
            row = (await conn.execute(text(
                "SELECT selection_stage_reached, selected, rejection_reason, "
                "strategy_rank, slots_available FROM candidates "
                "WHERE ticker = 'OBSERVED'"
            ))).one()
    finally:
        await engine.dispose()

    assert row[0] is True and row[1] is False
    assert row[2] == "capacity_censored"
    assert (row[3], row[4]) == (3, 0)
