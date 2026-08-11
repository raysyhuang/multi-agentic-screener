"""A pre-existing candidate row must survive the ledger migration as NULL.

`selection_stage_reached` and `selected` were first written NOT NULL DEFAULT
false. That would have rewritten every legacy candidate as "reached selection
and was not picked" — a measurement nobody ever made. Unknown has to stay
unknown, or the first query against this table silently counts fabricated
observations.

Same family as the empty-vs-missing attestation and the `{}`-vs-`None`
benchmark provenance: a falsey default standing in for absent data.

Requires Postgres, so it runs in the migrations CI job.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import text

from src.db.session import get_session

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_a_legacy_row_keeps_null_ledger_fields() -> None:
    """Insert a row without ledger fields, as pre-migration code would have."""
    async with get_session() as session:
        await session.execute(text("""
            INSERT INTO candidates
                (run_date, ticker, close_price, avg_daily_volume,
                 composite_score, signal_model)
            VALUES (:d, 'LEGACY', 10.0, 1000000, 50.0, 'sniper')
        """), {"d": date(2020, 1, 2)})

    async with get_session() as session:
        row = (await session.execute(text("""
            SELECT selection_stage_reached, selected, strategy_rank,
                   rejection_stage, rejection_reason, slots_total, correlated_with
            FROM candidates WHERE ticker = 'LEGACY'
        """))).one()

    assert all(v is None for v in row), (
        "legacy rows must stay NULL — a false default would fabricate the "
        f"observation that they reached selection and lost. Got: {row}"
    )

    async with get_session() as session:
        await session.execute(text("DELETE FROM candidates WHERE ticker = 'LEGACY'"))


async def test_new_rows_can_still_write_explicit_false() -> None:
    """Nullable must not prevent recording a genuine negative observation."""
    async with get_session() as session:
        await session.execute(text("""
            INSERT INTO candidates
                (run_date, ticker, close_price, avg_daily_volume,
                 composite_score, signal_model,
                 selection_stage_reached, selected, rejection_stage,
                 rejection_reason, strategy_rank, slots_total,
                 slots_occupied, slots_available, correlated_with, correlation)
            VALUES (:d, 'OBSERVED', 10.0, 1000000, 50.0, 'sniper',
                    true, false, 'capacity', 'capacity_censored', 3, 3, 3, 0,
                    NULL, NULL)
        """), {"d": date(2026, 8, 12)})

    async with get_session() as session:
        row = (await session.execute(text("""
            SELECT selection_stage_reached, selected, rejection_reason,
                   strategy_rank, slots_available
            FROM candidates WHERE ticker = 'OBSERVED'
        """))).one()

    assert row[0] is True and row[1] is False
    assert row[2] == "capacity_censored"
    assert row[3] == 3 and row[4] == 0

    async with get_session() as session:
        await session.execute(text("DELETE FROM candidates WHERE ticker = 'OBSERVED'"))
