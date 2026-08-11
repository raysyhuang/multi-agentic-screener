"""selection ledger on candidates

Revision ID: 0002_selection_ledger
Revises: 1c2d3e4f5a6b
Create Date: 2026-08-12

Records why each candidate was or was not selected, at the moment of the
decision. The sniper concurrency cap was previously a bare `continue`, so a
candidate dropped because the book was full looked identical to one ranked
below the daily quota — making "how often does capacity censor an otherwise
valid signal?" unanswerable from stored data.

All columns are nullable or carry a server default, so existing rows are valid
without backfill. They describe decisions that were never recorded and cannot
be reconstructed; leaving them NULL is the truthful representation.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002_selection_ledger"
down_revision: Union[str, Sequence[str], None] = "1c2d3e4f5a6b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Nullable, no server default: legacy rows predate the ledger and nothing
    # was observed about them. NOT NULL DEFAULT false would fabricate a fact —
    # every historical candidate would read as "reached selection and was not
    # picked", which is a measurement we never made.
    op.add_column("candidates", sa.Column(
        "selection_stage_reached", sa.Boolean(), nullable=True,
    ))
    op.add_column("candidates", sa.Column("strategy_rank", sa.Integer(), nullable=True))
    op.add_column("candidates", sa.Column("selected", sa.Boolean(), nullable=True))
    op.add_column("candidates", sa.Column("rejection_stage", sa.String(length=20), nullable=True))
    op.add_column("candidates", sa.Column("rejection_reason", sa.String(length=40), nullable=True))
    op.add_column("candidates", sa.Column("slots_total", sa.Integer(), nullable=True))
    op.add_column("candidates", sa.Column("slots_occupied", sa.Integer(), nullable=True))
    op.add_column("candidates", sa.Column("slots_available", sa.Integer(), nullable=True))
    op.add_column("candidates", sa.Column("correlated_with", sa.String(length=10), nullable=True))
    op.add_column("candidates", sa.Column("correlation", sa.Float(), nullable=True))


def downgrade() -> None:
    for col in (
        "correlation", "correlated_with", "slots_available", "slots_occupied", "slots_total",
        "rejection_reason", "rejection_stage", "selected", "strategy_rank",
        "selection_stage_reached",
    ):
        op.drop_column("candidates", col)
