"""Add external engine rerun revision audit fields and table

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-02-19 10:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add mutable "current view" audit metadata on external_engine_results.
    op.add_column(
        "external_engine_results",
        sa.Column("ingest_revision", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "external_engine_results",
        sa.Column("source_run_timestamp", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "external_engine_results",
        sa.Column("source_payload_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "external_engine_results",
        sa.Column(
            "last_ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.alter_column("external_engine_results", "ingest_revision", server_default=None)
    op.alter_column("external_engine_results", "last_ingested_at", server_default=None)

    # Immutable rerun audit log.
    op.create_table(
        "external_engine_result_revisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("engine_result_id", sa.Integer(), nullable=False),
        sa.Column("engine_name", sa.String(length=30), nullable=False),
        sa.Column("run_date", sa.Date(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="success"),
        sa.Column("regime", sa.String(length=20), nullable=True),
        sa.Column("picks_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_run_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_payload_hash", sa.String(length=64), nullable=True),
        sa.Column("payload", JSONB(), nullable=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["engine_result_id"],
            ["external_engine_results.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "engine_name",
            "run_date",
            "revision",
            name="uq_engine_result_revision_name_date_rev",
        ),
    )
    op.create_index(
        "ix_external_engine_result_revisions_engine_result_id",
        "external_engine_result_revisions",
        ["engine_result_id"],
        unique=False,
    )
    op.create_index(
        "ix_external_engine_result_revisions_engine_name",
        "external_engine_result_revisions",
        ["engine_name"],
        unique=False,
    )
    op.create_index(
        "ix_external_engine_result_revisions_run_date",
        "external_engine_result_revisions",
        ["run_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_external_engine_result_revisions_run_date", table_name="external_engine_result_revisions")
    op.drop_index("ix_external_engine_result_revisions_engine_name", table_name="external_engine_result_revisions")
    op.drop_index(
        "ix_external_engine_result_revisions_engine_result_id",
        table_name="external_engine_result_revisions",
    )
    op.drop_table("external_engine_result_revisions")

    op.drop_column("external_engine_results", "last_ingested_at")
    op.drop_column("external_engine_results", "source_payload_hash")
    op.drop_column("external_engine_results", "source_run_timestamp")
    op.drop_column("external_engine_results", "ingest_revision")
