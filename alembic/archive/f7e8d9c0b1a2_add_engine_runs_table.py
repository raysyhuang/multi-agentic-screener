"""Add engine_runs table

Revision ID: f7e8d9c0b1a2
Revises: 472c87280115
Create Date: 2026-03-06 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "f7e8d9c0b1a2"
down_revision = "472c87280115"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "engine_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("engine_name", sa.String(30), nullable=False),
        sa.Column("run_date", sa.Date(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("fetch_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fetch_finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fetch_duration_ms", sa.Integer(), nullable=True),
        sa.Column("picks_count", sa.Integer(), nullable=True),
        sa.Column("candidates_screened", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("payload_hash", sa.String(64), nullable=True),
        sa.Column("engine_result_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["engine_result_id"], ["external_engine_results.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "engine_name", "run_date", "attempt", name="uq_engine_run_name_date_attempt"
        ),
    )
    op.create_index(op.f("ix_engine_runs_engine_name"), "engine_runs", ["engine_name"])
    op.create_index(op.f("ix_engine_runs_run_date"), "engine_runs", ["run_date"])


def downgrade() -> None:
    op.drop_index(op.f("ix_engine_runs_run_date"), table_name="engine_runs")
    op.drop_index(op.f("ix_engine_runs_engine_name"), table_name="engine_runs")
    op.drop_table("engine_runs")
