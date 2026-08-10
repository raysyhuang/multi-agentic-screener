"""Add persisted multi-engine backtest reports table

Revision ID: a7b8c9d0e1f2
Revises: f3a1b2c4d5e6
Create Date: 2026-02-22 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, Sequence[str], None] = "f3a1b2c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "multi_engine_backtest_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("filename", sa.String(length=120), nullable=False),
        sa.Column("run_date", sa.Date(), nullable=True),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("trading_days", sa.Integer(), nullable=True),
        sa.Column("total_trades_all_tracks", sa.Integer(), nullable=True),
        sa.Column("engines", JSONB(), nullable=True),
        sa.Column("report", JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("filename"),
    )
    op.create_index(
        "ix_multi_engine_backtest_runs_filename",
        "multi_engine_backtest_runs",
        ["filename"],
        unique=True,
    )
    op.create_index(
        "ix_multi_engine_backtest_runs_run_date",
        "multi_engine_backtest_runs",
        ["run_date"],
        unique=False,
    )
    op.create_index(
        "ix_multi_engine_backtest_runs_start_date",
        "multi_engine_backtest_runs",
        ["start_date"],
        unique=False,
    )
    op.create_index(
        "ix_multi_engine_backtest_runs_end_date",
        "multi_engine_backtest_runs",
        ["end_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_multi_engine_backtest_runs_end_date", table_name="multi_engine_backtest_runs")
    op.drop_index("ix_multi_engine_backtest_runs_start_date", table_name="multi_engine_backtest_runs")
    op.drop_index("ix_multi_engine_backtest_runs_run_date", table_name="multi_engine_backtest_runs")
    op.drop_index("ix_multi_engine_backtest_runs_filename", table_name="multi_engine_backtest_runs")
    op.drop_table("multi_engine_backtest_runs")
