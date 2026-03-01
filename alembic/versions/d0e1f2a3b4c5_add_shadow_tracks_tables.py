"""Add shadow tracks tables for parallel parameter experiments

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-03-01 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "d0e1f2a3b4c5"
down_revision: Union[str, Sequence[str], None] = "c9d0e1f2a3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- shadow_tracks ---
    op.create_table(
        "shadow_tracks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(80), unique=True, nullable=False, index=True),
        sa.Column("generation", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("parent_track", sa.String(80), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("config", JSONB(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- shadow_track_picks ---
    op.create_table(
        "shadow_track_picks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("track_id", sa.Integer(), sa.ForeignKey("shadow_tracks.id"), nullable=False, index=True),
        sa.Column("run_date", sa.Date(), nullable=False, index=True),
        sa.Column("ticker", sa.String(10), nullable=False, index=True),
        sa.Column("direction", sa.String(10), nullable=False),
        sa.Column("strategy", sa.String(30), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("stop_loss", sa.Float(), nullable=True),
        sa.Column("target_price", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("holding_period", sa.Integer(), nullable=False),
        sa.Column("weight_pct", sa.Float(), nullable=True),
        sa.Column("source_engines", sa.String(100), nullable=True),
        sa.Column("outcome_resolved", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("actual_return", sa.Float(), nullable=True),
        sa.Column("exit_reason", sa.String(20), nullable=True),
        sa.Column("exit_date", sa.Date(), nullable=True),
        sa.Column("days_held", sa.Integer(), nullable=True),
        sa.Column("max_favorable", sa.Float(), nullable=True),
        sa.Column("max_adverse", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("track_id", "run_date", "ticker", name="uq_shadow_pick_track_date_ticker"),
    )

    # --- shadow_track_snapshots ---
    op.create_table(
        "shadow_track_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("track_id", sa.Integer(), sa.ForeignKey("shadow_tracks.id"), nullable=False, index=True),
        sa.Column("snapshot_date", sa.Date(), nullable=False, index=True),
        sa.Column("total_picks", sa.Integer(), server_default="0"),
        sa.Column("resolved_picks", sa.Integer(), server_default="0"),
        sa.Column("win_rate", sa.Float(), nullable=True),
        sa.Column("avg_return_pct", sa.Float(), nullable=True),
        sa.Column("total_return", sa.Float(), nullable=True),
        sa.Column("sharpe_ratio", sa.Float(), nullable=True),
        sa.Column("profit_factor", sa.Float(), nullable=True),
        sa.Column("max_drawdown", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("track_id", "snapshot_date", name="uq_shadow_snapshot_track_date"),
    )


def downgrade() -> None:
    op.drop_table("shadow_track_snapshots")
    op.drop_table("shadow_track_picks")
    op.drop_table("shadow_tracks")
