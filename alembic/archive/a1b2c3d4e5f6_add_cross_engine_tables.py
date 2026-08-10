"""Add cross-engine tables: external_engine_results, engine_pick_outcomes, cross_engine_synthesis

Revision ID: a1b2c3d4e5f6
Revises: f3a1b2c4d5e6
Create Date: 2026-02-17 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'f3a1b2c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'external_engine_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('engine_name', sa.String(30), nullable=False, index=True),
        sa.Column('run_date', sa.Date(), nullable=False, index=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='success'),
        sa.Column('regime', sa.String(20), nullable=True),
        sa.Column('picks_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('payload', JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('engine_name', 'run_date', name='uq_engine_result_name_date'),
    )

    op.create_table(
        'engine_pick_outcomes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('engine_name', sa.String(30), nullable=False, index=True),
        sa.Column('run_date', sa.Date(), nullable=False, index=True),
        sa.Column('ticker', sa.String(10), nullable=False, index=True),
        sa.Column('strategy', sa.String(30), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('stop_loss', sa.Float(), nullable=True),
        sa.Column('target_price', sa.Float(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('holding_period_days', sa.Integer(), nullable=False),
        sa.Column('outcome_resolved', sa.Boolean(), server_default='false'),
        sa.Column('actual_return_pct', sa.Float(), nullable=True),
        sa.Column('hit_target', sa.Boolean(), nullable=True),
        sa.Column('exit_reason', sa.String(20), nullable=True),
        sa.Column('days_held', sa.Integer(), nullable=True),
        sa.Column('max_favorable_pct', sa.Float(), nullable=True),
        sa.Column('max_adverse_pct', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('engine_name', 'run_date', 'ticker', name='uq_engine_pick_name_date_ticker'),
    )

    op.create_table(
        'cross_engine_synthesis',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('run_date', sa.Date(), nullable=False, index=True, unique=True),
        sa.Column('convergent_tickers', JSONB(), nullable=True),
        sa.Column('portfolio_recommendation', JSONB(), nullable=True),
        sa.Column('regime_consensus', sa.String(20), nullable=True),
        sa.Column('engines_reporting', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('executive_summary', sa.Text(), nullable=True),
        sa.Column('verifier_notes', JSONB(), nullable=True),
        sa.Column('credibility_weights', JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('cross_engine_synthesis')
    op.drop_table('engine_pick_outcomes')
    op.drop_table('external_engine_results')
