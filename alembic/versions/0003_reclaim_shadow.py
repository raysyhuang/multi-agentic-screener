"""reclaim shadow lanes

Revision ID: 0003_reclaim_shadow
Revises: 0002_selection_ledger
Create Date: 2026-10-04

Ten append-only tables for the RECLAIM native shadow lanes
(RECLAIM_MAS_SPEC_v1.1 §8.6). Nothing in the official pipeline reads them.

Every table carries a unique key and every insert is ON CONFLICT DO NOTHING,
so the earliest row wins and a same-day rerun writes nothing. `Signal` has no
uniqueness constraint, which is why `reclaim_clones` exists: its key is
inserted in the same transaction as the clone's Signal, and an existing key
means no Signal or Outcome is created.

Purely additive — no existing table or column changes, so the downgrade drops
only what this revision created.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0003_reclaim_shadow"
down_revision: Union[str, Sequence[str], None] = "0002_selection_ledger"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = (
    "reclaim_lane_registry",
    "reclaim_pool_runs",
    "reclaim_pool_members",
    "reclaim_episodes",
    "reclaim_events",
    "reclaim_triggers",
    "reclaim_risk_snapshots",
    "reclaim_earnings_snapshots",
    "reclaim_open_decisions",
    "reclaim_clones",
)


def upgrade() -> None:
    op.create_table(
        'reclaim_lane_registry',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('lane', sa.String(length=30), nullable=False),
        sa.Column('forward_start_date', sa.Date(), nullable=False),
        sa.Column('spec_sha', sa.String(length=64), nullable=True),
        sa.Column('code_sha', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('lane', name='uq_reclaim_lane_registry_lane'),
    )
    op.create_table(
        'reclaim_pool_runs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('lane', sa.String(length=30), nullable=False),
        sa.Column('snapshot_date', sa.Date(), nullable=False),
        sa.Column('status', sa.String(length=10), nullable=False),
        sa.Column('universe_count', sa.Integer(), nullable=True),
        sa.Column('members_count', sa.Integer(), nullable=True),
        sa.Column('ohlcv_cap_hits', sa.Integer(), nullable=True),
        sa.Column('dollar_volume_basis', sa.String(length=40), nullable=True),
        sa.Column('identity_method', sa.String(length=30), nullable=True),
        sa.Column('provenance', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('lane', 'snapshot_date', name='uq_reclaim_pool_run_lane_date'),
    )
    op.create_table(
        'reclaim_pool_members',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('lane', sa.String(length=30), nullable=False),
        sa.Column('snapshot_date', sa.Date(), nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('score_pct', sa.Float(), nullable=True),
        sa.Column('box_low', sa.Float(), nullable=True),
        sa.Column('box_high', sa.Float(), nullable=True),
        sa.Column('atr14', sa.Float(), nullable=True),
        sa.Column('mcap', sa.Float(), nullable=True),
        sa.Column('sector', sa.String(length=60), nullable=True),
        sa.Column('gate_inputs', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('lane', 'snapshot_date', 'symbol', name='uq_reclaim_pool_member_lane_date_symbol'),
    )
    op.create_table(
        'reclaim_episodes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('episode_id', sa.String(length=80), nullable=False),
        sa.Column('lane', sa.String(length=30), nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('start_date', sa.Date(), nullable=False),
        sa.Column('floor', sa.Float(), nullable=True),
        sa.Column('floor_type', sa.String(length=40), nullable=False),
        sa.Column('prehistory', sa.String(length=30), nullable=False),
        sa.Column('prehistory_reason', sa.String(length=40), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('episode_id', name='uq_reclaim_episode_id'),
    )
    op.create_table(
        'reclaim_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('episode_id', sa.String(length=80), nullable=False),
        sa.Column('event_date', sa.Date(), nullable=False),
        sa.Column('event', sa.String(length=30), nullable=False),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('episode_id', 'event_date', 'event', name='uq_reclaim_event'),
    )
    op.create_table(
        'reclaim_triggers',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('episode_id', sa.String(length=80), nullable=False),
        sa.Column('trigger_date', sa.Date(), nullable=False),
        sa.Column('lane', sa.String(length=30), nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('entry_session', sa.Date(), nullable=False),
        sa.Column('reclaim_date', sa.Date(), nullable=False),
        sa.Column('retest_date', sa.Date(), nullable=False),
        sa.Column('setup_low', sa.Float(), nullable=False),
        sa.Column('high_retest', sa.Float(), nullable=False),
        sa.Column('close_k', sa.Float(), nullable=False),
        sa.Column('sma50_k', sa.Float(), nullable=False),
        sa.Column('slope5', sa.Float(), nullable=True),
        sa.Column('floor', sa.Float(), nullable=True),
        sa.Column('prehistory', sa.String(length=30), nullable=False),
        sa.Column('prehistory_reason', sa.String(length=40), nullable=True),
        sa.Column('overlap_desc', sa.Boolean(), nullable=False),
        sa.Column('score_pct', sa.Float(), nullable=True),
        sa.Column('sector', sa.String(length=60), nullable=True),
        sa.Column('mcap', sa.Float(), nullable=True),
        sa.Column('mas_regime', sa.String(length=20), nullable=True),
        sa.Column('spy_regime', sa.String(length=20), nullable=True),
        sa.Column('detected_on', sa.Date(), nullable=False),
        sa.Column('forward_recorded', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('episode_id', 'trigger_date', name='uq_reclaim_trigger'),
    )
    op.create_table(
        'reclaim_risk_snapshots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('trigger_episode_id', sa.String(length=80), nullable=False),
        sa.Column('trigger_date', sa.Date(), nullable=False),
        sa.Column('control_episode_id', sa.String(length=80), nullable=False),
        sa.Column('control_symbol', sa.String(length=10), nullable=False),
        sa.Column('sector', sa.String(length=60), nullable=True),
        sa.Column('mcap', sa.Float(), nullable=True),
        sa.Column('score_pct', sa.Float(), nullable=True),
        sa.Column('sma_dist', sa.Float(), nullable=True),
        sa.Column('age', sa.Integer(), nullable=True),
        sa.Column('prehistory', sa.String(length=30), nullable=False),
        sa.Column('retest_pending', sa.Boolean(), nullable=False),
        sa.Column('confirm_pending', sa.Boolean(), nullable=False),
        sa.Column('triggered_on_k', sa.Boolean(), nullable=False),
        sa.Column('has_entry_bar', sa.Boolean(), nullable=True),
        sa.Column('control_open_e', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('trigger_episode_id', 'trigger_date', 'control_episode_id', name='uq_reclaim_risk_snapshot'),
    )
    op.create_table(
        'reclaim_earnings_snapshots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('trigger_date', sa.Date(), nullable=False),
        sa.Column('captured_at_utc', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ok', sa.Boolean(), nullable=False),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('coverage', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'trigger_date', name='uq_reclaim_earnings_snapshot'),
    )
    op.create_table(
        'reclaim_open_decisions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('episode_id', sa.String(length=80), nullable=False),
        sa.Column('trigger_date', sa.Date(), nullable=False),
        sa.Column('open_e', sa.Float(), nullable=True),
        sa.Column('stop', sa.Float(), nullable=False),
        sa.Column('sma50_k', sa.Float(), nullable=False),
        sa.Column('ext_ratio', sa.Float(), nullable=True),
        sa.Column('risk_pct', sa.Float(), nullable=True),
        sa.Column('reasons', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('mech_status', sa.String(length=30), nullable=False),
        sa.Column('earnings_status', sa.String(length=30), nullable=False),
        sa.Column('decided_at_utc', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('episode_id', 'trigger_date', name='uq_reclaim_open_decision'),
    )
    op.create_table(
        'reclaim_clones',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('episode_id', sa.String(length=80), nullable=False),
        sa.Column('trigger_date', sa.Date(), nullable=False),
        sa.Column('horizon', sa.Integer(), nullable=False),
        sa.Column('signal_id', sa.Integer(), nullable=True),
        sa.Column('outcome_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['signal_id'], ['signals.id']),
        sa.ForeignKeyConstraint(['outcome_id'], ['outcomes.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('episode_id', 'trigger_date', 'horizon', name='uq_reclaim_clone'),
    )


def downgrade() -> None:
    for table in reversed(_TABLES):
        op.drop_table(table)
