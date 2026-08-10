"""Add dataset_health to daily_runs and cross_engine_health to cross_engine_synthesis

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-17 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('daily_runs', sa.Column('dataset_health', JSONB(), nullable=True))
    op.add_column('cross_engine_synthesis', sa.Column('cross_engine_health', JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column('cross_engine_synthesis', 'cross_engine_health')
    op.drop_column('daily_runs', 'dataset_health')
