"""Add score_velocity to position_daily_metrics

Revision ID: f3a1b2c4d5e6
Revises: e9b5089b0025
Create Date: 2026-02-17 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'f3a1b2c4d5e6'
down_revision: Union[str, Sequence[str], None] = 'e9b5089b0025'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add score_velocity column."""
    op.add_column(
        'position_daily_metrics',
        sa.Column('score_velocity', sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Remove score_velocity column."""
    op.drop_column('position_daily_metrics', 'score_velocity')
