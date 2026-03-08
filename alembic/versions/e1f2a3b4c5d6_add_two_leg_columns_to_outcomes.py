"""Add two-leg trade columns to outcomes table (V1.2)

Revision ID: e1f2a3b4c5d6
Revises: a1b2c3d4e5f7
Create Date: 2026-03-08 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "e1f2a3b4c5d6"
down_revision = "a1b2c3d4e5f7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("outcomes", sa.Column("partial_exit_price", sa.Float(), nullable=True))
    op.add_column("outcomes", sa.Column("partial_exit_date", sa.Date(), nullable=True))
    op.add_column("outcomes", sa.Column("leg2_exit_reason", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("outcomes", "leg2_exit_reason")
    op.drop_column("outcomes", "partial_exit_date")
    op.drop_column("outcomes", "partial_exit_price")
