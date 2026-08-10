"""Add execution realism columns: signals.max_entry_price, outcomes.skip_reason

Revision ID: f1a2b3c4d5e6
Revises: e1f2a3b4c5d6
Create Date: 2026-03-19 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "f1a2b3c4d5e6"
down_revision = "e1f2a3b4c5d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("signals", sa.Column("max_entry_price", sa.Float(), nullable=True))
    op.add_column("outcomes", sa.Column("skip_reason", sa.String(30), nullable=True))


def downgrade() -> None:
    op.drop_column("outcomes", "skip_reason")
    op.drop_column("signals", "max_entry_price")
