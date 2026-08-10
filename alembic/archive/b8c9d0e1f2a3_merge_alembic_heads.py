"""Merge Alembic heads after backtest persistence branch

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2, d4e5f6a7b8c9
Create Date: 2026-02-22 14:10:00.000000

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "b8c9d0e1f2a3"
down_revision: Union[str, Sequence[str], None] = ("a7b8c9d0e1f2", "d4e5f6a7b8c9")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op merge revision."""
    pass


def downgrade() -> None:
    """No-op merge revision."""
    pass
