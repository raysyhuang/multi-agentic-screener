"""Add telegram_log table.

Revision ID: 472c87280115
Revises: f3a1b2c4d5e6
Create Date: 2026-03-02
"""

from alembic import op
import sqlalchemy as sa

revision = "472c87280115"
down_revision = "f3a1b2c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "telegram_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column("message_text", sa.Text(), nullable=False),
        sa.Column(
            "sent_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column("chat_id", sa.String(30), nullable=True),
        sa.Column("message_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_telegram_log_source", "telegram_log", ["source"])
    op.create_index("ix_telegram_log_sent_at", "telegram_log", ["sent_at"])


def downgrade() -> None:
    op.drop_index("ix_telegram_log_sent_at", table_name="telegram_log")
    op.drop_index("ix_telegram_log_source", table_name="telegram_log")
    op.drop_table("telegram_log")
