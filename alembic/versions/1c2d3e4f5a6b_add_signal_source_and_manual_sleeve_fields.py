"""Add signal source metadata and manual sleeve execution fields.

Revision ID: 1c2d3e4f5a6b
Revises: f1a2b3c4d5e6
Create Date: 2026-04-26 15:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "1c2d3e4f5a6b"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "signals",
        sa.Column(
            "signal_source",
            sa.String(length=30),
            nullable=False,
            server_default="mas_official",
        ),
    )
    op.add_column(
        "signals",
        sa.Column(
            "also_in_mas",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "signals",
        sa.Column(
            "suppressed_by_cross_model_ranking",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index(
        op.f("ix_signals_signal_source"),
        "signals",
        ["signal_source"],
        unique=False,
    )

    op.add_column("outcomes", sa.Column("manual_status", sa.String(length=20), nullable=True))
    op.add_column("outcomes", sa.Column("manual_entry_date", sa.Date(), nullable=True))
    op.add_column("outcomes", sa.Column("manual_entry_price", sa.Float(), nullable=True))
    op.add_column("outcomes", sa.Column("manual_exit_date", sa.Date(), nullable=True))
    op.add_column("outcomes", sa.Column("manual_exit_price", sa.Float(), nullable=True))
    op.add_column("outcomes", sa.Column("manual_exit_reason", sa.String(length=20), nullable=True))
    op.add_column("outcomes", sa.Column("manual_pnl_pct", sa.Float(), nullable=True))
    op.add_column("outcomes", sa.Column("manual_pnl_dollars", sa.Float(), nullable=True))
    op.add_column("outcomes", sa.Column("entry_slippage_pct", sa.Float(), nullable=True))
    op.add_column("outcomes", sa.Column("exit_slippage_pct", sa.Float(), nullable=True))
    op.add_column("outcomes", sa.Column("manual_notes", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("outcomes", "manual_notes")
    op.drop_column("outcomes", "exit_slippage_pct")
    op.drop_column("outcomes", "entry_slippage_pct")
    op.drop_column("outcomes", "manual_pnl_dollars")
    op.drop_column("outcomes", "manual_pnl_pct")
    op.drop_column("outcomes", "manual_exit_reason")
    op.drop_column("outcomes", "manual_exit_price")
    op.drop_column("outcomes", "manual_exit_date")
    op.drop_column("outcomes", "manual_entry_price")
    op.drop_column("outcomes", "manual_entry_date")
    op.drop_column("outcomes", "manual_status")

    op.drop_index(op.f("ix_signals_signal_source"), table_name="signals")
    op.drop_column("signals", "suppressed_by_cross_model_ranking")
    op.drop_column("signals", "also_in_mas")
    op.drop_column("signals", "signal_source")
