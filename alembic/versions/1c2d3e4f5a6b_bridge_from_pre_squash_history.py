"""bridge from the pre-squash history — no schema change

Revision ID: 1c2d3e4f5a6b
Revises: 0001_baseline
Create Date: 2026-08-10

This revision does nothing to the schema. It exists so that databases created
before the squash keep working without anyone having to intervene.

The problem it solves
---------------------
Squashing the sixteen historical revisions into `0001_baseline` removed every
revision id those databases might be stamped with. Production and the VPS paper
mirror both hold `1c2d3e4f5a6b` — the old head — in `alembic_version`. With that
id absent from the version path, the very next `alembic upgrade head` fails:

    CommandError: Can't locate revision identified by '1c2d3e4f5a6b'

That step runs before the morning pipeline in
`.github/workflows/scheduled-pipelines.yml`, so the squash alone would have
taken the book dark at 06:00 ET the following weekday.

Reusing the old head's id as a no-op successor to the baseline makes both paths
land in the same place with no manual step:

  * an existing database is already at `1c2d3e4f5a6b`, which is now head, so
    `upgrade head` has nothing to do;
  * a new database runs `0001_baseline` and then this no-op, ending at the same
    head with the same schema.

Why not `alembic stamp` instead
-------------------------------
A stamp would work, but it has to be executed by hand against every existing
database in the window between merging and the next scheduled run — and if it is
forgotten, the failure mode is a production outage rather than a warning. It
would also overwrite the one durable record of where those databases came from.
Keeping the old head's id means `alembic_version` still says, truthfully, that
this database descends from the pre-squash chain and was not built from the
baseline. See `alembic/archive/README.md`.

Databases stamped at some *intermediate* archived revision rather than the old
head are not covered here; none are known to exist. Such a database would fail
the same way and should be checked against the baseline and stamped
deliberately.
"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "1c2d3e4f5a6b"
down_revision: Union[str, Sequence[str], None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op: `0001_baseline` already created everything this schema has."""


def downgrade() -> None:
    """No-op: nothing was applied."""
