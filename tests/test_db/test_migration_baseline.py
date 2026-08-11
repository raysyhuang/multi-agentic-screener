"""Fast guard: every ORM table must be created by a migration on the version path.

The real proof that migrations reproduce the schema is the `migrations` CI job,
which runs `alembic upgrade head` plus `alembic check` against a real Postgres —
these migrations use JSONB and cannot run on SQLite, so there is no way to
assert it behaviourally inside the unit suite.

What this catches instead is the specific regression that created the problem in
the first place: a model gets added to `src/db/models.py` and no migration
creates its table. That went unnoticed for nine core tables (signals, outcomes,
daily_runs, candidates, agent_logs, pipeline_artifacts, divergence_events,
divergence_outcomes, near_misses) because nothing compared the two sides. It is
a source-level check by necessity, and it fails in one second rather than
waiting for a Postgres job.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.db.models import Base

VERSIONS_DIR = Path(__file__).resolve().parents[2] / "alembic" / "versions"
ARCHIVE_DIR = Path(__file__).resolve().parents[2] / "alembic" / "archive"


def _tables_created_on_version_path() -> set[str]:
    created: set[str] = set()
    for path in VERSIONS_DIR.glob("*.py"):
        created |= set(
            re.findall(r"create_table\(\s*['\"]([a-z_]+)['\"]", path.read_text())
        )
    return created


def test_every_orm_table_is_created_by_a_migration() -> None:
    missing = set(Base.metadata.tables) - _tables_created_on_version_path()
    assert not missing, (
        "these tables exist in the ORM but no migration creates them, so "
        f"`alembic upgrade head` on an empty database will not build them: {sorted(missing)}"
    )


def test_migrations_do_not_create_tables_the_orm_does_not_declare() -> None:
    """The reverse drift: a table created by migration but dropped from the ORM.

    Left unchecked this is how `cross_engine_synthesis` and
    `multi_engine_backtest_runs` outlived the 2026-07 cross-engine strip — still
    created on any fresh database, read by nothing.
    """
    orphans = _tables_created_on_version_path() - set(Base.metadata.tables)
    assert not orphans, (
        "migrations create tables the ORM no longer declares; drop them from the "
        f"baseline or restore the models: {sorted(orphans)}"
    )


def test_archived_migrations_are_off_the_version_path() -> None:
    """Archived history is kept for provenance but must never execute.

    Alembic only loads `version_locations` (alembic/versions). If an archived
    revision were moved back it would rejoin the chain and reintroduce the
    broken root.
    """
    assert ARCHIVE_DIR.is_dir(), "alembic/archive/ should retain the pre-squash history"
    assert list(ARCHIVE_DIR.glob("*.py")), "archive should not be empty"
    assert not (VERSIONS_DIR / "e9b5089b0025_add_position_daily_metrics_and_signal_.py").exists()


def test_the_pre_squash_revision_stays_on_the_version_path() -> None:
    """`1c2d3e4f5a6b` must remain reachable — production and the mirror hold it.

    Squashing removed every revision id those databases could be stamped with.
    If that id leaves the version path, `alembic upgrade head` fails with
    `Can't locate revision identified by '1c2d3e4f5a6b'` — and that command runs
    ahead of the morning pipeline, so the book would go dark rather than warn.

    Note the invariant is *reachable*, not *head*. An earlier version of this
    test asserted it must BE head, which was true when the bridge was the only
    revision above the baseline but would have blocked every migration written
    afterwards. Pinning an accident of the moment rather than the property that
    matters is its own kind of bug.
    """
    revisions: set[str] = set()
    for path in VERSIONS_DIR.glob("*.py"):
        text = path.read_text()
        if m := re.search(r"^revision:?\s*(?::\s*str\s*)?=\s*['\"]([^'\"]+)['\"]", text, re.M):
            revisions.add(m.group(1))

    assert "1c2d3e4f5a6b" in revisions, (
        "the pre-squash head left the version path; every existing database "
        f"would fail to upgrade. Revisions present: {sorted(revisions)}"
    )


def test_the_version_path_is_a_single_linear_chain() -> None:
    """Exactly one head. Two heads mean a database can be 'at head' twice over."""
    down_revisions: set[str] = set()
    revisions: set[str] = set()
    for path in VERSIONS_DIR.glob("*.py"):
        text = path.read_text()
        if m := re.search(r"^revision:?\s*(?::\s*str\s*)?=\s*['\"]([^'\"]+)['\"]", text, re.M):
            revisions.add(m.group(1))
        for d in re.findall(r"^down_revision.*=\s*['\"]([^'\"]+)['\"]", text, re.M):
            down_revisions.add(d)

    heads = revisions - down_revisions
    assert len(heads) == 1, f"expected one head, found {sorted(heads)}"


def test_version_path_has_exactly_one_root() -> None:
    """A single baseline with `down_revision = None`, and no other roots.

    Two roots mean two independent chains, which is how a database can report
    "at head" while missing half the schema.
    """
    roots = []
    for path in VERSIONS_DIR.glob("*.py"):
        text = path.read_text()
        rev = re.search(r"^revision:?\s*(?::\s*str\s*)?=\s*['\"]([^'\"]+)['\"]", text, re.M)
        if re.search(r"^down_revision.*=\s*None", text, re.M):
            roots.append(rev.group(1) if rev else path.name)
    assert roots == ["0001_baseline"], f"expected one root migration, found {roots}"
