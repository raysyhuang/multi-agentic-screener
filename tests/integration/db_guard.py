"""Refuse to run integration tests against a database that is not local.

These tests are not read-only. `test_pipeline_smoke.py` executes the real
`run_morning_pipeline`, which writes `DailyRun`, `Signal`, `Outcome` and
`PipelineArtifact` rows for TODAY, and `_persist_daily_run` updates the
existing row when one is already there. Pointed at the production database it
would overwrite the real run record for the day and insert fake positions in
synthetic tickers ("AAAA").

Nothing stopped that. `pytest` reads the developer's `.env`, whose
`DATABASE_URL` is a remote managed Postgres, so `pytest -m integration` on a
laptop wrote to a production-shaped database by default. CI supplies
`postgresql://postgres:postgres@localhost:5432/mas_ci`, so requiring a local
host costs CI nothing and closes the laptop case entirely.

Deliberately no environment opt-out: an escape hatch is set once and then
inherited forever. Running these tests against a remote database has to be a
reviewed edit to this file.
"""

from __future__ import annotations

from urllib.parse import urlsplit

# Loopback only. A container/service database is reached over a published port
# on localhost (this is how the CI `services:` Postgres is addressed).
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def dsn_rejection_reason(url: str) -> str | None:
    """Return why `url` must not be written to, or None if it is safe.

    Safe means: a PostgreSQL DSN whose host is loopback. Integration tests need
    real Postgres (the models use JSONB), so a missing or non-Postgres URL is a
    misconfiguration rather than a safe fallback, and is also rejected.
    """
    if not url:
        return "DATABASE_URL is empty; integration tests need a local PostgreSQL database"
    parts = urlsplit(url)
    if not parts.scheme.startswith("postgres"):
        return f"DATABASE_URL scheme is {parts.scheme!r}; integration tests need PostgreSQL"
    host = (parts.hostname or "").lower()
    if host not in LOCAL_HOSTS:
        return (
            f"DATABASE_URL points at {host!r}, which is not a local database. "
            "These tests WRITE (they run the real morning pipeline and persist "
            "today's run). Point DATABASE_URL at a local PostgreSQL instance, "
            "e.g. postgresql://postgres:postgres@localhost:5432/mas_ci"
        )
    return None


def is_local_dsn(url: str) -> bool:
    """True when `url` is a database these tests are allowed to write to."""
    return dsn_rejection_reason(url) is None
