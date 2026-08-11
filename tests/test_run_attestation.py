"""The run must publish an identity an outside observer can verify against.

A health check running outside GitHub — the VPS mirror — cannot reach Neon, and
the published dashboard snapshot exports `DailyRun` health but not per-run
`PipelineArtifact(stage="governance")`. So it can only infer health from the
workflow conclusion, which has now been green on a broken run twice: a DST-guard
skip, and a fail-closed NoTrade.

The attestation closes that: the workflow checks the exact run against the
database, where it IS reachable, and publishes the verdict as a job output. This
covers the half that lives in the pipeline — writing the run id somewhere the
workflow can find it, early enough that a crash still identifies itself.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src import main as main_mod


@pytest.mark.asyncio
async def test_the_run_id_is_published_when_asked(monkeypatch, tmp_path) -> None:
    target = tmp_path / "run_id"
    monkeypatch.setenv("MAS_RUN_ID_FILE", str(target))

    async def fine(*a, **k):
        return None

    monkeypatch.setattr(main_mod, "_run_pipeline_core", fine)

    await main_mod.run_morning_pipeline()

    published = target.read_text().strip()
    assert len(published) == 12, f"expected a 12-char run id, got {published!r}"


@pytest.mark.asyncio
async def test_a_crashed_run_still_publishes_its_identity(
    monkeypatch, tmp_path
) -> None:
    """The case the attestation exists for.

    A run that vanished and a run that recorded its own failure look identical
    from outside unless the failed one still says who it was.
    """
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock

    target = tmp_path / "run_id"
    monkeypatch.setenv("MAS_RUN_ID_FILE", str(target))

    async def boom(*a, **k):
        raise RuntimeError("induced")

    @asynccontextmanager
    async def fake_session():
        class _S:
            async def execute(self, *a, **k):
                r = MagicMock()
                r.scalar_one_or_none.return_value = None
                r.scalars.return_value.all.return_value = []
                return r

            def add(self, *a, **k):
                pass

            async def commit(self):
                pass

            async def rollback(self):
                pass

        yield _S()

    monkeypatch.setattr(main_mod, "_run_pipeline_core", boom)
    monkeypatch.setattr(main_mod, "get_session", fake_session)
    monkeypatch.setattr(main_mod, "send_alert", AsyncMock(return_value=False))

    assert await main_mod.run_morning_pipeline() is False
    assert target.read_text().strip(), "a crashed run must still name itself"


def test_publishing_is_optional(monkeypatch) -> None:
    """No env var, no file, no error — local and Heroku runs are unaffected."""
    monkeypatch.delenv("MAS_RUN_ID_FILE", raising=False)
    main_mod._publish_run_id("abc123")  # must not raise


def test_an_unwritable_path_never_fails_the_run(monkeypatch, tmp_path) -> None:
    """Diagnostics must not be able to take down what they observe.

    That inversion is exactly what the 2026-08-11 outage was — a provenance
    read, added for visibility, becoming the reason the book went to NoTrade.
    """
    monkeypatch.setenv("MAS_RUN_ID_FILE", str(tmp_path / "no" / "such" / "dir" / "id"))
    main_mod._publish_run_id("abc123")  # must not raise


def test_the_attestation_script_exists_and_is_runnable() -> None:
    """The workflow references it by path; a rename would break the gate."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "assert_run_attestation.py"
    assert script.exists(), "workflow step 'Attest the run recorded itself' needs this"

    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github" / "workflows" / "scheduled-pipelines.yml"
    ).read_text()
    assert "assert_run_attestation.py" in workflow
    assert "MAS_RUN_ID_FILE" in workflow, "the worker step must publish the id"
    for output in ("run_id:", "attested:", "governance_status:"):
        assert output in workflow, f"job output {output!r} missing — the mirror reads it"
