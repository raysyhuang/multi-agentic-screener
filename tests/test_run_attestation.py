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


def test_an_externally_minted_run_id_is_honoured(monkeypatch) -> None:
    """Identity must be ownable before anything fallible starts.

    init_db(), get_settings() and validate_keys_for_mode() all run before the
    pipeline, so an id minted inside it does not exist for a run that dies in
    startup — and the attestation would have nothing to query. The workflow
    mints it first and passes it in.
    """
    import asyncio

    monkeypatch.setenv("MAS_RUN_ID", "deadbeefcafe")
    captured = {}

    async def capture(today, settings, run_id, start_time, _state=None):
        captured["run_id"] = run_id

    monkeypatch.setattr(main_mod, "_run_pipeline_core", capture)
    asyncio.get_event_loop_policy()  # keep pytest-asyncio's loop handling happy
    asyncio.run(main_mod.run_morning_pipeline())

    assert captured["run_id"] == "deadbeefcafe"


def test_the_workflow_wires_identity_attestation_and_upload() -> None:
    """The gate depends on all three; any one missing silently breaks it."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "assert_run_attestation.py").exists()

    workflow = (
        root / ".github" / "workflows" / "scheduled-pipelines.yml"
    ).read_text()

    # Identity minted before the fallible steps.
    assert "Mint run identity" in workflow
    minted = workflow.index("Mint run identity")
    for later in ("Apply database migrations", "Run ${{ steps.resolve.outputs.pipeline }} pipeline"):
        assert workflow.index(later) > minted, (
            f"'{later}' runs before identity is minted; a failure there would "
            "leave nothing to attest against"
        )

    assert "assert_run_attestation.py" in workflow
    assert "--run-id " in workflow, "the minted id must be passed in"

    # The externally readable half. Job outputs are NOT exposed by the REST API,
    # so without the artifact the VPS mirror cannot read the attestation at all.
    assert "actions/upload-artifact" in workflow
    assert "mas-run-attestation" in workflow, (
        "the artifact name is the mirror's lookup key"
    )
    assert "if-no-files-found: error" in workflow, (
        "a missing attestation must never upload green — with `warn` a crashed "
        "script leaves no artifact and the mirror sees a successful upload"
    )
    assert "AttestationScriptFailed" in workflow, (
        "the step must write a canonical unhealthy fallback if the script "
        "cannot run at all"
    )


def test_a_database_error_never_leaks_its_message(monkeypatch, tmp_path) -> None:
    """Only the exception class reaches the artifact.

    Connection and config failures routinely carry the DSN — host, database,
    username, sometimes the password — and this file is downloadable by anyone
    with repo read access.
    """
    import asyncio
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "attest",
        Path(__file__).resolve().parents[1] / "scripts" / "assert_run_attestation.py",
    )
    attest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attest)

    secret = "postgresql://mas_user:hunter2@db.internal:5432/mas"

    class _Boom:
        async def __aenter__(self):
            raise RuntimeError(f"could not connect to {secret}")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(attest, "get_session", lambda: _Boom())

    out = tmp_path / "attestation.json"
    rc = asyncio.run(attest._check("run123", out))

    assert rc == 1
    payload = out.read_text()
    assert secret not in payload, "the DSN reached the downloadable artifact"
    assert "hunter2" not in payload
    record = json.loads(payload)
    assert record["db_error"] == "RuntimeError", record["db_error"]
    assert record["attested"] is False
