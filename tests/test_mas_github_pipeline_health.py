"""Tests for the health gate — the file that decides whether the measurement
lane runs at all.

Until 2026-08-20 this lived only on one host. The brief gates the entire lane
on this script's stdout prefix and exit code, a coupling nothing guarded.
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mas_github_pipeline_health as gate  # noqa: E402

SHA = "a" * 40


def attestation(**overrides) -> dict:
    base = {
        "run_id": "mas-run-1", "attested": True, "healthy": True,
        "governance_status": "success", "final_output_status": "success",
        "artifact_stages": [], "db_error": None,
        "github_run_id": 12345, "github_run_attempt": 1, "commit": SHA,
    }
    base.update(overrides)
    return base


# ── configuration ────────────────────────────────────────────────────────

def test_missing_repo_variable_fails_closed(monkeypatch):
    monkeypatch.delenv("MAS_HEALTH_REPO", raising=False)
    with pytest.raises(gate.ConfigError, match="MAS_HEALTH_REPO"):
        gate.resolve_repo()


def test_nonexistent_repo_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("MAS_HEALTH_REPO", str(tmp_path / "absent"))
    with pytest.raises(gate.ConfigError, match="does not exist"):
        gate.resolve_repo()


def test_no_host_path_is_embedded():
    text = (REPO / "scripts" / "mas_github_pipeline_health.py").read_text()
    for leak in ("/srv/", ".hermes", "/home/agent"):
        assert leak not in text, f"{leak!r} must not appear in a public repo"


# ── the cross-file contract the brief depends on ─────────────────────────

def test_healthy_prefix_is_exactly_what_the_brief_matches():
    """mas_daily_thread_brief gates on
    `health.startswith("MAS GitHub pipeline healthy")`. If this script's
    wording drifts, the lane silently stops running and nothing else fails.
    That coupling is implicit in the code, so it is asserted here.
    """
    text = (REPO / "scripts" / "mas_github_pipeline_health.py").read_text()
    assert "MAS GitHub pipeline healthy" in text, "this gate must still emit the prefix"

    brief_path = REPO / "scripts" / "mas_daily_thread_brief.py"
    if not brief_path.exists():
        pytest.skip(
            "the consumer lands in #104; this assertion begins enforcing the "
            "contract from both sides once that merges"
        )
    assert "MAS GitHub pipeline healthy" in brief_path.read_text(), (
        "the brief matches this exact prefix; if either side is reworded the "
        "lane stops running and nothing else fails"
    )


def test_pending_and_unhealthy_do_not_match_the_healthy_prefix():
    for line in ("MAS GitHub pipeline pending | awaiting",
                 "⚠️ MAS GitHub pipeline unhealthy | failed closed"):
        assert not line.startswith("MAS GitHub pipeline healthy")


# ── attestation validation ───────────────────────────────────────────────

def test_valid_attestation_passes():
    gate.validate_attestation(attestation(), run_id=12345, head_sha=SHA)


@pytest.mark.parametrize("field", ["run_id", "attested", "healthy", "commit"])
def test_missing_required_field_is_a_schema_mismatch(field):
    payload = attestation()
    del payload[field]
    with pytest.raises(RuntimeError, match="schema mismatch"):
        gate.validate_attestation(payload, run_id=12345, head_sha=SHA)


def test_extra_field_is_also_a_schema_mismatch():
    with pytest.raises(RuntimeError, match="schema mismatch"):
        gate.validate_attestation(attestation(surprise=1), run_id=12345, head_sha=SHA)


def test_attestation_for_a_different_run_is_refused():
    with pytest.raises(RuntimeError, match="run id mismatch"):
        gate.validate_attestation(attestation(), run_id=99999, head_sha=SHA)


def test_attestation_for_a_different_commit_is_refused():
    with pytest.raises(RuntimeError, match="commit mismatch"):
        gate.validate_attestation(attestation(), run_id=12345, head_sha="b" * 40)


@pytest.mark.parametrize("override", [{"attested": False}, {"healthy": False}])
def test_unattested_or_unhealthy_is_refused(override):
    with pytest.raises(RuntimeError, match="unhealthy run"):
        gate.validate_attestation(attestation(**override), run_id=12345, head_sha=SHA)


@pytest.mark.parametrize("override", [
    {"governance_status": "failure"}, {"final_output_status": "failure"},
])
def test_failed_governance_or_output_is_refused(override):
    with pytest.raises(RuntimeError, match="governance"):
        gate.validate_attestation(attestation(**override), run_id=12345, head_sha=SHA)


@pytest.mark.parametrize("override", [
    {"db_error": "connection refused"}, {"run_id": ""}, {"run_id": 5},
])
def test_db_error_or_missing_mas_run_id_is_refused(override):
    with pytest.raises(RuntimeError, match="DB error or missing"):
        gate.validate_attestation(attestation(**override), run_id=12345, head_sha=SHA)


# ── a green workflow is not a worker run ─────────────────────────────────

def _job(step_name: str, conclusion: str) -> list[dict]:
    return [{"name": "Run scheduled pipeline", "steps": [{"name": step_name, "conclusion": conclusion}]}]


def test_successful_morning_step_counts_as_a_worker_run():
    assert gate.worker_ran(_job("Run morning pipeline", "success")) is True


def test_dst_duplicate_no_op_does_not_count():
    """Two cron lines fire per slot; the off-season one skips. A green run is
    therefore not evidence that the worker did anything."""
    assert gate.worker_ran(_job("Run morning pipeline", "skipped")) is False


def test_other_jobs_are_ignored():
    assert gate.worker_ran([{"name": "Publish dashboard to GitHub Pages",
                             "steps": [{"name": "Run morning pipeline", "conclusion": "success"}]}]) is False


def test_no_jobs_is_not_a_worker_run():
    assert gate.worker_ran([]) is False


# ── run selection ────────────────────────────────────────────────────────

def _run(run_id: int, *, event="schedule", conclusion="success", created="2026-08-20T11:00:00Z") -> dict:
    return {"databaseId": run_id, "event": event, "status": "completed",
            "conclusion": conclusion, "createdAt": created, "url": f"https://x/{run_id}"}


NOW = datetime(2026, 8, 20, 15, 0, tzinfo=UTC)


def test_selects_a_scheduled_successful_run_with_a_real_worker():
    runs = [_run(1)]
    jobs = {1: _job("Run morning pipeline", "success")}
    assert gate.select_current_et_actual_run(runs, jobs, NOW)["databaseId"] == 1


def test_manual_dispatch_cannot_authorize_the_brief():
    """Useful for retrieval proof, but it must not stand in for the
    authoritative scheduled worker."""
    runs = [_run(1, event="workflow_dispatch")]
    jobs = {1: _job("Run morning pipeline", "success")}
    assert gate.select_current_et_actual_run(runs, jobs, NOW) is None


def test_failed_run_is_not_selected():
    runs = [_run(1, conclusion="failure")]
    jobs = {1: _job("Run morning pipeline", "success")}
    assert gate.select_current_et_actual_run(runs, jobs, NOW) is None


def test_yesterdays_run_is_not_selected():
    runs = [_run(1, created="2026-08-19T11:00:00Z")]
    jobs = {1: _job("Run morning pipeline", "success")}
    assert gate.select_current_et_actual_run(runs, jobs, NOW) is None


def test_green_run_whose_worker_skipped_is_not_selected():
    runs = [_run(1)]
    jobs = {1: _job("Run morning pipeline", "skipped")}
    assert gate.select_current_et_actual_run(runs, jobs, NOW) is None


def test_utc_evening_run_is_attributed_to_the_correct_et_date():
    """01:30Z on the 21st is still the 20th in New York."""
    assert gate.et_date(_run(1, created="2026-08-21T01:30:00Z")).isoformat() == "2026-08-20"


# ── which changes require a re-run ───────────────────────────────────────

@pytest.mark.parametrize("path", [
    "src/main.py", "alembic/versions/x.py", "scripts/export_dashboard_data.py",
    "pyproject.toml", ".github/workflows/scheduled-pipelines.yml",
])
def test_pipeline_relevant_paths_are_detected(path):
    assert gate.pipeline_relevant([path]) is True


@pytest.mark.parametrize("path", ["README.md", "docs/agent_alignment.md", "outputs/research/x.md"])
def test_documentation_changes_are_not_pipeline_relevant(path):
    assert gate.pipeline_relevant([path]) is False


def test_short_sha_is_twelve_characters():
    assert gate.short(SHA) == "a" * 12
    assert gate.short("") == "unknown"
