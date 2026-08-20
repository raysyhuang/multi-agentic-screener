#!/usr/bin/env python3
"""Fail-closed, GitHub-authoritative MAS health gate for the Boston paper mirror."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# The one deployment-specific value in this file. Supplied by the operator;
# NO DEFAULT, because a default publishes the layout of the host that runs it.
# Everything else below (GITHUB_REPO, WORKFLOW, PIPELINE_PREFIXES) is a
# repo-side fact and belongs in the repo.
#
#   MAS_HEALTH_REPO  the mirror checkout this gate fetches and fast-forwards


class ConfigError(RuntimeError):
    """Deployment configuration is missing or unusable."""


def resolve_repo() -> Path:
    raw = os.environ.get("MAS_HEALTH_REPO", "").strip()
    if not raw:
        raise ConfigError(
            "MAS_HEALTH_REPO is not set. This gate ships without host defaults so "
            "that no deployment layout is published in the repository; set it in "
            "the scheduler unit that invokes it."
        )
    repo = Path(raw).expanduser()
    if not repo.is_dir():
        raise ConfigError(f"MAS_HEALTH_REPO does not exist or is not a directory: {repo}")
    return repo
GITHUB_REPO = "raysyhuang/multi-agentic-screener"
WORKFLOW = "Scheduled Pipelines (GitHub-hosted)"
ET = ZoneInfo("America/New_York")
PIPELINE_PREFIXES = ("src/", "alembic/", "scripts/", "pyproject.toml", ".github/workflows/scheduled-pipelines.yml")


def run_text(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, timeout=90, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")
    return result.stdout.strip()


def run_bytes(command: list[str]) -> bytes:
    result = subprocess.run(command, capture_output=True, timeout=90, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")
    return result.stdout


def attestation_for_run(run_id: int) -> dict:
    """Download the workflow's durable per-run attestation artifact.

    Job outputs are not exposed by the Actions REST API.  The workflow uploads
    exactly one named JSON artifact, which is the public, GitHub-backed proof
    that the DB observed this particular run.
    """
    payload = json.loads(run_text([
        "gh", "api", f"repos/{GITHUB_REPO}/actions/runs/{run_id}/artifacts",
    ]))
    matches = [
        artifact for artifact in payload.get("artifacts", [])
        if artifact.get("name") == "mas-run-attestation" and not artifact.get("expired")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one live mas-run-attestation artifact, found {len(matches)}")
    artifact_id = matches[0].get("id")
    if not artifact_id:
        raise RuntimeError("attestation artifact has no id")
    archive = run_bytes([
        "gh", "api", f"repos/{GITHUB_REPO}/actions/artifacts/{artifact_id}/zip",
    ])
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            json_names = [name for name in bundle.namelist() if name.endswith(".json")]
            if len(json_names) != 1:
                raise RuntimeError(f"attestation archive must contain one JSON file, found {len(json_names)}")
            return json.loads(bundle.read(json_names[0]))
    except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid attestation artifact: {type(exc).__name__}") from exc


def validate_attestation(attestation: dict, *, run_id: int, head_sha: str) -> None:
    """Require durable DB evidence for exactly this successful GitHub worker."""
    required = {
        "run_id", "attested", "healthy", "governance_status",
        "final_output_status", "artifact_stages", "db_error",
        "github_run_id", "github_run_attempt", "commit",
    }
    if set(attestation) != required:
        raise RuntimeError("attestation schema mismatch")
    if str(attestation["github_run_id"]) != str(run_id):
        raise RuntimeError("attestation GitHub run id mismatch")
    if str(attestation["commit"]) != head_sha:
        raise RuntimeError("attestation commit mismatch")
    if not attestation["attested"] or not attestation["healthy"]:
        raise RuntimeError("attestation reports unhealthy run")
    if attestation["governance_status"] != "success" or attestation["final_output_status"] != "success":
        raise RuntimeError("attestation did not confirm successful governance/final output")
    if attestation["db_error"] is not None or not isinstance(attestation["run_id"], str) or not attestation["run_id"]:
        raise RuntimeError("attestation has DB error or missing MAS run id")


def short(sha: str) -> str:
    return sha[:12] if sha else "unknown"


def worker_ran(jobs: list[dict]) -> bool:
    """A green scheduled workflow is not enough: the DST duplicate may be a no-op."""
    for job in jobs:
        if job.get("name") != "Run scheduled pipeline":
            continue
        for step in job.get("steps") or []:
            if step.get("name") == "Run morning pipeline" and step.get("conclusion") == "success":
                return True
    return False


def et_date(run: dict) -> datetime.date:
    # GitHub returns "…Z"; fromisoformat accepts it only as "+00:00" before 3.11.
    # GitHub timestamps end in "Z"; normalise before parsing rather than
    # relying on version-specific fromisoformat behaviour.
    created = str(run["createdAt"])
    if created.endswith("Z"):
        created = created[:-1] + "+00:00"
    return datetime.fromisoformat(created).astimezone(ET).date()


def select_current_et_actual_run(runs: list[dict], jobs_by_id: dict[int, list[dict]], now: datetime) -> dict | None:
    today = now.astimezone(ET).date()
    for candidate in runs:
        # A manual dispatch is useful for retrieval proof, but must not authorize
        # the daily paper brief in place of the authoritative scheduled worker.
        if candidate.get("event") != "schedule":
            continue
        if candidate.get("status") != "completed" or candidate.get("conclusion") != "success":
            continue
        if et_date(candidate) != today:
            continue
        run_id = int(candidate["databaseId"])
        if worker_ran(jobs_by_id.get(run_id, [])):
            return candidate
    return None


def pipeline_relevant(paths: list[str]) -> bool:
    return any(path == prefix or path.startswith(prefix) for path in paths for prefix in PIPELINE_PREFIXES)


def job_steps(run_id: int) -> list[dict]:
    payload = run_text(["gh", "run", "view", str(run_id), "--repo", GITHUB_REPO, "--json", "jobs"])
    return json.loads(payload).get("jobs", [])


def fetch_and_fast_forward(REPO: Path) -> tuple[str, str]:
    # Only tracked modifications can alter the executable checkout; review drops
    # and other untracked scratch files must not wedge a read-only fast-forward.
    if run_text(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]):
        raise RuntimeError("VPS checkout is dirty; refusing automatic sync")
    # Network/DNS/TLS failures on the pull-only fetch are transient often enough
    # that a single attempt created false "unhealthy" daily notices.  Retrying
    # the exact read-only operation does not weaken the authority gate: a final
    # failure still prevents both synchronization and the paper mirror.
    fetch_error: RuntimeError | None = None
    for attempt in range(3):
        try:
            run_text(["git", "-C", str(REPO), "fetch", "origin", "--prune"])
            fetch_error = None
            break
        except RuntimeError as exc:
            fetch_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    if fetch_error is not None:
        raise RuntimeError("origin fetch failed after 3 attempts") from fetch_error
    run_text(["git", "-C", str(REPO), "merge", "--ff-only", "origin/main"])
    return (
        run_text(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        run_text(["git", "-C", str(REPO), "rev-parse", "origin/main"]),
    )


def main() -> int:
    try:
        REPO = resolve_repo()
    except ConfigError as exc:
        print(f"⚠️ MAS GitHub pipeline unhealthy | health gate failed closed: {exc}")
        return 1
    try:
        local_sha, remote_sha = fetch_and_fast_forward(REPO)
        payload = run_text(["gh", "run", "list", "--repo", GITHUB_REPO, "--workflow", WORKFLOW, "--limit", "12", "--json", "databaseId,status,conclusion,headSha,createdAt,url,event"])
        runs = json.loads(payload)
        if not runs:
            raise RuntimeError("GitHub returned no scheduled-pipeline runs")
        now = datetime.now(ET)
        today_runs = [r for r in runs if et_date(r) == now.date()]
        jobs_by_id = {int(r["databaseId"]): job_steps(int(r["databaseId"])) for r in today_runs if r.get("status") == "completed" and r.get("conclusion") == "success"}
        actual = select_current_et_actual_run(runs, jobs_by_id, now)
        if actual is None:
            failures = [r for r in today_runs if r.get("conclusion") == "failure"]
            if failures:
                failed = failures[0]
                print(f"⚠️ MAS GitHub pipeline unhealthy | current ET-date workflow failed | {failed['url']}")
                return 1
            print(f"MAS GitHub pipeline pending | VPS/main={short(local_sha)} | awaiting an actual morning worker run today ET")
            return 0
        actual_sha = str(actual.get("headSha") or "")
        actual_run_id = int(actual["databaseId"])
        changed = run_text(["git", "-C", str(REPO), "diff", "--name-only", f"{actual_sha}..{remote_sha}"]).splitlines() if actual_sha != remote_sha else []
        if pipeline_relevant(changed):
            print(f"MAS GitHub pipeline pending | VPS/main={short(local_sha)} | actual worker={short(actual_sha)} | pipeline-relevant main changes await execution | {actual['url']}")
            return 0
        validate_attestation(
            attestation_for_run(actual_run_id),
            run_id=actual_run_id,
            head_sha=actual_sha,
        )
        note = "" if not changed else " | newer docs/dashboard-only main changes accepted"
        print(f"MAS GitHub pipeline healthy | VPS/main={short(local_sha)} | actual morning worker={short(actual_sha)} | run=success | {actual['url']}{note}")
        return 0
    except Exception as exc:  # noqa: BLE001 - fail closed on ANY fault; a gate that
        # half-answers would authorise the lane on an unverified premise.
        print(f"⚠️ MAS GitHub pipeline unhealthy | health gate failed closed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
