from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "truth_paper_manifest.py"
POLICY_PATH = PROJECT_ROOT / "config" / "paper_tracks" / "truth_lean_v1.json"
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "mas-truth-paper.yml"


spec = importlib.util.spec_from_file_location("truth_paper_manifest", SCRIPT_PATH)
assert spec and spec.loader
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _candidate_repo(tmp_path: Path) -> tuple[Path, str]:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    _git(candidate, "init")
    _git(candidate, "config", "user.name", "Test User")
    _git(candidate, "config", "user.email", "test@example.invalid")
    (candidate / "candidate.txt").write_text("frozen candidate\n", encoding="utf-8")
    _git(candidate, "add", "candidate.txt")
    _git(candidate, "commit", "-m", "candidate")
    return candidate, _git(candidate, "rev-parse", "HEAD")


def _policy_for(candidate_sha: str) -> dict:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["candidate_ref"] = candidate_sha
    return policy


def test_truth_paper_policy_is_pinned_paper_only_and_hashable():
    policy = manifest.load_and_validate_policy(POLICY_PATH)

    assert policy["track_id"] == "mas-truth-paper-v1"
    assert policy["candidate_ref"] == "b54c7591f3a0228c535608d5b3387ce7f66a150f"
    assert policy["trading_mode"] == "PAPER"
    assert policy["recording_contract"]["allow_live_execution"] is False
    assert policy["recording_contract"]["immutable_candidate_ref"] is True
    assert len(manifest.policy_sha256(policy)) == 64


def test_truth_paper_policy_rejects_moving_ref(tmp_path):
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["candidate_ref"] = "truth-lean"
    bad = tmp_path / "bad-ref.json"
    bad.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate_ref"):
        manifest.load_and_validate_policy(bad)


def test_truth_paper_policy_rejects_live_execution(tmp_path):
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["trading_mode"] = "LIVE"
    bad = tmp_path / "bad-live.json"
    bad.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="trading_mode"):
        manifest.load_and_validate_policy(bad)


def test_truth_paper_policy_hash_is_canonical_not_source_whitespace(tmp_path):
    policy = manifest.load_and_validate_policy(POLICY_PATH)
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(policy, indent=4, sort_keys=False), encoding="utf-8")

    assert manifest.policy_sha256(policy) == manifest.policy_sha256(
        manifest.load_and_validate_policy(reformatted)
    )


def test_write_manifest_stamps_matching_clean_candidate_without_secret(tmp_path, monkeypatch):
    candidate, candidate_sha = _candidate_repo(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_policy_for(candidate_sha)), encoding="utf-8")
    output = tmp_path / "nested" / "manifest.json"
    monkeypatch.setenv("DATABASE_URL", "postgres://private-user:private-password@example.invalid/db")

    rendered = manifest.write_manifest(
        policy=manifest.load_and_validate_policy(policy_path),
        candidate_dir=candidate,
        output=output,
        pipeline="morning",
        runner_outcome="success",
        started_at="2026-07-21T00:00:00Z",
    )

    serialized = output.read_text(encoding="utf-8")
    assert output.exists()
    assert rendered["candidate_commit"] == candidate_sha
    assert rendered["runner_outcome"] == "success"
    assert "private-password" not in serialized
    assert "postgres://" not in serialized


def test_write_manifest_rejects_mismatched_or_dirty_candidate(tmp_path):
    candidate, candidate_sha = _candidate_repo(tmp_path)
    policy = _policy_for(candidate_sha)
    policy["candidate_ref"] = "0" * 40

    with pytest.raises(ValueError, match="checkout mismatch"):
        manifest.write_manifest(
            policy=policy,
            candidate_dir=candidate,
            output=tmp_path / "mismatch.json",
            pipeline="morning",
            runner_outcome="failure",
            started_at="2026-07-21T00:00:00Z",
        )

    policy["candidate_ref"] = candidate_sha
    (candidate / "candidate.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        manifest.write_manifest(
            policy=policy,
            candidate_dir=candidate,
            output=tmp_path / "dirty.json",
            pipeline="morning",
            runner_outcome="failure",
            started_at="2026-07-21T00:00:00Z",
        )


def test_workflow_keeps_database_isolated_paper_only_and_artifact_safe():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "DATABASE_URL: ${{ secrets.DATABASE_URL_TRUTH_LEAN }}" in text
    assert "TRADING_MODE: PAPER" in text
    assert "TELEGRAM_ALERT_PREFIX: MAS-TRUTH-PAPER" in text
    assert "tee " not in text
    assert "runner-${{ steps.resolve.outputs.pipeline }}.log" not in text
    assert re.findall(r"uses: actions/[^@]+@[0-9a-f]{40} # v[45]", text) == [
        "uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4",
        "uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5",
        "uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4",
        "uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4",
    ]


def test_module_does_not_depend_on_process_environment_for_policy_hash():
    policy = manifest.load_and_validate_policy(POLICY_PATH)
    original = manifest.policy_sha256(policy)
    old = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = "postgres://should-not-affect-policy"
    try:
        assert manifest.policy_sha256(policy) == original
    finally:
        if old is None:
            del os.environ["DATABASE_URL"]
        else:
            os.environ["DATABASE_URL"] = old
