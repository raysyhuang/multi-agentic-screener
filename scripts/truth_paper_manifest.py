#!/usr/bin/env python3
"""Validate a frozen MAS truth-paper track and write non-secret run manifests.

The scheduler lives on GitHub's default branch, while the candidate code is
checked out separately at a pinned immutable SHA.  This helper makes that
boundary visible in every retained run artifact without reading credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_CONFIG_FIELDS = {
    "schema_version",
    "track_id",
    "candidate_ref",
    "candidate_branch",
    "effective_date",
    "execution_mode",
    "trading_mode",
    "telegram_alert_prefix",
    "database_secret",
    "required_secrets",
    "recording_contract",
}


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def policy_sha256(data: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def load_and_validate_policy(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read policy JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("policy must be a JSON object")
    missing = sorted(_REQUIRED_CONFIG_FIELDS - data.keys())
    if missing:
        raise ValueError(f"policy missing required fields: {', '.join(missing)}")
    if data["schema_version"] != "mas_truth_paper_track_v1":
        raise ValueError("unsupported schema_version")
    if not _SHA_RE.fullmatch(str(data["candidate_ref"])):
        raise ValueError("candidate_ref must be a full lowercase 40-character commit SHA")
    if data["trading_mode"] != "PAPER":
        raise ValueError("truth-paper track must use trading_mode=PAPER")
    if data["recording_contract"].get("allow_live_execution") is not False:
        raise ValueError("truth-paper track must explicitly disallow live execution")
    if data["recording_contract"].get("immutable_candidate_ref") is not True:
        raise ValueError("truth-paper track must require an immutable candidate ref")
    if data["database_secret"] not in data["required_secrets"]:
        raise ValueError("database_secret must be included in required_secrets")
    return data


def git_value(candidate_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(candidate_dir), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_manifest(
    *,
    policy: dict[str, Any],
    candidate_dir: Path,
    output: Path,
    pipeline: str,
    runner_outcome: str,
    started_at: str,
) -> dict[str, Any]:
    candidate_sha = git_value(candidate_dir, "rev-parse", "HEAD")
    expected_sha = str(policy["candidate_ref"])
    if candidate_sha != expected_sha:
        raise ValueError(
            f"candidate checkout mismatch: expected {expected_sha}, got {candidate_sha}"
        )

    dirty = git_value(candidate_dir, "status", "--porcelain")
    if dirty:
        raise ValueError("candidate checkout is dirty; refusing to stamp a mutable run")

    manifest = {
        "schema_version": "mas_truth_paper_run_manifest_v1",
        "track_id": policy["track_id"],
        "policy_sha256": policy_sha256(policy),
        "candidate_ref": expected_sha,
        "candidate_branch": policy["candidate_branch"],
        "candidate_commit": candidate_sha,
        "effective_date": policy["effective_date"],
        "pipeline": pipeline,
        "runner_outcome": runner_outcome,
        "started_at_utc": started_at,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": policy["execution_mode"],
        "trading_mode": policy["trading_mode"],
        "alert_prefix": policy["telegram_alert_prefix"],
        "database_secret_name": policy["database_secret"],
        "recording_contract": policy["recording_contract"],
        "github": {
            "repository": os.getenv("GITHUB_REPOSITORY"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "workflow": os.getenv("GITHUB_WORKFLOW"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--print-field", choices=("candidate_ref", "policy_sha256", "track_id"))
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pipeline", choices=("morning", "afternoon"))
    parser.add_argument("--runner-outcome", choices=("success", "failure", "cancelled", "skipped"))
    parser.add_argument("--started-at")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        policy = load_and_validate_policy(args.policy)
        if args.print_field:
            value = policy_sha256(policy) if args.print_field == "policy_sha256" else policy[args.print_field]
            print(value)
            return 0
        if not all((args.candidate_dir, args.output, args.pipeline, args.runner_outcome, args.started_at)):
            raise ValueError("manifest mode requires candidate-dir, output, pipeline, runner-outcome, and started-at")
        manifest = write_manifest(
            policy=policy,
            candidate_dir=args.candidate_dir,
            output=args.output,
            pipeline=args.pipeline,
            runner_outcome=args.runner_outcome,
            started_at=args.started_at,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"truth-paper manifest error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
