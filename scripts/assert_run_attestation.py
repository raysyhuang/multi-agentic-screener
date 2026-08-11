"""Assert that a pipeline run left the evidence it is contractually required to.

Written for the VPS mirror's health gate. The mirror cannot see this: the
database is private, and the published dashboard snapshot exports `DailyRun`
health but not per-run `PipelineArtifact(stage="governance")`. So a health check
running outside GitHub has no way to verify the run actually recorded itself —
it can only observe the workflow's conclusion, which has now been green on a
broken run twice (a DST-guard skip, and a fail-closed NoTrade).

This runs inside the workflow, where the database IS reachable, checks the exact
run, and publishes the verdict as a GitHub job output. Consumers then read a
real attestation instead of inferring health from a green tick or scraping a
dashboard approximation.

Exits non-zero if the run left no governance artifact, which after PR #71 is
guaranteed on every path including early crashes — so its absence is a genuine
fault, not an expected state.

Usage:
    python scripts/assert_run_attestation.py --run-id <id>
    python scripts/assert_run_attestation.py --run-id-file "$RUNNER_TEMP/run_id"
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from src.db.models import PipelineArtifact  # noqa: E402
from src.db.session import close_db, get_session  # noqa: E402


def _emit(name: str, value: str) -> None:
    """Publish a GitHub Actions job output, and echo for the log."""
    print(f"{name}={value}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")


async def _check(run_id: str) -> int:
    async with get_session() as session:
        artifacts = (await session.execute(
            select(PipelineArtifact).where(PipelineArtifact.run_id == run_id)
        )).scalars().all()

    stages = {a.stage: a.status for a in artifacts}
    governance = next((a for a in artifacts if a.stage == "governance"), None)
    final = next((a for a in artifacts if a.stage == "final_output"), None)

    _emit("run_id", run_id)
    _emit("artifact_stages", ",".join(sorted(stages)) or "none")
    _emit("governance_status", governance.status if governance else "missing")
    _emit("final_output_status", final.status if final else "missing")

    if governance is None:
        print(
            f"::error::run {run_id} left no governance artifact. Since PR #71 one "
            "is written on every path, including crashes before the pipeline "
            "core starts, so absence means the run did not record itself.",
            file=sys.stderr,
        )
        _emit("attested", "false")
        return 1

    healthy = governance.status != "failed" and (final is None or final.status != "failed")
    _emit("attested", "true" if healthy else "false")

    if not healthy:
        print(
            f"::warning::run {run_id} recorded itself as failed "
            f"(governance={governance.status}, "
            f"final_output={final.status if final else 'missing'})",
            file=sys.stderr,
        )
        # Not an error exit: the run correctly recorded its own failure, which is
        # the contract working. The worker's exit code (PR #72) is what turns the
        # workflow red; this only publishes what the record says.
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--run-id-file", help="file the pipeline wrote its run id to")
    args = ap.parse_args()

    run_id = args.run_id
    if not run_id and args.run_id_file:
        path = Path(args.run_id_file)
        if not path.exists():
            print(
                f"::error::{path} does not exist — the pipeline never recorded a "
                "run id, so it did not get far enough to identify itself.",
                file=sys.stderr,
            )
            _emit("attested", "false")
            _emit("governance_status", "missing")
            sys.exit(1)
        run_id = path.read_text().strip()

    if not run_id:
        ap.error("one of --run-id or --run-id-file is required")

    async def _run() -> int:
        try:
            return await _check(run_id)
        finally:
            await close_db()

    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
