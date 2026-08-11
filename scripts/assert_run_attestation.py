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
    python scripts/assert_run_attestation.py --run-id <id> --out attestation.json

The caller owns the run id: the workflow mints MAS_RUN_ID before anything
fallible and passes the same value here, so a run that dies during startup can
still be attested against.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from src.db.models import PipelineArtifact  # noqa: E402
from src.db.session import close_db, get_session  # noqa: E402


def _emit(name: str, value: str) -> None:
    """Publish a GitHub Actions job output, and echo for the log.

    Genuinely non-throwing — the previous version claimed that while opening a
    file. A health check that can crash the run it is describing is the
    2026-08-11 outage in miniature, and a docstring asserting safety it does not
    implement is worse than no docstring.

    Job outputs are only visible to downstream jobs in the same workflow; the
    REST API does not expose them. They remain useful inside the run, but the
    artifact written by `--out` is what an external consumer reads.
    """
    print(f"{name}={value}")
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    try:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")
    except Exception as e:  # pragma: no cover - defensive
        print(f"::warning::could not write job output {name}: {e}", file=sys.stderr)


async def _check(run_id: str, out_path: Path | None) -> int:
    try:
        async with get_session() as session:
            artifacts = (await session.execute(
                select(PipelineArtifact).where(PipelineArtifact.run_id == run_id)
            )).scalars().all()
        db_error = None
    except Exception as e:
        # The database being unreachable is itself a fact the consumer needs,
        # and is distinct from "the run left no record".
        #
        # ONLY the exception class reaches the artifact. Connection and config
        # errors routinely carry the DSN — host, database, username, sometimes
        # the password — and this file is downloadable by anyone with repo read
        # access. The class name is enough to distinguish "cannot reach the
        # database" from "the run left no record", which is all the gate needs.
        artifacts, db_error = [], type(e).__name__
        print(
            f"::error::could not query run {run_id}: {type(e).__name__} "
            "(detail withheld — it can contain connection credentials)",
            file=sys.stderr,
        )

    stages = {a.stage: a.status for a in artifacts}
    governance = next((a for a in artifacts if a.stage == "governance"), None)
    final = next((a for a in artifacts if a.stage == "final_output"), None)

    attested = bool(governance) and not db_error
    healthy = (
        attested
        and governance.status != "failed"
        and (final is None or final.status != "failed")
    )

    record = {
        "run_id": run_id,
        "attested": attested,
        "healthy": healthy,
        "governance_status": governance.status if governance else "missing",
        "final_output_status": final.status if final else "missing",
        "artifact_stages": sorted(stages),
        "db_error": db_error,
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "commit": os.environ.get("GITHUB_SHA", ""),
    }

    for key in ("run_id", "governance_status", "final_output_status"):
        _emit(key, str(record[key]))
    _emit("attested", "true" if attested else "false")
    _emit("healthy", "true" if healthy else "false")

    if out_path is not None:
        # The durable half. Job outputs reach downstream jobs in this workflow
        # only — the REST API does not expose them — so an external consumer
        # (the VPS mirror) reads this file, uploaded as a named artifact.
        try:
            out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"wrote attestation to {out_path}")
        except Exception as e:
            print(f"::error::could not write attestation file: {e}", file=sys.stderr)
            return 1

    if db_error:
        return 1  # already reported above, without the detail

    if governance is None:
        print(
            f"::error::run {run_id} left no governance artifact. Since PR #71 one "
            "is written on every path, including crashes before the pipeline "
            "core starts, so absence means the run did not record itself.",
            file=sys.stderr,
        )
        return 1

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
    ap.add_argument("--out", help="write the attestation JSON here (uploaded as an artifact)")
    args = ap.parse_args()

    run_id = args.run_id

    if not run_id:
        ap.error("--run-id is required")

    out_path = Path(args.out) if args.out else None

    async def _run() -> int:
        try:
            return await _check(run_id, out_path)
        finally:
            try:
                await close_db()
            except Exception:
                pass

    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
