"""Nothing may touch the aggregator after the pipeline releases it.

`run_morning_pipeline` deliberately frees the aggregator once Step 4 finishes —
`aggregator.close(); del aggregator; aggregator = None` — a memory measure
inherited from the 512 MB Heroku dyno. Any attribute access after that point
raises `AttributeError: 'NoneType' object has no attribute ...`, and because the
pipeline fails closed, the result is a NoTrade day for the entire book.

That is not hypothetical. PR #64 added `aggregator.get_data_provenance()` to the
governance record ~900 lines below the teardown and took the 2026-08-11 run down
with exactly that error. The unit suite could not see it: no test executes
`run_morning_pipeline` end to end, so the ordering was never exercised.

A source-level check is the honest tool here. The behavioural version needs a
full pipeline run with a live database and every provider stubbed, which does
not exist and would not run in the unit suite. This costs milliseconds and
catches the whole class.
"""

from __future__ import annotations

import re
from pathlib import Path

MAIN_PY = Path(__file__).resolve().parents[1] / "src" / "main.py"

# `_state["aggregator"] = None` is bookkeeping on the state dict, not a use of
# the released local, so it is not an attribute access on `aggregator` itself.
_ATTRIBUTE_ACCESS = re.compile(r"(?<![\w\"'])aggregator\s*\.")


def _lines() -> list[str]:
    return MAIN_PY.read_text().splitlines()


def _release_line(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if line.strip() == "aggregator = None":
            return i
    raise AssertionError(
        "could not find the aggregator release in src/main.py — if the teardown "
        "was removed or renamed, update this guard rather than deleting it"
    )


def test_nothing_uses_the_aggregator_after_it_is_released() -> None:
    lines = _lines()
    release = _release_line(lines)

    offenders = [
        (n, line.strip())
        for n, line in enumerate(lines[release + 1:], start=release + 2)
        if _ATTRIBUTE_ACCESS.search(line) and not line.strip().startswith("#")
    ]

    assert not offenders, (
        "src/main.py uses the aggregator after it is set to None. The pipeline "
        "fails closed, so this is a NoTrade day for the whole book, not a "
        "logging glitch. Snapshot what you need before the teardown:\n  "
        + "\n  ".join(f"line {n}: {code}" for n, code in offenders)
    )


def test_provenance_is_snapshotted_before_the_release() -> None:
    """The specific fix for the 2026-08-11 outage, pinned.

    The governance record must read a dict captured before teardown, not call
    the aggregator at write time.
    """
    lines = _lines()
    release = _release_line(lines)

    snapshot_lines = [
        n for n, line in enumerate(lines[:release])
        if "data_provenance = aggregator.get_data_provenance()" in line
    ]
    assert snapshot_lines, "provenance must be captured before the aggregator is released"


def test_the_smoke_test_files_runs_under_the_et_market_date() -> None:
    """Guard the timezone assumption the integration test depends on.

    The smoke test queried `DailyRun.run_date == date.today()` — the runner's
    UTC date — while the pipeline writes `_trading_date_et()`. They differ
    between 00:00 and 04:00 UTC, so CI was green for twenty hours a day and red
    every night, on every PR regardless of its contents.

    A source-level guard is right here specifically because the behavioural test
    cannot be relied on to catch it: the integration test passes most of the
    day, so a regression would land clean and only surface hours later, on
    somebody else's unrelated change.
    """
    smoke = (
        Path(__file__).resolve().parents[1]
        / "tests" / "integration" / "test_pipeline_smoke.py"
    ).read_text()

    for line in smoke.splitlines():
        if "DailyRun.run_date ==" in line:
            assert "_trading_date_et()" in line, (
                "the smoke test must compare against the ET market date the "
                f"pipeline actually writes, not a UTC date: {line.strip()}"
            )
