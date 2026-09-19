"""The research registry must not rot.

`docs/research_registry.md` is the grounding pack agents read before proposing a
strategy idea, and the only place a hypothesis's variant count is recorded (the
input a deflated Sharpe needs). Two ways it silently goes stale:

  * a new `*_FINDINGS.md` lands under `outputs/research/` and nobody registers
    it, so the next agent re-tests a dead idea;
  * a row is added without a verdict or a variant count, which is the same as
    not recording the trial.

Both are checked against the real files. No network.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "docs" / "research_registry.md"
VERDICTS = ("SHIPPED", "REJECTED", "BLOCKED-DATA", "WATCH", "OPEN")
# System git can be unusable on a developer Mac (Xcode licence); CI has plain git.
_GIT_CANDIDATES = ("git", "/opt/homebrew/bin/git", "/usr/local/bin/git")


def _tracked_findings() -> list[str]:
    last_error = "no git candidate ran"
    for exe in _GIT_CANDIDATES:
        try:
            out = subprocess.run(
                [exe, "-C", str(REPO), "ls-files", "outputs/research"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = f"{exe}: {exc}"
            continue
        if out.returncode == 0:
            return sorted(
                Path(line).name for line in out.stdout.splitlines()
                if line.endswith("_FINDINGS.md")
            )
        last_error = f"{exe}: exit {out.returncode}: {out.stderr.strip()[:120]}"
    pytest.skip(f"git unusable, cannot list tracked findings ({last_error})")


def _registry_rows() -> list[list[str]]:
    text = REGISTRY.read_text(encoding="utf-8")
    match = re.search(r"<!-- registry:start -->(.*?)<!-- registry:end -->", text, re.S)
    assert match, "registry table markers missing"
    rows = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == "ID":
            continue
        rows.append(cells)
    return rows


def test_every_tracked_findings_file_is_registered():
    text = REGISTRY.read_text(encoding="utf-8")
    findings = _tracked_findings()
    assert findings, "expected tracked *_FINDINGS.md files under outputs/research"
    missing = [name for name in findings if name not in text]
    assert not missing, (
        "findings files not referenced in docs/research_registry.md — register the "
        f"hypothesis, its verdict and its variant count: {missing}"
    )


def test_every_row_records_a_verdict_and_a_variant_count():
    rows = _registry_rows()
    assert len(rows) >= 30, f"registry looks truncated: {len(rows)} rows"
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate registry IDs"
    for cells in rows:
        assert len(cells) == 9, f"{cells[0]}: expected 9 columns, got {len(cells)}"
        rid, _family, mechanism, _data, _gate, verdict, variants, _date, source = cells
        assert mechanism, f"{rid}: empty mechanism"
        assert verdict.startswith(VERDICTS), f"{rid}: verdict must start with one of {VERDICTS}"
        # A number, or an explicit lower bound — never blank, never a guess dressed as a fact.
        assert re.match(r"^(\d+|unrecorded \(≥\d+)", variants), (
            f"{rid}: variants must be a count or 'unrecorded (≥N…)', got {variants!r}"
        )
        assert source, f"{rid}: no citation"
