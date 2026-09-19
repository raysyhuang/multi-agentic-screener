"""The research registry must not rot.

`docs/research_registry.md` is the grounding pack agents read before proposing a
strategy idea, and the only place a hypothesis's variant count is recorded (the
input a deflated Sharpe needs). Two ways it silently goes stale:

  * a new `*_FINDINGS.md` lands under `outputs/research/` and nobody registers
    it, so the next agent re-tests a dead idea;
  * a row is added without a verdict or an honest variant count, which is the
    same as not recording the trial.

Both are checked against the real files. No network.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "docs" / "research_registry.md"
VERDICTS = ("SHIPPED", "REJECTED", "BLOCKED-DATA", "WATCH", "NO ACTION", "OPEN")
# System git can be unusable on a developer Mac (Xcode licence); CI has plain git.
_GIT_CANDIDATES = ("git", "/opt/homebrew/bin/git", "/usr/local/bin/git")
# An exact count is only allowed when the source states it; everything else must
# say what kind of number it is. A trailing parenthetical note is optional.
_VARIANTS = re.compile(
    r"^(?:\d+|~\d+ \([^)]+\)|unrecorded \(≥\d+[^)]*\)|n/a \(infrastructure\))(?: \([^)]*\))?$"
)


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
    message = f"git unusable, cannot list tracked findings ({last_error})"
    if os.environ.get("CI"):
        # In CI a skip would be a silent pass: the guard would stop guarding.
        pytest.fail(message)
    pytest.skip(message)


def _table_text() -> str:
    text = REGISTRY.read_text(encoding="utf-8")
    match = re.search(r"<!-- registry:start -->(.*?)<!-- registry:end -->", text, re.S)
    assert match, "registry table markers missing"
    return match.group(1)


def _registry_rows() -> list[list[str]]:
    rows = []
    for line in _table_text().splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == "ID":
            continue
        rows.append(cells)
    return rows


def _unregistered(findings: list[str]) -> list[str]:
    """Findings count as registered only when named INSIDE the registry table —
    a mention in prose elsewhere in the document records no verdict and no count."""
    table = _table_text()
    return [name for name in findings if name not in table]


def test_every_tracked_findings_file_is_registered():
    findings = _tracked_findings()
    assert findings, "expected tracked *_FINDINGS.md files under outputs/research"
    missing = _unregistered(findings)
    assert not missing, (
        "findings files not cited in the registry TABLE of docs/research_registry.md — "
        f"register the hypothesis, its verdict and its variant count: {missing}"
    )


def test_an_unreferenced_findings_file_fails_the_guard(monkeypatch):
    """Prove the guard bites: add one findings name nobody registered."""
    import tests.test_research_registry as mod

    ghost = "never_registered_zzz_FINDINGS.md"
    assert _unregistered([ghost]) == [ghost]
    monkeypatch.setattr(mod, "_tracked_findings", lambda: ["pead_FINDINGS.md", ghost])
    with pytest.raises(AssertionError, match="never_registered_zzz"):
        mod.test_every_tracked_findings_file_is_registered()


def test_a_prose_only_mention_does_not_count_as_registered():
    # The comparator docs are named in prose under "Operational records" but are
    # not table rows; the guard must not treat that as a registration.
    text = REGISTRY.read_text(encoding="utf-8")
    assert "COMPARATOR_PINNING_METHOD.md" in text
    assert _unregistered(["COMPARATOR_PINNING_METHOD.md"]) == ["COMPARATOR_PINNING_METHOD.md"]


def test_git_failure_fails_in_ci_and_skips_locally(monkeypatch):
    import tests.test_research_registry as mod

    monkeypatch.setattr(mod, "_GIT_CANDIDATES", ("/nonexistent/git-binary",))
    monkeypatch.setenv("CI", "true")
    with pytest.raises(pytest.fail.Exception):
        mod._tracked_findings()
    monkeypatch.delenv("CI")
    with pytest.raises(pytest.skip.Exception):
        mod._tracked_findings()


@pytest.mark.parametrize("cell, ok", [
    ("5", True),
    ("5 (Runs A–E)", True),
    ("~12 (declared by the author, not audited)", True),
    ("unrecorded (≥18; 6 ratio buckets × 3 horizons)", True),
    ("n/a (infrastructure)", True),
    ("3 vetoes, 0 outcome variants", False),   # a leading integer is not enough
    ("8 reported (all 32 cutoffs enumerated)", False),
    ("", False),
    ("several", False),
])
def test_variant_cell_grammar(cell, ok):
    assert bool(_VARIANTS.match(cell)) is ok


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
        assert _VARIANTS.match(variants), (
            f"{rid}: variants must be an exact count the source states, '~N (…)', "
            f"'unrecorded (≥N…)' or 'n/a (infrastructure)'; got {variants!r}"
        )
        assert source, f"{rid}: no citation"
