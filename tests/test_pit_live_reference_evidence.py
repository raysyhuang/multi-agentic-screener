"""The retraction's evidence must stay reproducible, not just readable.

#81 first shipped its correction as prose citing a "frozen live snapshot" that
lived only in a gitignored directory — an unreproducible retraction of an
unreproducible finding, which is the exact provenance failure Phase A exists to
remove. The committed artifact fixes that; these tests stop it rotting.

The vintage's raw tree is gitignored, so CI cannot regenerate the artifact. It
CAN enforce that the artifact is internally coherent and still says what the
findings document says it says — which is what actually decays.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "outputs" / "research" / "evidence" / "pit_live_reference_audit.json"
FINDINGS = REPO / "outputs" / "research" / "PIT_PHASE_A_HALT_FINDINGS.md"


@pytest.fixture(scope="module")
def evidence() -> dict:
    assert ARTIFACT.exists(), (
        f"{ARTIFACT.relative_to(REPO)} is missing — the retraction's evidence must "
        "be committed, not left in gitignored outputs/"
    )
    return json.loads(ARTIFACT.read_text())


def test_the_artifact_identifies_its_source_and_generator(evidence):
    """Provenance is the point: a number with no traceable origin is prose."""
    for field in ("sha256", "path", "origin"):
        assert evidence["source"].get(field), f"source.{field} missing"
    for field in ("sha256", "script", "command"):
        assert evidence["generator"].get(field), f"generator.{field} missing"
    assert len(evidence["source"]["sha256"]) == 64
    assert evidence["generator"]["deterministic"] is True


def test_the_boundary_is_recorded_as_data_not_prose(evidence):
    """The merge that partitions the window must be auditable itself."""
    b = evidence["boundary"]
    assert b["pr"] == 63
    assert b["merged_at_utc"] == "2026-08-11T11:25:10Z"
    assert len(b["merge_commit"]) == 40
    assert "2026-08-11" in b["rule"]


def test_the_etf_contamination_claim_is_internally_consistent(evidence):
    """Every ETF/FUND live candidate predates the fix — the load-bearing claim."""
    c = evidence["claim_1_etf_candidates"]
    assert c["before_fix"] + c["on_or_after_fix"] == c["total"] == len(c["rows"])
    assert c["on_or_after_fix"] == 0, (
        "an ETF candidate on or after #63 would undercut the retraction entirely"
    )
    assert c["before_fix"] > 0
    assert all(r["post_fix"] is False for r in c["rows"])
    assert all(r["held_type"] in ("ETF", "ETN", "FUND") for r in c["rows"])


def test_the_divergence_inverts_at_the_boundary(evidence):
    """Sign inversion at the merge is what distinguishes a live defect from a PIT one."""
    s = evidence["claim_2_divergence_split"]
    pre, post = s["pre_fix"], s["post_fix"]
    assert pre["n"] + post["n"] == s["overlap_dates"]
    assert pre["median_signed_pct"] < 0, "pre-fix PIT should read smaller than live"
    assert post["median_signed_pct"] > 0, "post-fix the sign must invert"


def test_the_conclusion_claims_neither_pass_nor_fail(evidence):
    """A gate whose baseline moved mid-window certifies nothing in either direction."""
    c = evidence["conclusion"]
    assert "DEFERRED" in c["section_A5_live_count_gate"]
    assert c["dataset_acceptance"] == "HALTED"
    assert "contaminated" in c["pre_fix_live_counts"]
    assert "insufficient" in c["post_fix_live_counts"]


def test_the_findings_document_cites_the_artifact_and_its_hash(evidence):
    """Prose and evidence must not drift apart — that drift is how #81 started."""
    text = FINDINGS.read_text()
    assert "pit_live_reference_audit.json" in text, (
        "the findings document does not cite the evidence artifact"
    )
    assert evidence["source"]["sha256"][:16] in text, (
        "the findings document does not carry the source hash it depends on"
    )
    for n in (evidence["claim_1_etf_candidates"]["total"],
              evidence["claim_3_etf_supply"]["median_per_day"]):
        assert str(n) in text, f"figure {n} is in the artifact but not the document"


@pytest.mark.skipif(
    not (REPO / "outputs" / "pit_universe" / "2026-08-12" / "raw" / "live").exists(),
    reason="vintage raw tree is gitignored; regeneration only runs where it exists",
)
def test_the_artifact_regenerates_byte_identically():
    """Where the frozen source exists, the artifact must be reproducible from it."""
    import subprocess

    result = subprocess.run(
        ["python", "scripts/pit_live_reference_audit.py",
         "--vintage", "2026-08-12", "--verify"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"regeneration diverged:\n{result.stdout}\n{result.stderr}"
