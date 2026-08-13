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
PROVENANCE = REPO / "outputs" / "research" / "evidence" / "source" / "pipeline_run_provenance.json"
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


def test_the_boundary_is_derived_from_runs_not_from_a_date_inequality(evidence):
    """The correction review demanded: assert the RULE, not that prose has a date.

    My first version asserted `D >= 2026-08-11` from cron-timing reasoning; review
    proposed `D > 2026-08-11`. Both are unsound — the pipeline ran seven times on
    2026-08-11 with commits on both sides of the #63 merge, and DailyRun is
    upserted, so the surviving row depends on which ran last. No inequality on the
    date can express that, so the boundary is derived per date from what executed.
    """
    b = evidence["boundary"]
    assert b["pr"] == 63
    assert b["merge_commit"] == "c14d6d6f777b058e18fa8e4440ab5c9b3095d7d2"
    assert b["merged_at_utc"] == "2026-08-11T11:25:10Z"
    assert len(b["source_sha256"]) == 64, "the derivation source must be hashed"

    prov = json.loads(PROVENANCE.read_text())
    assert prov["boundary_merge_commit"] == b["merge_commit"]

    aug11 = prov["by_et_date"]["2026-08-11"]
    assert aug11["classification"] == "INDETERMINATE", (
        "2026-08-11 ran on both sides of the merge and must be unassignable"
    )
    sides = {r["contains_pr63"] for r in aug11["runs"]}
    assert sides == {True, False}, (
        f"fixture premise broken: 2026-08-11 runs are all on one side ({sides})"
    )
    assert prov["by_et_date"]["2026-08-12"]["classification"] == "post_fix"
    assert prov["by_et_date"]["2026-08-10"]["classification"] == "pre_fix"


def test_no_sign_inversion_is_claimed(evidence):
    """The retracted claim must stay retracted.

    The +35% figure came from one date the derived boundary marks INDETERMINATE.
    Reporting it as a post-fix observation is what review struck down.
    """
    s = evidence["claim_2_divergence_split"]
    assert s["post_fix"]["n"] == 0, (
        "a clean post-fix observation would need re-argument, not silent reuse"
    )
    assert s["post_fix"]["median_signed_pct"] is None
    assert s["indeterminate"]["n"] >= 1
    assert "not evidence in either direction" in s["indeterminate"]["note"]
    assert "NO sign-inversion claim" in s["statement"]


def test_the_source_closure_is_declared_incomplete_and_names_what_is_missing(evidence):
    """An unclosed evidence chain must be VISIBLE, not silently skipped in CI.

    Claims 1-3 still need 284MB of licensed Polygon data in a gitignored tree, so
    a clean checkout cannot regenerate them. That is a real gap; this test makes
    it fail loudly in review rather than hide behind a skipped regeneration test.
    Flip `complete` to True once the vintage is published and this test updates
    with it.
    """
    sc = evidence["source_closure"]
    assert sc["complete"] is False, (
        "source closure now claims completeness — update this test and prove a "
        "clean checkout regenerates the artifact"
    )
    assert sc["missing"], "an incomplete closure must enumerate what is missing"
    assert "redistribution" in sc["blocker"] or "Release" in sc["blocker"]
    for rel in sc["committed"]:
        assert (REPO / rel).exists(), f"declared-committed source is absent: {rel}"


def test_committed_source_bytes_match_their_recorded_hashes(evidence):
    """Review's point: test the BYTES, not that prose says 625."""
    import hashlib

    src = REPO / "outputs" / "research" / "evidence" / "source" / "dashboard_minimal.json"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    assert digest == evidence["source"]["sha256"], (
        f"committed source bytes do not match the hash the artifact records: "
        f"{digest[:16]} vs {evidence['source']['sha256'][:16]}"
    )


def test_the_etf_contamination_claim_is_internally_consistent(evidence):
    """Every ETF/FUND live candidate predates the fix — the load-bearing claim."""
    c = evidence["claim_1_etf_candidates"]
    assert (c["before_fix"] + c["on_or_after_fix"] + c["indeterminate"]
            == c["total"] == len(c["rows"]))
    assert c["on_or_after_fix"] == 0, (
        "an ETF candidate on or after #63 would undercut the retraction entirely"
    )
    assert c["before_fix"] > 0
    assert all(r["boundary_class"] != "post_fix" for r in c["rows"])
    assert all(r["held_type"] in ("ETF", "ETN", "FUND") for r in c["rows"])


def test_every_overlap_date_is_classified_by_the_boundary(evidence):
    """Every overlap date lands in exactly one bucket; none escapes the partition.

    Named for what it asserts. It was called
    `test_the_divergence_inverts_at_the_boundary` with a docstring about sign
    inversion — a claim withdrawn once 2026-08-11 was shown INDETERMINATE. The
    assertions had already been rewritten; the name had not, which would have
    told the next reviewer the suite still stood behind a retracted finding.
    """
    s = evidence["claim_2_divergence_split"]
    assert (s["pre_fix"]["n"] + s["post_fix"]["n"] + s["indeterminate"]["n"]
            == s["overlap_dates"]), "every overlap date must be classified"
    assert s["pre_fix"]["median_signed_pct"] < 0


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


# ── tamper + fail-closed regressions ─────────────────────────────────────────

def _run(*args) -> tuple[int, str]:
    import subprocess
    r = subprocess.run(["python", "scripts/pit_live_reference_audit.py", *args],
                       cwd=REPO, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


SOURCES = (
    "outputs/research/evidence/source/dashboard_minimal.json",
    "outputs/research/evidence/source/pipeline_run_provenance.json",
)


@pytest.mark.parametrize("rel", SOURCES)
def test_tampering_with_any_committed_source_fails_the_hash_check(rel):
    """Both sources must be VERIFIED, not merely printed.

    The first version compared the dashboard hash and only printed the provenance
    hash. That left a live hole: edit pipeline_run_provenance.json, the date
    classification changes, the artifact's recorded boundary hash goes stale, and
    CI stays green on a boundary that no longer matches its evidence.
    """
    path = REPO / rel
    original = path.read_bytes()
    assert _run("--check-sources")[0] == 0, "clean tree should verify"
    try:
        path.write_bytes(original + b"\n")
        code, out = _run("--check-sources")
        assert code != 0, f"tampering with {rel} was not detected:\n{out}"
        assert "MISMATCH" in out
        assert rel in out
    finally:
        path.write_bytes(original)
    assert _run("--check-sources")[0] == 0, "restore should re-verify"


def test_every_declared_source_is_actually_hash_checked():
    """A source can be declared committed but silently left unverified."""
    artifact = json.loads(ARTIFACT.read_text())
    assert set(artifact["source_closure"]["committed"]) == set(SOURCES), (
        "declared-committed sources and hash-checked sources have diverged; "
        "the check is driven by a table that must cover every declared source"
    )


def test_certification_is_mechanically_blocked_while_closure_is_incomplete():
    """Prose saying 'provisional' is not a gate.

    Review's point: a skipped regeneration test is silence, not a fail-closed
    certification gate. While the 284MB Polygon closure is unavailable, any path
    that tries to treat this artifact as acceptance-grade must exit non-zero.
    """
    code, out = _run("--certify")

    assert code != 0, "certification succeeded with an incomplete evidence chain"
    assert "CERTIFICATION_BLOCKED: incomplete_source_closure" in out
    assert "raw/grouped" in out, "the blocker must name what is missing"


def test_the_committed_source_carries_no_position_or_pnl_fields():
    """Self-reported defect: the full export was committed and mis-described.

    It carried entry_price and unrealised P&L for 14 open positions while the
    provenance note claimed it contained none. Already-public data, so nothing
    was disclosed — but the description was false and the collection served no
    claim. The committed source is now a projection of only the fields read.
    """
    data = json.loads((REPO / SOURCES[0]).read_text())

    for banned in ("open_positions", "trades", "today_picks", "portfolio"):
        assert banned not in data, f"{banned} is committed but no claim reads it"

    allowed = set(data["_projection"]["candidates"]) | set(data["_projection"]["run_history"])
    for key in ("candidates", "run_history"):
        for row in data[key]:
            leaked = {f for f in row if f not in allowed}
            assert not leaked, f"{key} rows carry unprojected fields: {leaked}"
