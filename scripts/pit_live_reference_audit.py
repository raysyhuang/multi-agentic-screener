"""Reproduce the evidence behind the PIT-vs-live divergence retraction (#81).

The retraction claimed the live universe count was contaminated by ETFs before
`#63` merged. That claim was first published as prose, citing a "frozen live
snapshot" that lived only in a gitignored directory — an unreproducible
correction to an unreproducible finding, which is precisely the provenance
failure Phase A exists to eliminate.

This script regenerates every number in that conclusion from frozen bytes and
emits a compact evidence artifact that is committed to the repo. Output is
**byte-deterministic**: no wall-clock, no ordering that depends on filesystem
iteration, so re-running must reproduce the identical file and therefore the
identical SHA-256. A reviewer can verify the artifact by regenerating it rather
than by trusting it.

Usage:
    python scripts/pit_live_reference_audit.py --vintage 2026-08-12
    python scripts/pit_live_reference_audit.py --vintage 2026-08-12 --verify
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent / "outputs" / "pit_universe"
ARTIFACT = (
    Path(__file__).resolve().parent.parent
    / "outputs" / "research" / "evidence" / "pit_live_reference_audit.json"
)

# The definition change that partitions the comparison window.
#
# The boundary is NOT a date rule. My first version asserted "post-fix iff
# D >= 2026-08-11", reasoning from the merge time against cron timing; review
# proposed the strict "D > 2026-08-11" instead. Both are wrong, because the
# pipeline ran SEVEN times on 2026-08-11 with commits on BOTH sides of the merge
# (10:41Z c73568a5 without it, 11:39Z 4e0addb6 with it, and more), and DailyRun
# is upserted so the surviving row is whichever ran last. No inequality on the
# date can express that.
#
# So the boundary is derived per date from what actually executed, frozen in
# `evidence/source/pipeline_run_provenance.json`: a date is assignable only if
# every run that could have written its row sat on one side of the merge.
BOUNDARY_PR = 63
BOUNDARY_MERGE_COMMIT = "c14d6d6f777b058e18fa8e4440ab5c9b3095d7d2"
BOUNDARY_MERGED_AT_UTC = "2026-08-11T11:25:10Z"

EXCLUDED_TYPES = ("ETF", "ETN", "FUND")

# Committed source closure. These bytes are IN the repo, so the claims they
# support are reproducible from a clean checkout rather than from a directory
# that only exists on one machine.
EVIDENCE_SRC = Path(__file__).resolve().parent.parent / "outputs" / "research" / "evidence" / "source"
DASHBOARD_SRC = EVIDENCE_SRC / "dashboard_minimal.json"
RUN_PROVENANCE_SRC = EVIDENCE_SRC / "pipeline_run_provenance.json"


def load_boundary() -> dict:
    """Per-date classification derived from the runs that actually executed."""
    prov = json.loads(RUN_PROVENANCE_SRC.read_text())
    if prov["boundary_merge_commit"] != BOUNDARY_MERGE_COMMIT:
        raise SystemExit("run provenance describes a different merge boundary")
    return prov


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read())


def build_evidence(vintage: str) -> dict:
    from pit_universe_phase_a import (  # noqa: PLC0415
        ALLOWED_EXCHANGES, MIN_PRICE, MIN_SHARE_VOLUME,
        _classification_by_month, _label_for, build_membership,
    )

    # Read the COMMITTED copy, not the gitignored vintage. The two are verified
    # byte-identical below; reading the committed one is what makes a clean
    # checkout able to reproduce this at all.
    live_path = DASHBOARD_SRC
    if not live_path.exists():
        raise SystemExit(f"no committed dashboard source at {live_path}")
    live_min = json.loads(live_path.read_text())
    vintage_copy = ROOT / vintage / "raw" / "live" / "dashboard.json.gz"
    if vintage_copy.exists():
        # The committed file is a field PROJECTION of the vintage copy, not a
        # byte copy, so verify the projection instead of the hash: a projection
        # can drop fields but must never introduce or alter a row.
        parent = _read_gz(vintage_copy)
        if _sha256_file(vintage_copy) != live_min["_parent_export_sha256"]:
            raise SystemExit("committed source does not derive from this vintage's export")
        for key, fields in live_min["_projection"].items():
            got, want = live_min[key], parent.get(key) or []
            if len(got) != len(want):
                raise SystemExit(f"projection changed row count for {key}")
            for a, b in zip(got, want):
                if any(a[f] != b.get(f) for f in fields):
                    raise SystemExit(f"projection altered a value in {key}")

    boundary = load_boundary()
    klass = {d: v["classification"] for d, v in boundary["by_et_date"].items()}
    live = live_min
    labels_by_month = _classification_by_month(vintage)
    membership = build_membership(vintage)

    # ── claim 1: ETF/FUND names that appeared as RANKED LIVE candidates ──────
    etf_rows = []
    for row in live.get("candidates") or []:
        ticker, run_date = row.get("ticker"), row.get("run_date")
        if not ticker or not run_date:
            continue
        try:
            d = date.fromisoformat(str(run_date)[:10])
        except ValueError:
            continue
        held_type = (_label_for(labels_by_month, d).get(ticker) or {}).get("type")
        if held_type in EXCLUDED_TYPES:
            etf_rows.append({
                "date": str(d), "ticker": ticker, "held_type": held_type,
                "model": row.get("model"), "rank": row.get("rank"),
                "picked": row.get("picked"),
                "boundary_class": klass.get(str(d), "pre_fix"),
            })
    etf_rows.sort(key=lambda r: (r["date"], r["ticker"]))

    # ── claim 2: the pre/post signed-divergence split ────────────────────────
    pairs = []
    for row in live.get("run_history") or []:
        raw, live_n = row.get("date"), row.get("universe")
        if not raw or not live_n:
            continue
        try:
            d = date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        if d not in membership:
            continue
        pit_n = len(membership[d]["eligible_pre_mcap"])
        pairs.append({
            "date": str(d), "pit": pit_n, "live": live_n,
            "signed_pct": round(100.0 * (pit_n - live_n) / live_n, 2),
            "boundary_class": klass.get(str(d), "pre_fix"),
        })
    pairs.sort(key=lambda p: p["date"])
    pre = [p["signed_pct"] for p in pairs if p["boundary_class"] == "pre_fix"]
    post = [p["signed_pct"] for p in pairs if p["boundary_class"] == "post_fix"]
    indet = [p["signed_pct"] for p in pairs if p["boundary_class"] == "INDETERMINATE"]

    # ── claim 3: ETF/day passing price+volume, from PIT's own funnel ─────────
    etf_per_day = []
    for p in pairs:
        d = date.fromisoformat(p["date"])
        grouped = ROOT / vintage / "raw" / "grouped" / f"{d}.json.gz"
        if not grouped.exists():
            continue
        labels = _label_for(labels_by_month, d)
        n = 0
        for bar in _read_gz(grouped).get("results") or []:
            t, c, v = bar.get("T"), bar.get("c"), bar.get("v")
            if not t or c is None or v is None:
                continue
            if c < MIN_PRICE or v < MIN_SHARE_VOLUME:
                continue
            lab = labels.get(t) or {}
            if lab.get("exchange") in ALLOWED_EXCHANGES and lab.get("type") == "ETF":
                n += 1
        etf_per_day.append(n)

    return {
        "artifact": "pit_live_reference_audit",
        "schema_version": 1,
        "vintage": vintage,
        "source": {
            "path": "outputs/research/evidence/source/dashboard_minimal.json",
            "sha256": _sha256_file(live_path),
            "bytes": live_path.stat().st_size,
            "origin": "https://raysyhuang.github.io/multi-agentic-screener/data.json",
            "note": (
                "Field projection of the PUBLIC dashboard export, carrying only "
                "candidates{ticker,run_date,model,rank,picked} and "
                "run_history{date,universe} — the only fields any claim reads. "
                "open_positions, trades, today_picks and portfolio are excluded. "
                "An earlier revision committed the FULL export and described it as "
                "containing no positions, prices or P&L; that was false — it "
                "carried entry_price and unrealised P&L for 14 open positions. "
                "The data was already public, so nothing was disclosed, but the "
                "description was wrong and the collection was unnecessary."
            ),
        },
        "source_closure": {
            # Stated explicitly rather than implied. A clean checkout CANNOT
            # regenerate this artifact today: claims 1-3 need the vintage's
            # reference labels and grouped bars, which are 284MB of licensed
            # Polygon data in a gitignored tree. Publishing them is a vendor
            # redistribution question, not a technical one, so it is escalated
            # rather than decided here.
            "complete": False,
            "committed": [
                "outputs/research/evidence/source/dashboard_minimal.json",
                "outputs/research/evidence/source/pipeline_run_provenance.json",
            ],
            "missing": [
                "outputs/pit_universe/<vintage>/raw/reference/**  (17MB, Polygon)",
                "outputs/pit_universe/<vintage>/raw/grouped/**   (239MB, Polygon)",
            ],
            "blocker": (
                "Full closure needs the vintage published as a versioned Release "
                "asset with a committed manifest handle. 284MB exceeds git limits, "
                "and it is licensed vendor market data — redistribution terms must "
                "be confirmed before publication. Until then claims 1-3 are "
                "REPRODUCIBLE ONLY where the vintage exists, and CI can verify the "
                "committed subset and the artifact's internal consistency only."
            ),
            "verifiable_from_committed_bytes": [
                "boundary classification (pipeline_run_provenance.json)",
                "live universe counts per date (dashboard.json.gz)",
            ],
        },
        "generator": {
            "script": "scripts/pit_live_reference_audit.py",
            "sha256": _sha256_file(Path(__file__).resolve()),
            "command": f"python scripts/pit_live_reference_audit.py --vintage {vintage}",
            "deterministic": True,
        },
        "boundary": {
            "pr": BOUNDARY_PR,
            "merge_commit": BOUNDARY_MERGE_COMMIT,
            "merged_at_utc": BOUNDARY_MERGED_AT_UTC,
            "derivation": (
                "Per ET date, from the commits that actually ran the pipeline "
                "(evidence/source/pipeline_run_provenance.json). A date is "
                "assignable only if every run sat on one side of the merge; "
                "otherwise INDETERMINATE. NOT a date inequality — 2026-08-11 ran "
                "seven times with commits on both sides."
            ),
            "source_sha256": _sha256_file(RUN_PROVENANCE_SRC),
            "classification_counts": {
                k: sum(1 for v in klass.values() if v == k)
                for k in ("pre_fix", "post_fix", "INDETERMINATE")
            },
        },
        "claim_1_etf_candidates": {
            "statement": (
                "ETF/FUND/ETN names appeared as ranked LIVE candidates only before #63."
            ),
            "total": len(etf_rows),
            "before_fix": sum(1 for r in etf_rows if r["boundary_class"] == "pre_fix"),
            "on_or_after_fix": sum(1 for r in etf_rows if r["boundary_class"] == "post_fix"),
            "indeterminate": sum(1 for r in etf_rows if r["boundary_class"] == "INDETERMINATE"),
            "date_range": [etf_rows[0]["date"], etf_rows[-1]["date"]] if etf_rows else None,
            "rows": etf_rows,
        },
        "claim_2_divergence_split": {
            "statement": (
                "Median PIT-vs-live signed count divergence, partitioned by which "
                "commit produced each date. NO sign-inversion claim is made: the "
                "earlier version rested on a single observation that the derived "
                "boundary shows is INDETERMINATE, not post-fix."
            ),
            "overlap_dates": len(pairs),
            "pre_fix": {
                "n": len(pre),
                "median_signed_pct": round(statistics.median(pre), 2) if pre else None,
            },
            "post_fix": {
                "n": len(post),
                "median_signed_pct": round(statistics.median(post), 2) if post else None,
            },
            "indeterminate": {
                "n": len(indet),
                "median_signed_pct": round(statistics.median(indet), 2) if indet else None,
                "note": "excluded from both sides; not evidence in either direction",
            },
            "pairs": pairs,
        },
        "claim_3_etf_supply": {
            "statement": (
                "Count of ETF-labelled tickers per day that clear PIT's price and "
                "volume floors on an allowed exchange — the population live's dead "
                "gate admitted. Excludes REITs, which live also filters and PIT "
                "does not, so this UNDERSTATES the contamination."
            ),
            "median_per_day": int(statistics.median(etf_per_day)) if etf_per_day else None,
            "days_measured": len(etf_per_day),
        },
        "conclusion": {
            "pre_fix_live_counts": "contaminated; unusable as a PIT baseline",
            "post_fix_live_counts": (
                f"{len(post)} clean observation(s) in the overlap; insufficient. "
                f"{len(indet)} date(s) INDETERMINATE (runs on both sides of the merge)."
            ),
            "section_A5_live_count_gate": "DEFERRED — neither passed nor failed",
            "dataset_acceptance": "HALTED",
        },
    }


# Every committed source, paired with the artifact field that records its hash.
# Driven by a table rather than ad-hoc lines: the first version verified the
# dashboard and merely PRINTED the provenance hash, which left a live hole —
# edit pipeline_run_provenance.json, the date classification changes, the
# artifact's recorded boundary hash goes stale, and CI stays green.
_VERIFIED_SOURCES = (
    ("outputs/research/evidence/source/dashboard_minimal.json",
     ("source", "sha256")),
    ("outputs/research/evidence/source/pipeline_run_provenance.json",
     ("boundary", "source_sha256")),
)


def _recorded(artifact: dict, path: tuple[str, ...]) -> str:
    node = artifact
    for key in path:
        node = node[key]
    return node


def check_sources() -> int:
    """Compare every committed source against the hash the artifact records."""
    artifact = json.loads(ARTIFACT.read_text())
    repo = Path(__file__).resolve().parent.parent
    ok = True

    declared = set(artifact["source_closure"]["committed"])
    verified = {rel for rel, _ in _VERIFIED_SOURCES}
    if declared != verified:
        print(f"  UNVERIFIED SOURCE(S): {sorted(declared - verified)}")
        print("  every declared-committed source must have a recorded hash to check")
        ok = False

    for rel, field in _VERIFIED_SOURCES:
        path = repo / rel
        if not path.exists():
            print(f"  MISSING   {rel}")
            ok = False
            continue
        actual = _sha256_file(path)
        expected = _recorded(artifact, field)
        if actual == expected:
            print(f"  OK        {actual[:16]}  {rel}")
        else:
            print(f"  MISMATCH  {rel}")
            print(f"            artifact[{'.'.join(field)}] = {expected[:16]}")
            print(f"            actual bytes             = {actual[:16]}")
            ok = False
    return 0 if ok else 1


CERTIFICATION_BLOCKED = "CERTIFICATION_BLOCKED: incomplete_source_closure"


def certify() -> int:
    """Fail closed while the evidence chain is incomplete.

    Prose saying "provisional" is not a gate. Until the vintage's raw closure is
    available, certification must be mechanically UNAVAILABLE — this exits
    non-zero so any pipeline that tries to treat the artifact as
    acceptance-grade stops, rather than relying on a reader noticing a caveat.
    A skipped regeneration test is not a fail-closed gate either; it is silence.
    """
    artifact = json.loads(ARTIFACT.read_text())
    closure = artifact["source_closure"]

    if not closure.get("complete"):
        print(CERTIFICATION_BLOCKED)
        print(f"  missing: {closure.get('missing')}")
        print(f"  blocker: {closure.get('blocker')}")
        return 2

    if check_sources() != 0:
        print("CERTIFICATION_BLOCKED: source_hash_mismatch")
        return 2

    print("sources verified and closure complete — regeneration required to certify")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintage", default="2026-08-12")
    ap.add_argument("--check-sources", action="store_true",
                    help="verify committed source bytes match their recorded hashes")
    ap.add_argument("--certify", action="store_true",
                    help="fail closed unless the evidence chain is complete")
    ap.add_argument("--verify", action="store_true",
                    help="regenerate and diff against the committed artifact")
    args = ap.parse_args()

    if args.check_sources:
        raise SystemExit(check_sources())

    if args.certify:
        raise SystemExit(certify())

    evidence = build_evidence(args.vintage)
    blob = json.dumps(evidence, indent=2, sort_keys=True) + "\n"

    if args.verify:
        if not ARTIFACT.exists():
            raise SystemExit(f"no committed artifact at {ARTIFACT}")
        committed = ARTIFACT.read_text()
        if committed == blob:
            print(f"VERIFIED — regenerated output is byte-identical\n"
                  f"  sha256 {hashlib.sha256(blob.encode()).hexdigest()}")
            return
        raise SystemExit(
            "MISMATCH — the committed artifact does not match a regeneration from "
            "the frozen source. Either the source changed or the generator did."
        )

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(blob)
    print(f"wrote {ARTIFACT.relative_to(Path(__file__).resolve().parent.parent)}")
    print(f"  artifact sha256 {hashlib.sha256(blob.encode()).hexdigest()}")
    print(f"  source   sha256 {evidence['source']['sha256']}")


if __name__ == "__main__":
    main()
