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

# The definition change that partitions the comparison window. Recorded as data,
# not prose, so the boundary rule is auditable alongside the numbers it produces.
# Retrieved with: gh pr view 63 --json number,title,mergedAt,mergeCommit
BOUNDARY = {
    "pr": 63,
    "title": "Actually exclude ETFs from the universe",
    "merged_at_utc": "2026-08-11T11:25:10Z",
    "merge_commit": "c14d6d6f777b058e18fa8e4440ab5c9b3095d7d2",
    "rule": (
        "An ET market date D is POST-fix iff D >= 2026-08-11. The merge landed "
        "11:25Z = 07:25 ET, before that session's morning pipeline, so 2026-08-11 "
        "is the first run that could apply the gate."
    ),
}

EXCLUDED_TYPES = ("ETF", "ETN", "FUND")
CUTOVER = date(2026, 8, 11)


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

    live_path = ROOT / vintage / "raw" / "live" / "dashboard.json.gz"
    if not live_path.exists():
        raise SystemExit(f"no live snapshot at {live_path}")

    live = _read_gz(live_path)
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
                "post_fix": d >= CUTOVER,
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
            "post_fix": d >= CUTOVER,
        })
    pairs.sort(key=lambda p: p["date"])
    pre = [p["signed_pct"] for p in pairs if not p["post_fix"]]
    post = [p["signed_pct"] for p in pairs if p["post_fix"]]

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
            "path": str(live_path.relative_to(ROOT.parent.parent)),
            "sha256": _sha256_file(live_path),
            "bytes": live_path.stat().st_size,
            "origin": "https://raysyhuang.github.io/multi-agentic-screener/data.json",
            "note": (
                "Public dashboard export, frozen into the vintage's raw tree and "
                "hashed into manifest.json. Contains no positions, prices or P&L."
            ),
        },
        "generator": {
            "script": "scripts/pit_live_reference_audit.py",
            "sha256": _sha256_file(Path(__file__).resolve()),
            "command": f"python scripts/pit_live_reference_audit.py --vintage {vintage}",
            "deterministic": True,
        },
        "boundary": BOUNDARY,
        "claim_1_etf_candidates": {
            "statement": (
                "ETF/FUND/ETN names appeared as ranked LIVE candidates only before #63."
            ),
            "total": len(etf_rows),
            "before_fix": sum(1 for r in etf_rows if not r["post_fix"]),
            "on_or_after_fix": sum(1 for r in etf_rows if r["post_fix"]),
            "date_range": [etf_rows[0]["date"], etf_rows[-1]["date"]] if etf_rows else None,
            "rows": etf_rows,
        },
        "claim_2_divergence_split": {
            "statement": (
                "Median PIT-vs-live signed count divergence inverts at the boundary."
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
            "post_fix_live_counts": f"only {len(post)} observation(s); insufficient",
            "section_A5_live_count_gate": "DEFERRED — neither passed nor failed",
            "dataset_acceptance": "HALTED",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintage", default="2026-08-12")
    ap.add_argument("--verify", action="store_true",
                    help="regenerate and diff against the committed artifact")
    args = ap.parse_args()

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
