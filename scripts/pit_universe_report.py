"""Phase A normalization, manifest and diagnostic report.

Contract §7 (diagnostics), §5 (manifest/replay), §A.5 (halt thresholds).

Produces evidence, not conclusions. Nothing here computes a strategy result, and
the ATR distribution is REPORTED for comparison — never used to select
membership (§2, §9).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger("pit.report")

ROOT = Path(__file__).resolve().parent.parent / "outputs" / "pit_universe"

# §A.5 halt thresholds. Breaching any of these blocks research consumption.
HALT_TYPE_UNKNOWN_PCT = 1.0
HALT_EXCHANGE_UNKNOWN_PCT = 1.0
HALT_DRIFT_EXCHANGE_PCT = 0.5


def _calendar_provenance() -> dict:
    """Which calendar decided the session list.

    Recorded because it is a dataset input, not a build detail: a different
    version that revised a historical holiday would produce a different set of
    dates from the same code and the same API, so a vintage is only replayable
    against a known calendar version.
    """
    try:
        import pandas_market_calendars as mcal  # noqa: PLC0415

        return {"library": "pandas_market_calendars", "version": mcal.__version__}
    except ImportError:
        return {"library": "pandas_market_calendars", "version": None,
                "note": "not installed — session list unverifiable"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_raw(path: Path) -> dict:
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read())


def _atr_pct_by_ticker(vintage: str, sample_days: int = 60) -> dict[str, float]:
    """ATR(14)% over the most recent sessions, as a DIAGNOSTIC only.

    Reported so the cache's volatility profile can be compared with live
    observations. It never influences membership — conditioning the universe on
    ATR is the selection bias §2 exists to prevent.
    """
    grouped = sorted((ROOT / vintage / "raw" / "grouped").glob("*.json.gz"))[-sample_days:]
    bars: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for path in grouped:
        for bar in _read_raw(path).get("results", []) or []:
            t, h, low, c = bar.get("T"), bar.get("h"), bar.get("l"), bar.get("c")
            if t and h is not None and low is not None and c is not None:
                bars[t].append((h, low, c))

    out: dict[str, float] = {}
    for ticker, series in bars.items():
        if len(series) < 15:
            continue
        trs = []
        for i in range(1, len(series)):
            h, low, _ = series[i]
            prev_close = series[i - 1][2]
            trs.append(max(h - low, abs(h - prev_close), abs(low - prev_close)))
        atr = statistics.fmean(trs[-14:])
        last_close = series[-1][2]
        if last_close > 0:
            out[ticker] = 100.0 * atr / last_close
    return out


def _audit_results(vintage: str) -> dict:
    """Compare date-specific classification against the forward-held label (§3b)."""
    audit_dir = ROOT / vintage / "raw" / "audit"
    if not audit_dir.exists():
        return {"ran": False}

    from pit_universe_phase_a import (  # noqa: PLC0415
        ALLOWED_EXCHANGES as ALLOWED,
        _classification_by_month,
    )

    labels_by_month = _classification_by_month(vintage)
    per_month: dict[str, dict] = defaultdict(lambda: {
        "sampled": 0, "verifiable": 0, "unverifiable": 0,
        # Only pairs whose forward-held label EXISTS can test drift. A pair with
        # no held label is a different fact and is counted separately: treating
        # absent as a differing value made every unknown-bucket sample look like
        # a drift disagreement, which is 100% false positives.
        "labelled": 0, "type_disagree": 0, "exchange_disagree": 0,
        "false_exclusion": 0, "contamination": 0,
        # An exchange disagreement only matters if it crosses the eligible set.
        # LNG on 2024-02-22 was held as AMEX and is actually NYSE: common stock,
        # wrongly excluded. Tracking drift without tracking whether it changed
        # membership reports noise and misses the one case that counts.
        "exchange_membership_flip": 0,
        "resolvable_unknown": 0,
    })

    for month_dir in sorted(audit_dir.glob("*")):
        y, m = (int(x) for x in month_dir.name.split("-"))
        held = labels_by_month.get((y, m), {})
        for path in sorted(month_dir.glob("*.json.gz")):
            payload = _read_raw(path)
            meta = payload.get("_audit", {})
            actual = payload.get("results") or {}
            ticker = meta.get("ticker")
            rec = per_month[month_dir.name]
            rec["sampled"] += 1
            if payload.get("_not_found") or not actual:
                # No date-specific record. Neither agreement nor disagreement —
                # counting it as agreement would understate drift, which is the
                # direction that lets a bad cadence pass.
                rec["unverifiable"] += 1
                continue
            rec["verifiable"] += 1

            from pit_universe_phase_a import _EXCHANGE_MAP  # noqa: PLC0415

            held_label = held.get(ticker) or {}
            held_type = held_label.get("type")
            actual_type = actual.get("type")
            actual_exch = _EXCHANGE_MAP.get(actual.get("primary_exchange", ""), "")

            if not held_type:
                # The monthly snapshot carried no label for this ticker, so it
                # was excluded as type_unknown. That the per-ticker endpoint DOES
                # resolve it is worth reporting — the snapshot is incomplete
                # relative to it — but it is not classification DRIFT, which is
                # what the cadence is on trial for.
                if actual_type:
                    rec["resolvable_unknown"] += 1
                continue

            rec["labelled"] += 1
            if actual_type and held_type != actual_type:
                rec["type_disagree"] += 1
                # Direction matters: one contaminates, one silently shrinks.
                if held_type == "CS" and actual_type != "CS":
                    rec["contamination"] += 1
                elif actual_type == "CS":
                    rec["false_exclusion"] += 1
            if actual_exch and held_label.get("exchange") != actual_exch:
                rec["exchange_disagree"] += 1
                held_ok = held_label.get("exchange") in ALLOWED
                actual_ok = actual_exch in ALLOWED
                if held_ok != actual_ok and (actual_type or held_type) == "CS":
                    rec["exchange_membership_flip"] += 1

    return {"ran": True, "per_month": dict(per_month)}


def unknown_rate_gates(membership: dict) -> tuple[dict, list[str]]:
    """Per-month and trailing-12-month unknown rates (§A.5).

    The pooled rate this replaces was unfit for purpose: across 751 sessions a
    single month in which EVERY ticker was unclassifiable contributes ~2.7% to
    the pooled figure, so a total classification outage for a month lands under
    a 1% pooled threshold and passes. The gate must bind on the window in which
    the damage is concentrated, so it is evaluated per month, and again on a
    trailing 12-month window to catch slow degradation that no single month
    breaches.
    """
    by_month: dict[str, dict[str, float]] = defaultdict(
        lambda: {"pre": 0, "type_unknown": 0, "exchange_unknown": 0}
    )
    for d in sorted(membership):
        rec = membership[d]
        key = f"{d.year:04d}-{d.month:02d}"
        by_month[key]["pre"] += len(rec["pre_classification"])
        by_month[key]["type_unknown"] += rec["exclusions"].get("type_unknown", 0)
        by_month[key]["exchange_unknown"] += rec["exclusions"].get("exchange_unknown", 0)

    months = sorted(by_month)
    per_month, halts = {}, []
    for m in months:
        r = by_month[m]
        denom = max(1, r["pre"])
        t_pct = 100.0 * r["type_unknown"] / denom
        e_pct = 100.0 * r["exchange_unknown"] / denom
        per_month[m] = {
            "pre_classification": r["pre"],
            "type_unknown_pct": round(t_pct, 4),
            "exchange_unknown_pct": round(e_pct, 4),
        }
        if t_pct > HALT_TYPE_UNKNOWN_PCT:
            halts.append(f"{m}: type_unknown {t_pct:.2f}% > {HALT_TYPE_UNKNOWN_PCT}% (monthly)")
        if e_pct > HALT_EXCHANGE_UNKNOWN_PCT:
            halts.append(
                f"{m}: exchange_unknown {e_pct:.2f}% > {HALT_EXCHANGE_UNKNOWN_PCT}% (monthly)"
            )

    trailing = {}
    for i in range(len(months)):
        window = months[max(0, i - 11): i + 1]
        if len(window) < 12:
            continue
        pre = sum(by_month[m]["pre"] for m in window)
        t_pct = 100.0 * sum(by_month[m]["type_unknown"] for m in window) / max(1, pre)
        trailing[months[i]] = round(t_pct, 4)
        if t_pct > HALT_TYPE_UNKNOWN_PCT:
            halts.append(f"{months[i]}: trailing-12m type_unknown {t_pct:.2f}% > {HALT_TYPE_UNKNOWN_PCT}%")

    return {"per_month": per_month, "trailing_12m_type_unknown_pct": trailing}, halts


def write_report(vintage: str) -> dict:
    from pit_universe_phase_a import build_membership  # noqa: PLC0415

    base = ROOT / vintage
    membership = build_membership(vintage)
    if not membership:
        raise SystemExit(f"no grouped data under {base}/raw/grouped — run `spine` first")

    dates = sorted(membership)
    distinct_eligible: set[str] = set()
    daily_counts, exclusion_totals = [], Counter()
    for d in dates:
        rec = membership[d]
        distinct_eligible.update(rec["eligible_pre_mcap"])
        daily_counts.append({
            "date": str(d),
            "traded": len(rec["traded"]),
            "pre_classification": len(rec["pre_classification"]),
            "eligible_pre_mcap": len(rec["eligible_pre_mcap"]),
        })
        exclusion_totals.update(rec["exclusions"])

    # Unknown rates against the pre-classification denominator (§A.5).
    pre_total = sum(c["pre_classification"] for c in daily_counts)
    type_unknown_pct = 100.0 * exclusion_totals.get("type_unknown", 0) / max(1, pre_total)
    exch_unknown_pct = 100.0 * exclusion_totals.get("exchange_unknown", 0) / max(1, pre_total)

    atr = _atr_pct_by_ticker(vintage)
    eligible_atr = sorted(v for t, v in atr.items() if t in distinct_eligible)
    quantiles = {}
    if eligible_atr:
        for q in (10, 25, 50, 75, 90, 95):
            idx = min(len(eligible_atr) - 1, int(len(eligible_atr) * q / 100))
            quantiles[f"p{q}"] = round(eligible_atr[idx], 3)
        quantiles["share_atr_ge_5pct"] = round(
            100.0 * sum(1 for v in eligible_atr if v >= 5.0) / len(eligible_atr), 2
        )

    # Pooled rates are REPORTED for continuity but no longer gate anything —
    # they cannot see a single catastrophic month (see unknown_rate_gates).
    windowed, halts = unknown_rate_gates(membership)

    ledger_path = base / "request_ledger.jsonl"
    ledger_summary = {"present": ledger_path.exists()}
    if ledger_path.exists():
        calls = failures = 0
        for line in ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line).get("event")
            except json.JSONDecodeError:
                continue
            if event == "request":
                calls += 1
            elif event == "failure":
                failures += 1
        ledger_summary = {"present": True, "calls": calls, "durable_failures": failures}
        if failures:
            # A vintage with unrecovered holes is incomplete by construction;
            # reporting it as a dataset would present a partial universe as a
            # whole one, which is the silent-truncation failure mode.
            halts.append(f"request ledger records {failures} unrecovered failure(s)")
    else:
        halts.append("no request ledger — provenance of this vintage is unverifiable")

    divergence, divergence_halts = live_divergence(vintage)
    halts.extend(divergence_halts)

    audit = _audit_results(vintage)
    if audit.get("ran"):
        for month, rec in sorted(audit["per_month"].items()):
            if rec["type_disagree"]:
                halts.append(
                    f"{month}: {rec['type_disagree']} security-type disagreement(s) "
                    f"(contamination={rec['contamination']}, "
                    f"false_exclusion={rec['false_exclusion']}) — zero tolerated"
                )
            if rec["labelled"]:
                exch_pct = 100.0 * rec["exchange_disagree"] / rec["labelled"]
                if exch_pct > HALT_DRIFT_EXCHANGE_PCT:
                    halts.append(f"{month}: exchange drift {exch_pct:.2f}% > {HALT_DRIFT_EXCHANGE_PCT}%")

    raw_files = sorted((base / "raw").rglob("*.json.gz"))
    manifest = {
        "vintage": vintage,
        "timezone": "America/New_York",
        "date_range_et": [str(dates[0]), str(dates[-1])],
        "sessions": len(dates),
        "phase": "A",
        "constraints_applied": {
            "min_price": 5.0, "min_share_volume": 500_000,
            "exchanges": ["NYSE", "NASDAQ"], "type": "CS",
            "market_cap": "NOT APPLIED — Phase B",
        },
        "classification_policy": "forward-held monthly, applied from snapshot date only",
        "raw_file_count": len(raw_files),
        "raw_hashes": {str(p.relative_to(base)): _sha256(p) for p in raw_files},
        "distinct_eligible_tickers_pre_mcap": len(distinct_eligible),
        "exclusion_totals": dict(exclusion_totals),
        "unknown_rates_pct_pooled_DIAGNOSTIC_ONLY": {
            "type_unknown": round(type_unknown_pct, 4),
            "exchange_unknown": round(exch_unknown_pct, 4),
        },
        "unknown_rate_gates": windowed,
        "request_ledger": ledger_summary,
        "atr_pct_quantiles_diagnostic_only": quantiles,
        "classification_audit": audit,
        "live_universe_divergence": divergence,
        "calendar": _calendar_provenance(),
        "halts": halts,
        "phase_b_gate": {
            "distinct_tickers": len(distinct_eligible),
            "quarters": 12,
            "projected_detail_calls": len(distinct_eligible) * 12,
            "ceiling": 75_000,
            "within_ceiling": len(distinct_eligible) * 12 <= 75_000,
        },
    }

    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (base / "daily_counts.json").write_text(json.dumps(daily_counts, indent=2))

    print(json.dumps({k: v for k, v in manifest.items() if k != "raw_hashes"}, indent=2))
    return manifest


def verify(vintage: str, manifest_path: Path | None = None) -> dict:
    """Recompute every raw hash against a manifest and report discrepancies.

    This is what makes the replay claim checkable rather than asserted. The
    manifest is committed to the repo while `outputs/` is gitignored, so the
    authority for "these are the bytes the dataset was built from" lives in git
    history, and a vintage restored from a Release archive can be proven
    identical to the one that produced the accepted result.

    Exits non-zero on any discrepancy: a verification that reports problems and
    returns success is decorative.
    """
    base = ROOT / vintage
    src = manifest_path or (base / "manifest.json")
    if not src.exists():
        raise SystemExit(f"no manifest at {src} — nothing to verify against")

    expected = json.loads(src.read_text()).get("raw_hashes", {})
    if not expected:
        raise SystemExit(f"{src} carries no raw_hashes")

    present = {
        str(p.relative_to(base)): p
        for p in sorted((base / "raw").rglob("*.json.gz"))
    }

    missing = sorted(set(expected) - set(present))
    extra = sorted(set(present) - set(expected))
    mismatched = [
        rel for rel in sorted(set(expected) & set(present))
        if _sha256(present[rel]) != expected[rel]
    ]

    result = {
        "vintage": vintage,
        "manifest": str(src),
        "files_expected": len(expected),
        "files_present": len(present),
        "missing": missing[:20],
        "missing_count": len(missing),
        "extra": extra[:20],
        "extra_count": len(extra),
        "mismatched": mismatched[:20],
        "mismatched_count": len(mismatched),
        "verified": not (missing or extra or mismatched),
    }
    print(json.dumps(result, indent=2))
    if not result["verified"]:
        raise SystemExit(
            f"VERIFY FAILED: {len(missing)} missing, {len(extra)} extra, "
            f"{len(mismatched)} mismatched"
        )
    logger.info("verified %d raw files against %s", len(expected), src)
    return result


def package(vintage: str) -> Path:
    """Build the Release archive and stamp its own hash.

    Produces the artifact only. Uploading it is a deliberate, separately
    authorised act: a vintage contains a full market history and publishing is
    not reversible, so this never calls `gh release` on its own.
    """
    import tarfile

    base = ROOT / vintage
    archive = base.parent / f"pit-universe-{vintage}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(base / "raw", arcname=f"{vintage}/raw")
        for extra in ("manifest.json", "request_ledger.jsonl", "daily_counts.json"):
            if (base / extra).exists():
                tar.add(base / extra, arcname=f"{vintage}/{extra}")

    digest = _sha256(archive)
    (base / "archive.sha256").write_text(f"{digest}  {archive.name}\n")
    size_mb = archive.stat().st_size / (1 << 20)
    logger.info("archive %s (%.1f MB) sha256=%s", archive.name, size_mb, digest)
    print(json.dumps({
        "archive": str(archive), "sha256": digest, "size_mb": round(size_mb, 1),
        "upload": (
            f"gh release create pit-universe-{vintage} '{archive}' "
            f"--title 'PIT universe vintage {vintage}' --notes-file "
            f"outputs/research/PIT_PHASE_A_HALT_FINDINGS.md"
        ),
    }, indent=2))
    return archive


def live_divergence(vintage: str) -> tuple[dict, list[str]]:
    """Compare PIT membership against what the LIVE pipeline actually saw.

    The §3b audit tests PIT against a vendor endpoint. This tests it against
    production, which is the only source that can show the universe is too
    NARROW in a way that mattered — a name the live book ranked, on a date PIT
    says it was ineligible, is a false exclusion with a real consequence.

    Direction is attributed three ways, because "absent from PIT" has three very
    different meanings and collapsing them hides the important one:

      pit_false_exclusion  PIT itself labels it CS on an allowed exchange, so PIT
                           contradicts its own constraints — a real defect.
      pit_stricter_than_live
                           PIT excludes it on a constraint LIVE DOES NOT HAVE.
                           `type == "CS"` is the case: src/signals/filter.py
                           gates exchange, ETF/fund flags, price, volume and
                           market cap, but never requires common stock. So PIT
                           silently drops ADRs the book actually trades.
      explained_by_live_gates
                           PIT excludes it and live should have too — the ETF
                           gate was dead until #63 (TQQQ ranked 97.5). Here PIT
                           is right and live was wrong.

    The middle bucket is the one worth the check. Lumping ADRs in with ETFs, as
    the first version of this function did, reports a PIT over-restriction as a
    live defect and inverts the conclusion.
    """
    base = ROOT / vintage
    live_path = base / "raw" / "live" / "dashboard.json.gz"
    if not live_path.exists():
        return {"ran": False, "reason": "no live snapshot — run `divergence-fetch`"}, [
            "live-universe divergence check has no snapshot to run against"
        ]

    from pit_universe_phase_a import _classification_by_month, build_membership  # noqa: PLC0415

    live = _read_raw(live_path)
    membership = build_membership(vintage)
    labels = _classification_by_month(vintage)
    covered = set(membership)

    from datetime import date as _date  # noqa: PLC0415

    pit_false_exclusions, pit_stricter, live_gate_misses = [], [], []
    out_of_range = 0
    checked = 0
    for row in live.get("candidates") or []:
        ticker, run_date = row.get("ticker"), row.get("run_date")
        if not ticker or not run_date:
            continue
        try:
            d = _date.fromisoformat(str(run_date)[:10])
        except ValueError:
            continue
        if d not in covered:
            out_of_range += 1
            continue
        checked += 1
        if ticker in set(membership[d]["eligible_pre_mcap"]):
            continue
        # Absent from PIT. Attribute it.
        keys = [k for k in labels if k <= (d.year, d.month)]
        held = (labels[max(keys)].get(ticker) or {}) if keys else {}
        record = {
            "ticker": ticker, "date": str(d), "model": row.get("model"),
            "picked": row.get("picked"), "rank": row.get("rank"),
            "held_type": held.get("type"), "held_exchange": held.get("exchange"),
            "traded_that_day": ticker in set(membership[d]["traded"]),
        }
        held_type = held.get("type")
        on_allowed_exchange = held.get("exchange") in ("NYSE", "NASDAQ")
        if held_type == "CS" and on_allowed_exchange:
            pit_false_exclusions.append(record)
        elif held_type in ("ETF", "FUND", "ETN"):
            live_gate_misses.append(record)
        else:
            # Everything else — ADRC above all — is PIT applying a constraint
            # live does not have.
            pit_stricter.append(record)

    result = {
        "ran": True,
        "live_candidates_in_range": checked,
        "out_of_vintage_range": out_of_range,
        "pit_false_exclusions": pit_false_exclusions[:25],
        "pit_false_exclusion_count": len(pit_false_exclusions),
        "pit_stricter_than_live": pit_stricter[:25],
        "pit_stricter_than_live_count": len(pit_stricter),
        "pit_stricter_picked_count": sum(1 for r in pit_stricter if r.get("picked")),
        "pit_stricter_types": sorted({r.get("held_type") for r in pit_stricter}),
        "pit_stricter_tickers": sorted({r["ticker"] for r in pit_stricter}),
        "explained_by_live_gates": live_gate_misses[:25],
        "explained_by_live_gates_count": len(live_gate_misses),
        "explained_by_live_gates_tickers": sorted({r["ticker"] for r in live_gate_misses}),
    }
    halts = []
    if pit_false_exclusions:
        halts.append(
            f"live divergence: {len(pit_false_exclusions)} live candidate(s) that PIT "
            f"itself labels CS on NYSE/NASDAQ were absent from the eligible set"
        )
    if pit_stricter:
        picked = sum(1 for r in pit_stricter if r.get("picked"))
        halts.append(
            f"live divergence: PIT excludes {len(pit_stricter)} live candidate-day(s) "
            f"({picked} actually PICKED) on constraints live does not apply — "
            f"types {sorted({r.get('held_type') for r in pit_stricter})}"
        )
    return result, halts
