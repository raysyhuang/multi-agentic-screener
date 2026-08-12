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

    from pit_universe_phase_a import _classification_by_month  # noqa: PLC0415

    labels_by_month = _classification_by_month(vintage)
    per_month: dict[str, dict] = defaultdict(lambda: {
        "sampled": 0, "type_disagree": 0, "exchange_disagree": 0,
        "false_exclusion": 0, "contamination": 0,
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

            held_label = held.get(ticker) or {}
            held_type = held_label.get("type")
            actual_type = actual.get("type")
            if actual_type and held_type != actual_type:
                rec["type_disagree"] += 1
                # Direction matters: one contaminates, one silently shrinks.
                if held_type == "CS" and actual_type != "CS":
                    rec["contamination"] += 1
                elif held_type != "CS" and actual_type == "CS":
                    rec["false_exclusion"] += 1

            from pit_universe_phase_a import _EXCHANGE_MAP  # noqa: PLC0415
            actual_exch = _EXCHANGE_MAP.get(actual.get("primary_exchange", ""), "")
            if actual_exch and held_label.get("exchange") != actual_exch:
                rec["exchange_disagree"] += 1

    return {"ran": True, "per_month": dict(per_month)}


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

    halts = []
    if type_unknown_pct > HALT_TYPE_UNKNOWN_PCT:
        halts.append(f"type_unknown {type_unknown_pct:.2f}% > {HALT_TYPE_UNKNOWN_PCT}%")
    if exch_unknown_pct > HALT_EXCHANGE_UNKNOWN_PCT:
        halts.append(f"exchange_unknown {exch_unknown_pct:.2f}% > {HALT_EXCHANGE_UNKNOWN_PCT}%")

    audit = _audit_results(vintage)
    if audit.get("ran"):
        for month, rec in sorted(audit["per_month"].items()):
            if rec["type_disagree"]:
                halts.append(
                    f"{month}: {rec['type_disagree']} security-type disagreement(s) "
                    f"(contamination={rec['contamination']}, "
                    f"false_exclusion={rec['false_exclusion']}) — zero tolerated"
                )
            if rec["sampled"]:
                exch_pct = 100.0 * rec["exchange_disagree"] / rec["sampled"]
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
        "unknown_rates_pct": {
            "type_unknown": round(type_unknown_pct, 4),
            "exchange_unknown": round(exch_unknown_pct, 4),
        },
        "atr_pct_quantiles_diagnostic_only": quantiles,
        "classification_audit": audit,
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
