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
# R2 (contract §12): with §3a-v2 transition resolution no legitimate mechanism
# produces a drift disagreement on either axis, so both are zero tolerance. The
# former 0.5%/month exchange limit was unresolvable at ~134 labelled pairs/month
# (the smallest non-zero rate expressible is 0.75%).
HALT_DRIFT_EXCHANGE_DISAGREEMENTS = 0
HALT_DRIFT_EXCHANGE_PCT_V2 = 0.5   # v2 vintages replay their original rule
# §A.5: PIT daily count vs contemporaneous live eligible count.
HALT_LIVE_COUNT_DIVERGENCE_PCT = 15.0
# §A.5-v2 / R5: the live-count gate is evaluated only over a window in which no
# universe-definition change merged, and only with at least this many clean
# live observations. (a) is set by the independent verifier, not by the author
# who has seen the vintage; until it is set the gate reports DEFERRED.
LIVE_GATE_MIN_CLEAN_OBS: int = 60   # R5(a), set by Codex 2026-10-06: ~one quarter of daily runs
# R5 (b): a merge touching any of these is a universe-definition change.
UNIVERSE_DEFINITION_PATHS = (
    "src/signals/filter.py",
    "src/data/fmp_client.py",
    "src/data/universe_selection.py",
)


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


def _eligible_types(vintage: str):
    from pit_universe_phase_a import eligible_types_for  # noqa: PLC0415
    return eligible_types_for(vintage)


def _contract_version(vintage: str) -> str:
    from pit_universe_phase_a import contract_version  # noqa: PLC0415
    return contract_version(vintage)


HALT_MCAP_UNKNOWN_PCT = 5.0       # §A.5
HALT_MCAP_BAND_DISAGREE_PCT = 2.0  # §3d part 1, per month


def mcap_gates(vintage: str, membership: dict) -> tuple[dict, list[str]]:
    """§A.5 market-cap-unknown rate and the §3d threshold audit (band + sentinel)."""
    from pit_universe_phase_a import (  # noqa: PLC0415
        MIN_MCAP, expected_mcap_plan, mcap_audit_sample,
    )

    halts: list[str] = []
    per_month: dict[str, dict] = defaultdict(lambda: {"pre_mcap": 0, "mcap_unknown": 0})
    for d, rec in membership.items():
        m = per_month[f"{d.year:04d}-{d.month:02d}"]
        m["pre_mcap"] += len(rec["eligible_pre_mcap"])
        m["mcap_unknown"] += rec["exclusions"].get("mcap_unknown", 0)
    for month, m in sorted(per_month.items()):
        rate = 100.0 * m["mcap_unknown"] / max(1, m["pre_mcap"])
        m["mcap_unknown_pct"] = round(rate, 4)
        if rate > HALT_MCAP_UNKNOWN_PCT:
            halts.append(f"{month}: market-cap unknown {rate:.2f}% > {HALT_MCAP_UNKNOWN_PCT}%")

    audit: dict = {"ran": False}
    plan_path = ROOT / vintage / "mcap_audit_plan.json"
    if not plan_path.exists():
        halts.append("§3d threshold audit has not run — phase B not accepted")
    else:
        plan = json.loads(plan_path.read_text())
        sample = mcap_audit_sample(vintage)
        expected_rows = {f"{m}/{part}/{t}_{d}": (part, str(d), t, v)
                         for m, rows in sample.items() for part, d, t, v in rows}
        if plan != expected_mcap_plan(vintage, sample):
            halts.append("§3d audit plan is not exactly the plan the current estimates and sampler "
                         "produce — rerun on a new vintage")
        else:
            root = ROOT / vintage / "raw" / "mcap_audit"
            observed = {f"{p.parent.parent.name}/{p.parent.name}/{p.name.replace('.json.gz', '')}"
                        for p in root.glob("*/*/*.json.gz")} if root.exists() else set()
            planned = set(plan["pairs"])
            if observed != planned:
                halts.append(f"§3d audit set != plan ({len(planned - observed)} missing, "
                             f"{len(observed - planned)} unplanned)")
            else:
                months: dict[str, dict] = defaultdict(lambda: {"band": 0, "band_disagree": 0,
                                                               "sentinel": 0, "sentinel_flip": 0,
                                                               "unverifiable": 0})
                for p in sorted(root.glob("*/*/*.json.gz")):
                    payload = _read_raw(p)
                    meta = payload.get("_audit", {})
                    key = f"{p.parent.parent.name}/{p.parent.name}/{p.name.replace('.json.gz', '')}"
                    part, d_s, t, v = expected_rows[key]
                    if (meta.get("part"), meta.get("date"), meta.get("ticker")) != (part, d_s, t) \
                            or meta.get("estimate") != v:
                        # Exact: the stored estimate must BE the recomputed one, so no
                        # tolerance can straddle the $300M threshold.
                        halts.append(f"§3d audit record {key} does not match its planned pair")
                        continue
                    actual = (payload.get("results") or {}).get("market_cap")
                    rec = months[p.parent.parent.name]
                    if payload.get("_not_found") or actual is None:
                        rec["unverifiable"] += 1
                        continue
                    flipped = (v > MIN_MCAP) != (float(actual) > MIN_MCAP)   # recomputed estimate
                    if meta["part"] == "band":
                        rec["band"] += 1
                        rec["band_disagree"] += int(flipped)
                    else:
                        rec["sentinel"] += 1
                        rec["sentinel_flip"] += int(flipped)
                for month, rec in sorted(months.items()):
                    if rec["band"]:
                        pct = 100.0 * rec["band_disagree"] / rec["band"]
                        rec["band_disagree_pct"] = round(pct, 2)
                        if pct > HALT_MCAP_BAND_DISAGREE_PCT:
                            halts.append(f"{month}: §3d band disagreement {pct:.1f}% > "
                                         f"{HALT_MCAP_BAND_DISAGREE_PCT}%")
                    if rec["sentinel_flip"]:
                        halts.append(f"{month}: §3d sentinel flip ({rec['sentinel_flip']}) — zero tolerated")
                    if rec["unverifiable"]:
                        # Fail closed: an unanswered pair is not evidence of agreement,
                        # and the contract defines no replacement policy.
                        halts.append(f"{month}: {rec['unverifiable']} §3d pair(s) unverifiable "
                                     "(no as-of market_cap) — audit not passed")
                audit = {"ran": True, "per_month": dict(months)}
    return {"mcap_unknown_by_month": dict(per_month), "threshold_audit": audit}, halts


def _code_sha() -> str | None:
    import subprocess
    for exe in ("git", "/opt/homebrew/bin/git"):
        try:
            out = subprocess.run([exe, "-C", str(ROOT.parent.parent), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0:
            return out.stdout.strip()
    return None


def _audit_results(vintage: str) -> dict:
    """Compare date-specific classification against the forward-held label (§3b)."""
    audit_dir = ROOT / vintage / "raw" / "audit"
    if not audit_dir.exists():
        return {"ran": False}

    from pit_universe_phase_a import (  # noqa: PLC0415
        ALLOWED_EXCHANGES as ALLOWED,
        _classification_by_month,
        eligible_types_for,
        resolved_label,
        resolved_overrides,
    )

    ELIGIBLE_TYPES = eligible_types_for(vintage)

    labels_by_month = _classification_by_month(vintage)
    overrides = resolved_overrides(vintage, labels_by_month)
    planned: set[str] | None = None
    from pit_universe_phase_a import contract_version, overrides_fingerprint  # noqa: PLC0415
    if contract_version(vintage) == "v3":
        plan_path = ROOT / vintage / "audit_plan.json"
        if not plan_path.exists():
            return {"ran": False, "reason": "no audit plan — the v3 audit has not run"}
        plan = json.loads(plan_path.read_text())
        from pit_universe_phase_a import SAMPLER_VERSION  # noqa: PLC0415
        if plan.get("sampler_version") != SAMPLER_VERSION:
            return {"ran": False, "stale": True,
                    "reason": f"audit plan sampler {plan.get('sampler_version')!r} != {SAMPLER_VERSION!r}"}
        if plan.get("overrides_sha256") != overrides_fingerprint(vintage):
            return {"ran": False, "stale": True,
                    "reason": "audit sample was drawn under different transition results"}
        planned = set(plan["pairs"])
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
        for path in sorted(month_dir.glob("*.json.gz")):
            if planned is not None and f"{month_dir.name}/{path.name.replace('.json.gz', '')}" not in planned:
                continue
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

            from datetime import date as _d  # noqa: PLC0415
            # The label under audit is the one membership actually used: the
            # forward-held monthly label after §3a-v2 transition overrides.
            held_label = resolved_label(labels_by_month, overrides, ticker,
                                        _d.fromisoformat(meta.get("date"))) or {}
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
                if held_type in ELIGIBLE_TYPES and actual_type not in ELIGIBLE_TYPES:
                    rec["contamination"] += 1
                elif actual_type in ELIGIBLE_TYPES and held_type not in ELIGIBLE_TYPES:
                    rec["false_exclusion"] += 1
            if actual_exch and held_label.get("exchange") != actual_exch:
                rec["exchange_disagree"] += 1
                held_ok = held_label.get("exchange") in ALLOWED
                actual_ok = actual_exch in ALLOWED
                if held_ok != actual_ok and (actual_type or held_type) in ELIGIBLE_TYPES:
                    rec["exchange_membership_flip"] += 1

    if planned is not None:
        observed = {f"{d.name}/{p.name.replace('.json.gz', '')}"
                    for d in audit_dir.glob("*") for p in d.glob("*.json.gz")}
        missing, extra = sorted(planned - observed), sorted(observed - planned)
        if missing or extra:
            return {"ran": False, "incomplete": bool(missing), "missing_count": len(missing),
                    "extra_count": len(extra),
                    "reason": f"audit set != plan: {len(missing)} planned pair(s) never observed "
                              f"(e.g. {missing[:3]}), {len(extra)} unplanned file(s) "
                              f"(e.g. {extra[:3]}) — complete the audit / quarantine extras"}
    return {"ran": True, "per_month": dict(per_month)}


def unknown_rate_gates(membership: dict, v3: bool = True) -> tuple[dict, list[str]]:
    """§A.5 unknown-rate gates, both of them, for both metrics.

    Two rules, and they catch different failures:

      absolute   monthly rate > 1%              — a bad month, on its own terms
      relative   monthly rate > 2x the trailing
                 12-month MEDIAN for that metric — a month that is out of
                                                   character for this dataset

    The pooled rate this replaces was unfit for either purpose: across 751
    sessions a single ruined month is diluted by 36 healthy ones. My first
    attempt then implemented the relative rule as "pooled trailing type rate vs
    a fixed 1%", which is neither the contract's statistic (median, not mean),
    nor its comparison (2x baseline, not a constant), nor its scope (it omitted
    exchange entirely). Writing a gate that is easier to compute than the one
    that was frozen is silently reinterpreting the contract.

    R6 (contract §12): when the trailing median is 0 — the normal state for
    `exchange_unknown` — the ratio is undefined, not maximally strict, so the
    absolute gate governs that month and the 2x rule applies only to a positive
    median. The three relative breaches the literal rule found (2025-05, 2025-09,
    2026-06, type axis) all had positive medians and stay caught.
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

    METRICS = (
        ("type_unknown", HALT_TYPE_UNKNOWN_PCT),
        ("exchange_unknown", HALT_EXCHANGE_UNKNOWN_PCT),
    )
    months = sorted(by_month)
    rates: dict[str, dict[str, float]] = {}
    for m in months:
        r = by_month[m]
        denom = max(1, r["pre"])
        rates[m] = {
            metric: 100.0 * r[metric] / denom for metric, _ in METRICS
        }

    per_month, halts = {}, []
    for i, m in enumerate(months):
        entry = {"pre_classification": by_month[m]["pre"]}
        for metric, absolute in METRICS:
            rate = rates[m][metric]
            entry[f"{metric}_pct"] = round(rate, 4)

            if rate > absolute:
                halts.append(f"{m}: {metric} {rate:.2f}% > {absolute}% (monthly absolute)")

            # Trailing 12 months, strictly PRIOR to m: including m in its own
            # baseline lets a bad month raise the bar it is judged against.
            window = [rates[k][metric] for k in months[max(0, i - 12): i]]
            if len(window) < 12:
                entry[f"{metric}_trailing_median_pct"] = None
                continue
            median = statistics.median(window)
            entry[f"{metric}_trailing_median_pct"] = round(median, 4)
            if (median > 0 or not v3) and rate > 2.0 * median:
                halts.append(
                    f"{m}: {metric} {rate:.4f}% > 2x trailing-12m median "
                    f"{median:.4f}% (relative)"
                )
        per_month[m] = entry

    rule = ("absolute >1%; relative >2x trailing-12m median when that median > 0 (R6)" if v3
            else "absolute >1% and relative >2x trailing-12m median")
    return {"per_month": per_month, "rule": rule}, halts


def live_count_divergence(vintage: str, membership: dict) -> tuple[dict, list[str]]:
    """§A.5 — PIT daily eligible COUNT vs the contemporaneous live count.

    Distinct from the per-candidate check in `live_divergence`, which asks
    whether specific names PIT excluded were traded live. This asks whether the
    universe is the right SIZE, and it is the gate the contract actually froze:
    median divergence over the overlapping dates above 15% halts.

    A universe can pass the per-name check and still fail this one — if PIT is
    uniformly half the size of live, no individual live candidate need be
    missing from it while the population is wrong.
    """
    base = ROOT / vintage
    from pit_universe_phase_a import contract_version  # noqa: PLC0415
    if contract_version(vintage) == "v3":
        return live_count_gate_v3(vintage, membership)
    live_path = base / "raw" / "live" / "dashboard.json.gz"
    if not live_path.exists():
        return {"ran": False}, ["live count divergence has no snapshot to run against"]

    live = _read_raw(live_path)
    from datetime import date as _date  # noqa: PLC0415

    pairs = []
    for row in live.get("run_history") or []:
        raw_date, live_n = row.get("date"), row.get("universe")
        if not raw_date or not live_n:
            continue
        try:
            d = _date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            continue
        if d not in membership:
            continue
        pit_n = len(membership[d]["eligible_pre_mcap"])
        pairs.append({
            "date": str(d), "pit": pit_n, "live": live_n,
            "divergence_pct": round(100.0 * abs(pit_n - live_n) / max(1, live_n), 2),
        })

    if not pairs:
        return {"ran": True, "overlap_dates": 0}, [
            "live count divergence: no overlapping dates — gate could not be evaluated"
        ]

    median_div = statistics.median(p["divergence_pct"] for p in pairs)
    result = {
        "ran": True,
        "overlap_dates": len(pairs),
        "median_divergence_pct": round(median_div, 2),
        "threshold_pct": HALT_LIVE_COUNT_DIVERGENCE_PCT,
        "sample": sorted(pairs, key=lambda p: -p["divergence_pct"])[:10],
    }
    halts = []
    if median_div > HALT_LIVE_COUNT_DIVERGENCE_PCT:
        halts.append(
            f"live count divergence: median {median_div:.1f}% > "
            f"{HALT_LIVE_COUNT_DIVERGENCE_PCT}% over {len(pairs)} overlapping date(s)"
        )
    return result, halts


def universe_definition_boundaries(repo: Path | None = None) -> list:
    """Dates on which a universe-definition change merged to main (R5 b).

    Detected from git history, never remembered by hand: any commit on main
    touching the definition paths. The live-count gate only trusts live runs
    strictly after the latest boundary.
    """
    import subprocess
    from datetime import date as _date  # noqa: PLC0415

    repo = repo or ROOT.parent.parent
    for exe in ("git", "/opt/homebrew/bin/git"):
        try:
            out = subprocess.run(
                [exe, "-C", str(repo), "log", "--first-parent", "origin/main",
                 "--format=%cs", "--", *UNIVERSE_DEFINITION_PATHS],
                capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0:
            return sorted({_date.fromisoformat(x) for x in out.stdout.split()})
    raise RuntimeError("git unusable: cannot detect universe-definition boundaries (R5 b)")


def live_count_gate_v3(vintage: str, membership: dict, boundaries: list | None = None) -> tuple[dict, list[str]]:
    """§A.5 live-count gate under v3 (R5): clean window, pairing, and comparability.

    Three rules, each fail-safe toward DEFERRED (neither pass nor fail):
      * comparability — live's count applies mcap >= $300M; a pre-market-cap
        PIT count is not the same quantity, so the gate waits for Phase B;
      * clean window — only live runs strictly after the latest universe-
        definition merge count (R5 b), and at least LIVE_GATE_MIN_CLEAN_OBS of
        them must exist (R5 a);
      * pairing — a live row is dated by its morning RUN date R; the PIT session
        it describes is previous_trading_day(R).
    """
    from datetime import date as _date  # noqa: PLC0415

    import pandas_market_calendars as mcal  # noqa: PLC0415

    has_mcap = bool(membership) and all("eligible" in rec for rec in membership.values())
    if boundaries is None:
        try:
            boundaries = universe_definition_boundaries()
        except RuntimeError as exc:
            return {"ran": True, "rule": "R5", "status": "DEFERRED",
                    "reason": f"universe-definition history unavailable ({exc}); "
                              "an unknown history is never treated as 'no changes'"}, []
    if not boundaries:
        return {"ran": True, "rule": "R5", "status": "DEFERRED",
                "reason": "git history returned no universe-definition commits — implausible, "
                          "refusing to treat it as a clean window"}, []
    last_change = boundaries[-1]
    live_path = ROOT / vintage / "raw" / "live" / "dashboard.json.gz"
    rows = (_read_raw(live_path).get("run_history") or []) if live_path.exists() else []
    clean = []
    for row in rows:
        try:
            r = _date.fromisoformat(str(row.get("date"))[:10])
        except ValueError:
            continue
        if row.get("universe") and r > last_change:
            clean.append((r, row["universe"]))
    result = {"ran": True, "rule": "R5", "last_definition_change": str(last_change),
              "clean_live_observations": len(clean), "min_required": LIVE_GATE_MIN_CLEAN_OBS,
              "threshold_pct": HALT_LIVE_COUNT_DIVERGENCE_PCT}
    if not has_mcap:
        result["status"] = "DEFERRED"
        result["reason"] = ("PIT count is pre-market-cap (or Phase B is incomplete); live applies "
                            "mcap >= $300M — evaluate after Phase B covers every session")
        return result, []
    if len(clean) < LIVE_GATE_MIN_CLEAN_OBS:
        result["status"] = "DEFERRED"
        result["reason"] = f"{len(clean)} clean live observations < {LIVE_GATE_MIN_CLEAN_OBS}"
        return result, []
    cal = mcal.get_calendar("NYSE")
    pairs = []
    for r, live_n in clean:
        prev = cal.schedule(start_date=r - _td(days=10), end_date=r - _td(days=1)).index
        if not len(prev):
            continue
        d = prev[-1].date()
        if d in membership:
            pit_n = len(membership[d]["eligible"])
            pairs.append(100.0 * abs(pit_n - live_n) / max(1, live_n))
    if len(pairs) < LIVE_GATE_MIN_CLEAN_OBS:
        result["status"] = "DEFERRED"
        result["reason"] = f"only {len(pairs)} clean observations fall inside the vintage"
        return result, []
    med = statistics.median(pairs)
    result.update({"status": "EVALUATED", "paired": len(pairs), "median_divergence_pct": round(med, 2)})
    halts = []
    if med > HALT_LIVE_COUNT_DIVERGENCE_PCT:
        halts.append(f"live count divergence (R5): median {med:.1f}% > {HALT_LIVE_COUNT_DIVERGENCE_PCT}% "
                     f"over {len(pairs)} clean paired observations")
    return result, halts


def _td(**kw):
    from datetime import timedelta  # noqa: PLC0415
    return timedelta(**kw)


def write_report(vintage: str) -> dict:
    from pit_universe_phase_a import build_membership  # noqa: PLC0415

    from pit_universe_phase_a import contract_version, require_transitions_complete  # noqa: PLC0415

    base = ROOT / vintage
    v3 = contract_version(vintage) == "v3"
    phase_b = None
    if v3:
        require_transitions_complete(vintage)
        from pit_universe_phase_a import build_membership_with_mcap, phase_b_status  # noqa: PLC0415
        phase_b = phase_b_status(vintage)
        phase_b = {k: v for k, v in phase_b.items() if k not in ("missing", "extra")} | {
            "missing_count": len(phase_b["missing"]), "extra_count": len(phase_b["extra"])}
    membership = (build_membership_with_mcap(vintage) if phase_b and phase_b["complete"]
                  else build_membership(vintage))
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
    windowed, halts = unknown_rate_gates(membership, v3=v3)
    mcap_section = None
    if v3:
        if not (phase_b and phase_b["complete"]):
            halts.append(f"phase B (market cap) incomplete: {phase_b} — dataset not signed off")
        else:
            mcap_section, mcap_halts = mcap_gates(vintage, membership)
            halts.extend(mcap_halts)

    ledger_path = base / "request_ledger.jsonl"
    ledger_summary = {"present": ledger_path.exists()}
    if ledger_path.exists():
        calls = failures = malformed = 0
        for line in ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line).get("event")
            except json.JSONDecodeError:
                malformed += 1
                continue
            if event == "request":
                calls += 1
            elif event == "failure":
                failures += 1
            else:
                malformed += 1
        ledger_summary = {"present": True, "calls": calls, "durable_failures": failures}
        if v3 and malformed:
            # A skipped record understates spend; v2 keeps its original tolerance.
            halts.append(f"request ledger has {malformed} malformed/unknown record(s) — spend unverifiable")
            ledger_summary["malformed"] = malformed
        if failures:
            # A vintage with unrecovered holes is incomplete by construction;
            # reporting it as a dataset would present a partial universe as a
            # whole one, which is the silent-truncation failure mode.
            halts.append(f"request ledger records {failures} unrecovered failure(s)")
    else:
        halts.append("no request ledger — provenance of this vintage is unverifiable")

    phase_b_ledger = None
    if v3 and phase_b and phase_b["complete"]:
        from pit_universe_phase_a import PHASE_B_CALL_CEILING  # noqa: PLC0415
        lp = base / "request_ledger_phase_b.jsonl"
        if not lp.exists():
            halts.append("no phase B request ledger — market-cap provenance is unverifiable")
        else:
            calls = failures = malformed = 0
            for line in lp.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line).get("event")
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if ev not in ("request", "failure"):
                    malformed += 1
                calls += ev == "request"
                failures += ev == "failure"
            if malformed:
                halts.append(f"phase B ledger has {malformed} malformed record(s) — spend unverifiable")
            if calls > PHASE_B_CALL_CEILING:
                halts.append(f"phase B ledger shows {calls} calls > ceiling {PHASE_B_CALL_CEILING}")
            phase_b_ledger = {"calls": calls, "durable_failures": failures,
                              "ceiling": PHASE_B_CALL_CEILING,
                              "headroom": PHASE_B_CALL_CEILING - calls}
            if failures:
                halts.append(f"phase B ledger records {failures} unrecovered failure(s)")

    divergence, divergence_halts = live_divergence(vintage)
    halts.extend(divergence_halts)
    count_divergence, count_halts = live_count_divergence(vintage, membership)
    halts.extend(count_halts)

    audit = _audit_results(vintage)
    if v3 and not audit.get("ran"):
        halts.append(f"classification audit not usable: {audit.get('reason')}")
    if audit.get("ran"):
        for month, rec in sorted(audit["per_month"].items()):
            if rec["type_disagree"]:
                halts.append(
                    f"{month}: {rec['type_disagree']} security-type disagreement(s) "
                    f"(contamination={rec['contamination']}, "
                    f"false_exclusion={rec['false_exclusion']}) — zero tolerated"
                )
            if v3 and rec["exchange_disagree"] > HALT_DRIFT_EXCHANGE_DISAGREEMENTS:
                halts.append(
                    f"{month}: {rec['exchange_disagree']} exchange disagreement(s) "
                    f"(membership flips={rec['exchange_membership_flip']}) — zero tolerated (R2)"
                )
            elif not v3 and rec["labelled"]:
                exch_pct = 100.0 * rec["exchange_disagree"] / rec["labelled"]
                if exch_pct > HALT_DRIFT_EXCHANGE_PCT_V2:
                    halts.append(f"{month}: exchange drift {exch_pct:.2f}% > {HALT_DRIFT_EXCHANGE_PCT_V2}%")

    raw_files = sorted((base / "raw").rglob("*.json.gz"))
    if v3:
        # Completion markers are evidence too: hash them so an archive cannot
        # verify with its pagination proof missing or altered.
        raw_files = sorted(raw_files + list((base / "raw").rglob("_complete.json")))
    n_quarters = len({(d.year, (d.month - 1) // 3) for d in dates}) if v3 else 12
    manifest = {
        "vintage": vintage,
        "timezone": "America/New_York",
        "date_range_et": [str(dates[0]), str(dates[-1])],
        "sessions": len(dates),
        "phase": "A",
        "constraints_applied": {
            "min_price": 5.0, "min_share_volume": 500_000,
            "exchanges": ["NYSE", "NASDAQ"],
            "type": sorted(_eligible_types(vintage)) if v3 else "CS",
            "market_cap": "NOT APPLIED — Phase B",
        },
        "classification_policy": (
            "forward-held monthly from the snapshot date, with membership-relevant "
            "transitions resolved to the day (§3a-v2)" if v3
            else "forward-held monthly, applied from snapshot date only"),
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
        "live_count_divergence": count_divergence,
        "calendar": _calendar_provenance(),
        "halts": halts,
        "phase_b_gate": {
            "distinct_tickers": len(distinct_eligible),
            "quarters": n_quarters,
            "projected_detail_calls": len(distinct_eligible) * n_quarters,
            "ceiling": 75_000,
            "within_ceiling": len(distinct_eligible) * n_quarters <= 75_000,
        },
    }

    if v3:
        # Schema additions are v3-only so a v2 vintage's manifest replays exactly.
        manifest["phase"] = "B" if (phase_b and phase_b["complete"]) else "A"
        manifest["phase_b"] = phase_b
        manifest["phase_b_ledger"] = phase_b_ledger
        # The 3-year projection (distinct x all quarters vs 75,000) is superseded by
        # the eligible-quarter rule and the 140,000 ceiling (Ray, 2026-10-06).
        from pit_universe_phase_a import PHASE_B_CALL_CEILING  # noqa: PLC0415
        manifest["phase_b_gate"] = {
            "rule": "shares lookups for (ticker, quarter) pairs eligible pre-mcap on >= 1 session",
            "lookups_required": phase_b["expected"] if phase_b else None,
            "ceiling": PHASE_B_CALL_CEILING,
        }
        manifest["market_cap"] = mcap_section
        if phase_b and phase_b["complete"]:
            manifest["constraints_applied"]["market_cap"] = (
                "> $300M, estimated as quarterly as-of shares x split multiplier x unadjusted close (§3c, R9)")
            manifest["distinct_eligible_tickers"] = len({t for r in membership.values() for t in r["eligible"]})
            for row in daily_counts:
                from datetime import date as _d  # noqa: PLC0415
                row["eligible"] = len(membership[_d.fromisoformat(row["date"])]["eligible"])
        stamp = base / "contract.json"
        manifest["contract_version"] = _contract_version(vintage)
        manifest["contract_stamp_sha256"] = _sha256(stamp)
        manifest["normalization_version"] = "phase-a/v3"
        manifest["code_sha"] = _code_sha()
        manifest["config_sha256"] = hashlib.sha256(stamp.read_bytes()).hexdigest()
        from pit_universe_phase_a import transition_status  # noqa: PLC0415
        manifest["transition_resolution"] = {k: v for k, v in transition_status(vintage).items()
                                             if k != "missing"}

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

    manifest_doc = json.loads(src.read_text())
    if manifest_doc.get("vintage") not in (None, vintage) or (
            manifest_doc.get("contract_version") == "v3" and manifest_doc.get("vintage") != vintage):
        raise SystemExit(f"VERIFY FAILED: manifest is for vintage {manifest_doc.get('vintage')!r}, "
                         f"not {vintage!r}")
    expected = manifest_doc.get("raw_hashes", {})
    if not expected:
        raise SystemExit(f"{src} carries no raw_hashes")
    stamp = base / "contract.json"
    local_v = json.loads(stamp.read_text()).get("version") if stamp.exists() else "v2"
    manifest_v = manifest_doc.get("contract_version", "v2")
    if local_v != manifest_v:
        raise SystemExit(f"VERIFY FAILED: local contract {local_v} != manifest contract {manifest_v}")
    if manifest_v not in ("v2", "v3"):
        raise SystemExit(f"VERIFY FAILED: unknown contract version {manifest_v!r}")
    if manifest_v == "v3":
        digest = _sha256(stamp)
        if digest != manifest_doc.get("contract_stamp_sha256"):
            raise SystemExit("VERIFY FAILED: contract.json does not match the manifest — "
                             "the vintage would not replay under its own rules")
        if hashlib.sha256(stamp.read_bytes()).hexdigest() != manifest_doc.get("config_sha256"):
            raise SystemExit("VERIFY FAILED: config hash mismatch")

    present = {
        str(p.relative_to(base)): p
        for p in sorted((base / "raw").rglob("*.json.gz"))
    }
    if manifest_v == "v3":
        present.update({str(p.relative_to(base)): p
                        for p in sorted((base / "raw").rglob("_complete.json"))})

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


_REPO_RE = __import__("re").compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def _repo_is_private(repo: str) -> bool | None:
    """True only if GitHub says the repository is private; None if unknown."""
    import subprocess
    try:
        out = subprocess.run(["gh", "api", f"repos/{repo}", "--jq", ".private"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() == "true"


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
        for extra in ("manifest.json", "request_ledger.jsonl", "daily_counts.json",
                      "contract.json", "audit_plan.json"):
            if (base / extra).exists():
                tar.add(base / extra, arcname=f"{vintage}/{extra}")

    digest = _sha256(archive)
    (base / "archive.sha256").write_text(f"{digest}  {archive.name}\n")
    size_mb = archive.stat().st_size / (1 << 20)
    logger.info("archive %s (%.1f MB) sha256=%s", archive.name, size_mb, digest)
    # R8: the vintage contains licensed Polygon data and this repo is PUBLIC, so
    # an upload command is only emitted for an explicitly configured private
    # repository, and always with --repo.
    import os
    target = os.environ.get("PIT_RELEASE_REPO", "").strip()
    out = {"archive": str(archive), "sha256": digest, "size_mb": round(size_mb, 1), "upload": None}
    if not target:
        out["note"] = "no upload: set PIT_RELEASE_REPO to a PRIVATE repository (R8)"
    elif not _REPO_RE.fullmatch(target):
        out["note"] = f"no upload: {target!r} is not an owner/repo name"
    elif _repo_is_private(target) is not True:
        out["note"] = (f"no upload: {target} is not verifiably PRIVATE (R8). This vintage holds "
                       "licensed Polygon data and must never go to a public repository.")
    else:
        # An argument vector, not a shell string: nothing here is ever interpolated by a shell.
        out["upload"] = ["gh", "release", "create", f"pit-universe-{vintage}", str(archive),
                         "--repo", target, "--title", f"PIT universe vintage {vintage}",
                         "--notes", "manifest sha256 committed in the screener repo"]
        out["note"] = f"{target} verified PRIVATE"
    print(json.dumps(out, indent=2))
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

    from pit_universe_phase_a import (  # noqa: PLC0415
        _classification_by_month, build_membership, contract_version, eligible_types_for,
        resolved_label, resolved_overrides,
    )

    live = _read_raw(live_path)
    membership = build_membership(vintage)
    labels = _classification_by_month(vintage)
    covered = set(membership)
    v3 = contract_version(vintage) == "v3"
    eligible_types = eligible_types_for(vintage)
    overrides = resolved_overrides(vintage, labels) if v3 else {}
    sessions_sorted = sorted(covered)

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
        if v3:
            # Live rows carry the morning RUN date; the session they describe is
            # the previous trading session (R5 pairing).
            prior = [x for x in sessions_sorted if x < d]
            d = prior[-1] if prior and (d - prior[-1]).days <= 5 else None
            if d is None:
                out_of_range += 1
                continue
        if d not in covered:
            out_of_range += 1
            continue
        checked += 1
        if ticker in set(membership[d]["eligible_pre_mcap"]):
            continue
        # Absent from PIT. Attribute it.
        if v3:
            held = resolved_label(labels, overrides, ticker, d) or {}
        else:
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
        if held_type in eligible_types and on_allowed_exchange:
            # Under v3 the observable price/volume and history gates can also
            # exclude it; only a name passing those is a real PIT contradiction.
            if v3 and ticker not in set(membership[d]["pre_classification"]):
                pit_stricter.append({**record, "reason": "price_or_volume_on_D"})
                continue
            if v3:
                # Resolved label eligible, observables pass, still absent: the
                # only remaining gate is the §11 history rule, which live lacks.
                pit_stricter.append({**record, "reason": "insufficient_history"})
                continue
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
