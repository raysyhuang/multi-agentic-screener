#!/usr/bin/env python3
"""Compare external engine collection outputs between local and HTTP modes.

This is an A/B harness for collector-level comparisons only. It runs
``src.engines.collector.collect_engine_results`` twice (``local`` then ``http``),
normalizes the engine payloads, computes hashes/diffs, and writes a JSON artifact.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ROUND_FLOATS = 6
DIAGNOSTIC_KEYS = (
    "degraded_execution",
    "used_fallback_output",
    "fallback_reason",
    "zero_pick_fallback_used",
    "subprocess_timed_out",
    "subprocess_exception",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run collector in local and http modes and compare engine payloads.",
    )
    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help="Target date (YYYY-MM-DD). Passed to collector; http mode may ignore it.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "engine_ab",
        help="Directory for JSON comparison artifacts.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of local+http A/B repetitions (default: 1).",
    )
    parser.add_argument(
        "--include-normalized-payloads",
        action="store_true",
        help="Include normalized payload bodies in artifact (larger files).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["local", "http"],
        default=["local", "http"],
        help="Modes to run, in sequence (default: local http).",
    )
    parser.add_argument(
        "--mode-timeout-s",
        type=float,
        default=720.0,
        help=(
            "Collector timeout per mode in seconds (default: 720). "
            "Note: local thread-backed work may continue briefly after timeout."
        ),
    )
    return parser.parse_args()


def _parse_target_date(raw: str | None) -> date | None:
    if not raw:
        return None
    return date.fromisoformat(raw)


def _round_float(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, ROUND_FLOATS)
    return value


def _stable_json_hash(obj: Any) -> str:
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_pick(pick: Any) -> dict[str, Any]:
    # Import locally to avoid ruff E402 when sys.path is modified above.
    pick_data = pick.model_dump() if hasattr(pick, "model_dump") else dict(pick)

    metadata = pick_data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    # Keep only comparison-relevant metadata; skip paths/timestamps/noisy counters.
    meta_keep: dict[str, Any] = {}
    for key in sorted(metadata):
        key_l = str(key).lower()
        if any(token in key_l for token in ("timestamp", "time", "duration", "path", "file", "log")):
            continue
        value = metadata[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            meta_keep[str(key)] = _round_float(value)
        elif isinstance(value, list):
            meta_keep[str(key)] = sorted(
                [_round_float(v) for v in value if isinstance(v, (str, int, float, bool)) or v is None],
                key=lambda x: str(x),
            )

    risk_factors = pick_data.get("risk_factors") or []
    if not isinstance(risk_factors, list):
        risk_factors = []

    return {
        "ticker": str(pick_data.get("ticker", "")).upper(),
        "strategy": str(pick_data.get("strategy", "")),
        "entry_price": _round_float(pick_data.get("entry_price")),
        "stop_loss": _round_float(pick_data.get("stop_loss")),
        "target_price": _round_float(pick_data.get("target_price")),
        "confidence": _round_float(pick_data.get("confidence")),
        "holding_period_days": pick_data.get("holding_period_days"),
        "raw_score": _round_float(pick_data.get("raw_score")),
        "thesis": pick_data.get("thesis"),
        "risk_factors": sorted([str(x) for x in risk_factors]),
        "metadata": meta_keep,
    }


def _normalize_payload(payload: Any) -> dict[str, Any]:
    payload_data = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
    diagnostics = payload_data.get("diagnostics") or {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    normalized = {
        "engine_name": payload_data.get("engine_name"),
        "engine_version": payload_data.get("engine_version"),
        "run_date": payload_data.get("run_date"),
        "regime": payload_data.get("regime"),
        "status": payload_data.get("status"),
        "candidates_screened": payload_data.get("candidates_screened"),
        "diagnostics_flags": {
            key: diagnostics.get(key)
            for key in DIAGNOSTIC_KEYS
            if key in diagnostics
        },
        "picks": sorted(
            [_normalize_pick(pick) for pick in (payload_data.get("picks") or [])],
            key=lambda p: (p["ticker"], p["strategy"], str(p["entry_price"]), str(p["confidence"])),
        ),
    }
    return normalized


def _engine_summary(normalized: dict[str, Any]) -> dict[str, Any]:
    picks = normalized["picks"]
    ticker_set = sorted({p["ticker"] for p in picks})
    ticker_strategy_set = sorted({f'{p["ticker"]}|{p["strategy"]}' for p in picks})
    return {
        "engine_name": normalized["engine_name"],
        "run_date": normalized["run_date"],
        "status": normalized["status"],
        "regime": normalized["regime"],
        "candidates_screened": normalized["candidates_screened"],
        "pick_count": len(picks),
        "tickers": ticker_set,
        "ticker_strategy_keys": ticker_strategy_set,
        "hash": _stable_json_hash(normalized),
        "diagnostics_flags": normalized.get("diagnostics_flags", {}),
    }


def _coerce_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_coerce_jsonable(v) for v in value]
    return value


async def _run_mode_once(
    mode: str,
    target_date: date | None,
    include_payloads: bool,
    mode_timeout_s: float,
) -> dict[str, Any]:
    import src.config as config_module
    from src.engines import collector

    settings = config_module.get_settings()
    old_mode = settings.engine_run_mode
    settings.engine_run_mode = mode
    payloads = []
    failures = []
    timed_out = False
    try:
        started = time.perf_counter()
        try:
            payloads, failures = await asyncio.wait_for(
                collector.collect_engine_results(target_date=target_date),
                timeout=mode_timeout_s,
            )
        except TimeoutError:
            timed_out = True
        duration_s = round(time.perf_counter() - started, 3)
    finally:
        settings.engine_run_mode = old_mode

    normalized_by_engine: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for payload in payloads:
        norm = _normalize_payload(payload)
        engine_name = str(norm["engine_name"])
        normalized_by_engine[engine_name] = norm
        summaries.append(_engine_summary(norm))

    summaries.sort(key=lambda r: r["engine_name"])

    result: dict[str, Any] = {
        "mode": mode,
        "duration_s": duration_s,
        "timed_out": timed_out,
        "engine_count": len(payloads),
        "failure_count": len(failures),
        "failures": [_coerce_jsonable(f) for f in failures],
        "engines": summaries,
    }
    if include_payloads:
        result["normalized_payloads"] = normalized_by_engine
    return result


def _jaccard(a: set[str], b: set[str]) -> float | None:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return None
    return round(len(a & b) / len(union), 6)


def _compare_two_modes(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    by_engine_a = {e["engine_name"]: e for e in a.get("engines", [])}
    by_engine_b = {e["engine_name"]: e for e in b.get("engines", [])}
    engine_names = sorted(set(by_engine_a) | set(by_engine_b))

    per_engine: list[dict[str, Any]] = []
    exact_hash_matches = 0
    for name in engine_names:
        ea = by_engine_a.get(name)
        eb = by_engine_b.get(name)
        row: dict[str, Any] = {"engine_name": name}
        if ea is None or eb is None:
            row["status"] = "missing_in_one_mode"
            row["present_in_a"] = ea is not None
            row["present_in_b"] = eb is not None
        else:
            tickers_a = set(ea.get("tickers", []))
            tickers_b = set(eb.get("tickers", []))
            keys_a = set(ea.get("ticker_strategy_keys", []))
            keys_b = set(eb.get("ticker_strategy_keys", []))
            hash_match = ea.get("hash") == eb.get("hash")
            if hash_match:
                exact_hash_matches += 1
            row.update({
                "status": "compared",
                "run_date_a": ea.get("run_date"),
                "run_date_b": eb.get("run_date"),
                "run_date_match": ea.get("run_date") == eb.get("run_date"),
                "hash_match": hash_match,
                "pick_count_a": ea.get("pick_count"),
                "pick_count_b": eb.get("pick_count"),
                "ticker_jaccard": _jaccard(tickers_a, tickers_b),
                "ticker_strategy_jaccard": _jaccard(keys_a, keys_b),
                "only_in_a": sorted(tickers_a - tickers_b),
                "only_in_b": sorted(tickers_b - tickers_a),
                "degraded_a": bool((ea.get("diagnostics_flags") or {}).get("degraded_execution")),
                "degraded_b": bool((eb.get("diagnostics_flags") or {}).get("degraded_execution")),
            })
        per_engine.append(row)

    all_compared = [r for r in per_engine if r.get("status") == "compared"]
    run_date_match_all = all(r.get("run_date_match") for r in all_compared) if all_compared else False

    return {
        "mode_a": a.get("mode"),
        "mode_b": b.get("mode"),
        "engines": per_engine,
        "summary": {
            "engine_names": engine_names,
            "compared_engines": len(all_compared),
            "exact_hash_matches": exact_hash_matches,
            "exact_hash_match_rate": round(exact_hash_matches / len(all_compared), 6) if all_compared else None,
            "run_date_match_all_compared": run_date_match_all,
            "duration_a_s": a.get("duration_s"),
            "duration_b_s": b.get("duration_s"),
            "failures_a": a.get("failure_count"),
            "failures_b": b.get("failure_count"),
        },
    }


def _artifact_name(target_date: date | None, repetition: int, modes: list[str]) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    td = target_date.isoformat() if target_date else "latest"
    mode_tag = "-vs-".join(modes)
    return f"engine_ab_{td}_{mode_tag}_r{repetition:02d}_{stamp}.json"


async def _run_once(
    *,
    target_date: date | None,
    modes: list[str],
    include_payloads: bool,
    mode_timeout_s: float,
    repetition: int,
    output_dir: Path,
) -> Path:
    import src.config as config_module

    settings = config_module.get_settings()
    mode_runs: list[dict[str, Any]] = []
    for mode in modes:
        mode_runs.append(await _run_mode_once(mode, target_date, include_payloads, mode_timeout_s))

    comparisons: list[dict[str, Any]] = []
    if len(mode_runs) >= 2:
        for idx in range(len(mode_runs) - 1):
            comparisons.append(_compare_two_modes(mode_runs[idx], mode_runs[idx + 1]))

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "target_date_requested": target_date.isoformat() if target_date else None,
        "repetition": repetition,
        "http_urls_configured": {
            "koocore_api_url": bool(getattr(settings, "koocore_api_url", "")),
            "gemini_api_url": bool(getattr(settings, "gemini_api_url", "")),
        },
        "runs": mode_runs,
        "comparisons": comparisons,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _artifact_name(target_date, repetition, modes)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _print_console_summary(report_path: Path, artifact: dict[str, Any]) -> None:
    print(f"Saved: {report_path}")
    for run in artifact.get("runs", []):
        print(
            f"[{run['mode']}] engines={run['engine_count']} failures={run['failure_count']} "
            f"duration={run['duration_s']}s timed_out={run.get('timed_out', False)}",
        )
        for eng in run.get("engines", []):
            print(
                f"  - {eng['engine_name']}: run_date={eng['run_date']} picks={eng['pick_count']} "
                f"hash={eng['hash'][:12]} degraded={bool((eng.get('diagnostics_flags') or {}).get('degraded_execution'))}",
            )
    for cmp_report in artifact.get("comparisons", []):
        summary = cmp_report.get("summary", {})
        print(
            f"[compare {cmp_report.get('mode_a')} vs {cmp_report.get('mode_b')}] "
            f"hash_match_rate={summary.get('exact_hash_match_rate')} "
            f"run_date_match_all={summary.get('run_date_match_all_compared')}",
        )


async def _main_async(args: argparse.Namespace) -> int:
    target_date = _parse_target_date(args.target_date)
    for repetition in range(1, args.repeat + 1):
        path = await _run_once(
            target_date=target_date,
            modes=list(args.modes),
            include_payloads=args.include_normalized_payloads,
            mode_timeout_s=args.mode_timeout_s,
            repetition=repetition,
            output_dir=args.output_dir,
        )
        artifact = json.loads(path.read_text(encoding="utf-8"))
        _print_console_summary(path, artifact)
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
