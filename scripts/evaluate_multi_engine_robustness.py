#!/usr/bin/env python3
"""Evaluate multi-engine backtest reports for robustness tuning.

This script compares synthesis tracks across windows (1Y/3Y/5Y) with a
balanced objective that prioritizes 5Y robustness while enforcing simple
non-regression guardrails on shorter windows.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WINDOW_WEIGHTS = {"1Y": 0.15, "3Y": 0.25, "5Y": 0.60}


@dataclass
class TrackMetrics:
    trades: int
    win_rate: float
    profit_factor: float
    sharpe: float
    pnl_points: float
    drawdown_points: float


@dataclass
class ReportRecord:
    path: Path
    label: str
    window: str
    date_range: str
    trading_days: int
    tracks: dict[str, TrackMetrics]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _window_label(start: str, end: str) -> str:
    if (start, end) == ("2025-02-20", "2026-02-20"):
        return "1Y"
    if (start, end) == ("2023-02-20", "2026-02-20"):
        return "3Y"
    if (start, end) == ("2021-02-20", "2026-02-20"):
        return "5Y"
    return f"{start}->{end}"


def _extract_track(node: dict[str, Any] | None) -> TrackMetrics | None:
    if not node:
        return None
    summary = node.get("summary", node)
    if not isinstance(summary, dict):
        return None
    return TrackMetrics(
        trades=_safe_int(summary.get("total_trades"), 0),
        win_rate=_safe_float(summary.get("win_rate"), 0.0),
        profit_factor=_safe_float(summary.get("profit_factor"), 0.0),
        sharpe=_safe_float(summary.get("sharpe_ratio", summary.get("sharpe")), 0.0),
        pnl_points=_safe_float(
            summary.get("cumulative_trade_pnl_points", summary.get("total_return_pct")), 0.0
        ),
        drawdown_points=_safe_float(
            summary.get("trade_pnl_drawdown_points", summary.get("max_drawdown_pct")), 0.0
        ),
    )


def load_report(path: Path) -> ReportRecord:
    obj = json.loads(path.read_text())
    dr = obj.get("date_range", {})
    start = str(dr.get("start"))
    end = str(dr.get("end"))
    label = (
        (((obj.get("tuning_meta") or {}) if isinstance(obj.get("tuning_meta"), dict) else {})
         .get("config_label"))
        or ((((obj.get("config") or {}) if isinstance(obj.get("config"), dict) else {})
             .get("tuning_meta") or {}).get("config_label"))
        or path.stem
    )

    tracks: dict[str, TrackMetrics] = {}
    per_engine = obj.get("per_engine") or {}
    synthesis = obj.get("synthesis") or {}
    benchmark = obj.get("benchmark") or {}

    mapping = {
        "mas": per_engine.get("mas"),
        "koocore_d": per_engine.get("koocore_d"),
        "gemini_stst": per_engine.get("gemini_stst"),
        "eq_synth": synthesis.get("equal_weight"),
        "cred_synth": synthesis.get("credibility_weight"),
        "regime_gated": synthesis.get("regime_gated"),
        "sized_synth": synthesis.get("sized_synth"),
    }
    for name, node in mapping.items():
        m = _extract_track(node)
        if m is not None:
            tracks[name] = m

    # SPY benchmark has different shape; keep only pnl_points.
    spy = benchmark.get("spy_buy_and_hold") if isinstance(benchmark, dict) else None
    if isinstance(spy, dict):
        tracks["spy_benchmark"] = TrackMetrics(
            trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe=0.0,
            pnl_points=_safe_float(spy.get("total_return_pct"), 0.0),
            drawdown_points=0.0,
        )

    return ReportRecord(
        path=path,
        label=str(label),
        window=_window_label(start, end),
        date_range=f"{start} -> {end}",
        trading_days=_safe_int(obj.get("trading_days"), 0),
        tracks=tracks,
    )


def _delta_score(cur: TrackMetrics, base: TrackMetrics) -> float:
    """Weighted delta score for one window (higher is better)."""
    d_pf = cur.profit_factor - base.profit_factor
    d_sharpe = cur.sharpe - base.sharpe
    dd_improve_ratio = 0.0
    if base.drawdown_points > 0:
        dd_improve_ratio = (base.drawdown_points - cur.drawdown_points) / base.drawdown_points
    trade_ratio = (cur.trades / base.trades) if base.trades > 0 else 1.0
    trade_stability_pen = abs(1.0 - trade_ratio)
    return (d_pf * 100.0) + (d_sharpe * 50.0) + (dd_improve_ratio * 25.0) - (trade_stability_pen * 10.0)


def _guardrail_failures(cur: TrackMetrics, base: TrackMetrics, window: str) -> list[str]:
    failures: list[str] = []
    if window in {"1Y", "3Y"}:
        if cur.profit_factor < base.profit_factor - 0.03:
            failures.append(f"{window} PF regression >0.03")
        if cur.sharpe < base.sharpe - 0.07:
            failures.append(f"{window} Sharpe regression >0.07")
        if base.drawdown_points > 0 and cur.drawdown_points > base.drawdown_points * 1.15:
            failures.append(f"{window} drawdown worsened >15%")
    return failures


def _five_year_improved(cur: TrackMetrics, base: TrackMetrics) -> bool:
    if cur.profit_factor >= base.profit_factor + 0.03:
        return True
    if cur.sharpe >= base.sharpe + 0.05:
        return True
    if (
        base.drawdown_points > 0
        and cur.drawdown_points <= base.drawdown_points * 0.90
        and cur.profit_factor >= base.profit_factor - 0.01
    ):
        return True
    return False


def _group_records(records: list[ReportRecord]) -> dict[str, dict[str, ReportRecord]]:
    grouped: dict[str, dict[str, ReportRecord]] = {}
    for r in records:
        grouped.setdefault(r.label, {})
        existing = grouped[r.label].get(r.window)
        if existing is None or r.path.stat().st_mtime >= existing.path.stat().st_mtime:
            grouped[r.label][r.window] = r
    return grouped


def _print_track_table(records_by_window: dict[str, ReportRecord], track: str) -> None:
    print(f"\nTrack: {track}")
    print("window  trades  win   pf     sharpe  pnl_pts   dd_pts   label")
    for window in sorted(records_by_window.keys(), key=lambda w: ("135".find(w[:1]), w)):
        r = records_by_window[window]
        m = r.tracks.get(track)
        if not m:
            continue
        print(
            f"{window:5s} {m.trades:6d} {m.win_rate:5.3f} "
            f"{m.profit_factor:6.3f} {m.sharpe:7.3f} {m.pnl_points:8.2f} {m.drawdown_points:8.2f} {r.label}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multi-engine backtest robustness")
    parser.add_argument("reports", nargs="+", help="Backtest report JSON files")
    parser.add_argument("--baseline", help="Baseline report JSON (single-window fallback)")
    parser.add_argument("--baseline-1y", help="Baseline 1Y report JSON")
    parser.add_argument("--baseline-3y", help="Baseline 3Y report JSON")
    parser.add_argument("--baseline-5y", help="Baseline 5Y report JSON")
    parser.add_argument("--track", default="cred_synth", help="Primary synth track to score (default: cred_synth)")
    args = parser.parse_args()

    report_paths = [Path(p) for p in args.reports]
    if not any([args.baseline, args.baseline_1y, args.baseline_3y, args.baseline_5y]):
        raise SystemExit("Provide --baseline or at least one of --baseline-1y/--baseline-3y/--baseline-5y")

    records = [load_report(p) for p in report_paths if Path(p).exists()]
    if not records:
        raise SystemExit("No report files found")

    grouped = _group_records(records)
    baseline_by_window: dict[str, ReportRecord] = {}
    baseline = None
    for window, path_str in [
        ("1Y", args.baseline_1y),
        ("3Y", args.baseline_3y),
        ("5Y", args.baseline_5y),
    ]:
        if not path_str:
            continue
        p = Path(path_str)
        if not p.exists():
            raise SystemExit(f"Baseline not found: {p}")
        rec = load_report(p)
        baseline_by_window[window] = rec
        if baseline is None:
            baseline = rec
    if args.baseline:
        p = Path(args.baseline)
        if not p.exists():
            raise SystemExit(f"Baseline not found: {p}")
        rec = load_report(p)
        baseline_by_window.setdefault(rec.window, rec)
        if baseline is None:
            baseline = rec
    assert baseline is not None

    print("Multi-engine robustness evaluator")
    print("Note: uses trade-PnL points and trade-PnL drawdown points, not capital-normalized returns.")
    print(f"Primary track: {args.track}")
    print("Baselines:")
    for w in ("1Y", "3Y", "5Y"):
        rec = baseline_by_window.get(w)
        if rec:
            print(f"  {w}: {rec.path} (label={rec.label})")

    # Print raw summaries by label/window for core synth tracks.
    for label, by_window in sorted(grouped.items()):
        print(f"\n=== Candidate: {label} ===")
        for track in ("eq_synth", "cred_synth", "regime_gated", "sized_synth"):
            if any(track in rec.tracks for rec in by_window.values()):
                _print_track_table(by_window, track)

    print("\n=== Scoring (vs baseline, primary track only) ===")
    print("candidate  score   5Y_ok  guardrails  notes")
    for label, by_window in sorted(grouped.items()):
        if label == baseline.label:
            continue
        total_score = 0.0
        failures: list[str] = []
        five_y_ok = None
        used_windows = 0
        for window, weight in WINDOW_WEIGHTS.items():
            cur_rec = by_window.get(window)
            base_rec = baseline_by_window.get(window)
            if not cur_rec or not base_rec:
                continue
            cur_m = cur_rec.tracks.get(args.track)
            base_m = base_rec.tracks.get(args.track)
            if not cur_m or not base_m:
                continue
            total_score += weight * _delta_score(cur_m, base_m)
            used_windows += 1
            failures.extend(_guardrail_failures(cur_m, base_m, window))
            if window == "5Y":
                five_y_ok = _five_year_improved(cur_m, base_m)

        notes = []
        if used_windows == 0:
            notes.append("no comparable windows")
        if five_y_ok is False:
            notes.append("5Y improvement threshold not met")
        if five_y_ok is None:
            notes.append("no 5Y report")
        print(
            f"{label:18s} {total_score:7.2f} "
            f"{str(bool(five_y_ok)) if five_y_ok is not None else 'n/a':5s} "
            f"{len(failures):9d}  "
            + (", ".join(failures + notes) if (failures or notes) else "PASS")
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
