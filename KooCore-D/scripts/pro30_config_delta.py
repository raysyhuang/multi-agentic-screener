#!/usr/bin/env python3
"""
Pro30-scoped config delta reporter.

Compares only the keys that can materially affect Pro30 output determinism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PRO30_RELEVANT_KEYS = [
    "universe.mode",
    "universe.cache_file",
    "universe.cache_max_age_days",
    "universe.manual_include_file",
    "universe.r2000_include_file",
    "universe.manual_include_mode",
    "liquidity.min_avg_dollar_volume_20d",
    "technicals.lookback_days",
    "technicals.price_min",
    "quality_filters_30d.rvol_min",
    "quality_filters_30d.atr_pct_min",
    "quality_filters_30d.near_high_max_pct",
    "quality_filters_30d.breakout_rsi_min",
    "quality_filters_30d.reversal_rsi_max",
    "quality_filters_30d.reversal_dist_to_high_min_pct",
    "quality_filters_30d.min_score",
    "outputs_30d.top_n_breakout",
    "outputs_30d.top_n_reversal",
    "outputs_30d.top_n_total",
    "attention_pool.rvol_min",
    "attention_pool.atr_pct_min",
    "attention_pool.min_abs_day_move_pct",
    "attention_pool.lookback_days",
    "attention_pool.enable_intraday",
    "regime_gate.enabled",
    "regime_gate.action",
    "runtime.polygon_primary",
    "runtime.polygon_fallback",
    "runtime.polygon_max_workers",
    "runtime.allow_partial_day_attention",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config(config_path: str):
    import sys

    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.core.config import load_config

    return load_config(config_path)


def _get_nested(data: dict, dotted_key: str):
    cur = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main() -> int:
    parser = argparse.ArgumentParser(description="Show Pro30-relevant config deltas.")
    parser.add_argument("--config-a", default="config/default.yaml", help="Baseline config path")
    parser.add_argument("--config-b", default="config/phase5.yaml", help="Comparison config path")
    parser.add_argument("--label-a", default="default", help="Label for baseline")
    parser.add_argument("--label-b", default="phase5", help="Label for comparison")
    parser.add_argument("--show-equal", action="store_true", help="Include unchanged keys")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    cfg_a = _load_config(args.config_a)
    cfg_b = _load_config(args.config_b)

    delta = {}
    for key in PRO30_RELEVANT_KEYS:
        a_val = _get_nested(cfg_a, key)
        b_val = _get_nested(cfg_b, key)
        if args.show_equal or a_val != b_val:
            delta[key] = {
                args.label_a: a_val,
                args.label_b: b_val,
                "changed": a_val != b_val,
            }

    print("Pro30-Relevant Config Delta")
    print("=" * 30)
    if not delta:
        print("No differences found.")
    else:
        for key in sorted(delta.keys()):
            item = delta[key]
            marker = "*" if item["changed"] else " "
            print(f"{marker} {key}")
            print(f"    {args.label_a}: {item[args.label_a]}")
            print(f"    {args.label_b}: {item[args.label_b]}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(delta, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved JSON delta: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

