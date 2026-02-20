#!/usr/bin/env python3
"""
Deterministic Pro30 config comparison harness.

Runs Pro30 twice for the same as-of date using two configs, then compares:
- CSV row counts
- unique ticker counts
- symmetric ticker set differences
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime as dt
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_pipeline():
    import sys

    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.core.config import load_config
    from src.pipelines.pro30 import run_pro30

    return load_config, run_pro30


def _run_one(
    label: str,
    config_path: str,
    asof_date,
    base_out_dir: Path,
):
    load_config, run_pro30 = _load_pipeline()
    cfg = load_config(config_path)
    run_dir = base_out_dir / label.lower()
    run_dir.mkdir(parents=True, exist_ok=True)

    result = run_pro30(
        config=cfg,
        asof_date=asof_date,
        output_date=asof_date,
        run_dir=run_dir,
    )

    csv_path = result.get("candidates_csv")
    if not csv_path:
        return {
            "label": label,
            "config_path": config_path,
            "run_dir": str(run_dir),
            "candidates_csv": None,
            "rows": 0,
            "unique": 0,
            "tickers": [],
        }

    df = pd.read_csv(csv_path)
    tickers = []
    if not df.empty and "Ticker" in df.columns:
        tickers = [str(t).strip() for t in df["Ticker"].tolist() if str(t).strip()]

    return {
        "label": label,
        "config_path": config_path,
        "run_dir": str(run_dir),
        "candidates_csv": str(csv_path),
        "rows": int(len(df)),
        "unique": int(len(set(tickers))),
        "tickers": sorted(set(tickers)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Pro30 outputs across two configs.")
    parser.add_argument("--date", required=True, help="As-of date YYYY-MM-DD")
    parser.add_argument("--config-a", default="config/default.yaml", help="First config path")
    parser.add_argument("--config-b", default="config/phase5.yaml", help="Second config path")
    parser.add_argument("--label-a", default="DEFAULT", help="Label for first config")
    parser.add_argument("--label-b", default="PHASE5", help="Label for second config")
    parser.add_argument(
        "--out-dir",
        default="outputs/repro",
        help="Output directory for comparison artifacts",
    )
    args = parser.parse_args()

    asof_date = dt.strptime(args.date, "%Y-%m-%d").date()
    root = _project_root()
    out_dir = root / args.out_dir / args.date
    out_dir.mkdir(parents=True, exist_ok=True)

    run_a = _run_one(args.label_a, args.config_a, asof_date, out_dir)
    run_b = _run_one(args.label_b, args.config_b, asof_date, out_dir)

    set_a = set(run_a["tickers"])
    set_b = set(run_b["tickers"])
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)

    report = {
        "date": args.date,
        "a": {k: v for k, v in run_a.items() if k != "tickers"},
        "b": {k: v for k, v in run_b.items() if k != "tickers"},
        "diff": {
            "a_only_count": len(only_a),
            "b_only_count": len(only_b),
            "a_only_tickers": only_a,
            "b_only_tickers": only_b,
        },
    }

    report_path = out_dir / f"pro30_compare_{args.label_a.lower()}_vs_{args.label_b.lower()}_{args.date}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{args.label_a}: rows={run_a['rows']} unique={run_a['unique']}")
    print(f"{args.label_b}: rows={run_b['rows']} unique={run_b['unique']}")
    print(f"Diff ({args.label_a.lower()}-only): {len(only_a)}")
    print(f"Diff ({args.label_b.lower()}-only): {len(only_b)}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

