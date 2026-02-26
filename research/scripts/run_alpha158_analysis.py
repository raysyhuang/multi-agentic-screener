"""Run Alpha158 factor IC analysis on local Qlib data.

Loads Qlib binary data, runs the Alpha158 feature handler, computes
per-factor Spearman IC, and reports the top 20 factors by absolute IC.

Usage:
    python research/scripts/run_alpha158_analysis.py

Prerequisites:
    1. pip install -r research/requirements-research.txt
    2. python research/scripts/convert_ohlcv_to_qlib.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
QLIB_DIR = DATA_DIR / "qlib_data"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs"


def run_analysis() -> None:
    try:
        import qlib
        from qlib.data import D
        from qlib.contrib.data.handler import Alpha158
    except ImportError:
        print("ERROR: qlib not installed. Run: pip install -r research/requirements-research.txt")
        sys.exit(1)

    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("ERROR: scipy/pandas not installed.")
        sys.exit(1)

    if not QLIB_DIR.exists():
        print(f"ERROR: Qlib data not found at {QLIB_DIR}")
        print("Run: python research/scripts/convert_ohlcv_to_qlib.py")
        sys.exit(1)

    # Initialize Qlib
    qlib.init(provider_uri=str(QLIB_DIR), region="us")

    end_date = date.today()
    start_date = end_date - timedelta(days=365)

    print(f"Running Alpha158 analysis ({start_date} to {end_date})...")

    # Load Alpha158 features
    handler = Alpha158(
        instruments="all",
        start_time=str(start_date),
        end_time=str(end_date),
    )

    df = handler.fetch()
    if df.empty:
        print("No data returned from Alpha158 handler.")
        return

    print(f"Feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")

    # Compute forward 5-day return as the target
    close = D.features(
        D.instruments("all"),
        ["$close"],
        start_time=str(start_date),
        end_time=str(end_date),
    )
    if close.empty:
        print("Could not load close prices for return computation.")
        return

    close.columns = ["close"]
    close["fwd_5d_ret"] = close.groupby(level=0)["close"].pct_change(5).shift(-5)
    target = close["fwd_5d_ret"].dropna()

    # Align features and target
    common_idx = df.index.intersection(target.index)
    if len(common_idx) < 100:
        print(f"Not enough aligned data ({len(common_idx)} rows). Need >= 100.")
        return

    df_aligned = df.loc[common_idx]
    target_aligned = target.loc[common_idx]

    print(f"Aligned: {len(common_idx)} observations")

    # Compute per-factor IC
    ic_results = {}
    for col in df_aligned.columns:
        series = df_aligned[col].dropna()
        aligned_target = target_aligned.loc[series.index]
        if len(series) < 50:
            continue
        if series.nunique() <= 1:
            continue
        corr, pval = spearmanr(series, aligned_target)
        ic_results[col] = {"ic": round(float(corr), 4), "p_value": round(float(pval), 4)}

    # Sort by absolute IC
    sorted_factors = sorted(ic_results.items(), key=lambda x: abs(x[1]["ic"]), reverse=True)

    # Print top 20
    print(f"\nTop 20 factors by |IC| ({len(ic_results)} total factors):")
    print(f"{'Factor':<40} {'IC':>8} {'p-value':>10}")
    print("-" * 60)
    for name, vals in sorted_factors[:20]:
        print(f"{name:<40} {vals['ic']:>8.4f} {vals['p_value']:>10.4f}")

    # Save full results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"alpha158_ic_{date.today().isoformat()}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "date": date.today().isoformat(),
                "n_factors": len(ic_results),
                "n_observations": len(common_idx),
                "top_20": [{"factor": k, **v} for k, v in sorted_factors[:20]],
                "all_factors": {k: v for k, v in sorted_factors},
            },
            f,
            indent=2,
        )
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    run_analysis()
