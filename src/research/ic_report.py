"""IC Report CLI — run manually to analyze engine confidence calibration.

Usage:
    python -m src.research.ic_report
    python -m src.research.ic_report --lookback 180
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import date, datetime, timezone

from src.research.ic_analysis import (
    compute_engine_ic,
    compute_cross_engine_independence,
    EngineICReport,
    PairwiseCorrelation,
)

logger = logging.getLogger(__name__)

_KNOWN_ENGINES = ["koocore_d", "gemini_stst", "top3_7d", "mas_quant_screener"]


def _format_table(reports: list[EngineICReport]) -> str:
    """Format engine IC reports as a readable table."""
    header = f"{'Engine':<20} {'IC':>7} {'p-value':>9} {'N':>5} {'Hit Rate':>10} {'Brier':>7}"
    separator = "\u2500" * len(header)
    lines = [header, separator]
    for r in reports:
        lines.append(
            f"{r.engine_name:<20} {r.ic:>7.3f} {r.p_value:>9.4f} {r.n:>5d} "
            f"{r.hit_rate:>9.2f} {r.brier_score:>7.3f}"
        )
    return "\n".join(lines)


def _format_correlations(correlations: list[PairwiseCorrelation]) -> str:
    """Format pairwise correlations."""
    if not correlations:
        return "  (no engine pair data)"
    lines: list[str] = []
    for c in correlations:
        lines.append(
            f"  {c.engine_a} \u2194 {c.engine_b}: "
            f"{c.correlation:.2f} ({c.n_overlap} overlapping picks)"
        )
    return "\n".join(lines)


def _to_serializable(reports: list[EngineICReport], correlations: list[PairwiseCorrelation]) -> dict:
    """Convert results to JSON-serializable dict."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engines": [
            {
                "engine_name": r.engine_name,
                "ic": r.ic,
                "p_value": r.p_value,
                "n": r.n,
                "hit_rate": r.hit_rate,
                "brier_score": r.brier_score,
            }
            for r in reports
        ],
        "cross_engine_correlations": [
            {
                "engine_a": c.engine_a,
                "engine_b": c.engine_b,
                "correlation": c.correlation,
                "p_value": c.p_value,
                "n_overlap": c.n_overlap,
            }
            for c in correlations
        ],
    }


async def run_report(lookback_days: int = 90) -> None:
    """Run IC analysis and print + save results."""
    from src.db.session import get_session

    reports: list[EngineICReport] = []

    async with get_session() as session:
        for engine in _KNOWN_ENGINES:
            report = await compute_engine_ic(
                engine, session, lookback_days=lookback_days
            )
            reports.append(report)

        correlations = await compute_cross_engine_independence(
            session, lookback_days=lookback_days
        )

    # Print to stdout
    print(f"\nIC Report (lookback={lookback_days}d, as of {date.today()})")
    print()
    print(_format_table(reports))
    print()
    print("Cross-Engine Independence (prediction correlation):")
    print(_format_correlations(correlations))
    print()

    # Save JSON
    output_dir = os.path.join("outputs", "ic_reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ic_{date.today().isoformat()}.json")
    with open(output_path, "w") as f:
        json.dump(_to_serializable(reports, correlations), f, indent=2)
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Engine IC Analysis Report")
    parser.add_argument(
        "--lookback", type=int, default=90,
        help="Lookback window in days (default: 90)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_report(lookback_days=args.lookback))


if __name__ == "__main__":
    main()
