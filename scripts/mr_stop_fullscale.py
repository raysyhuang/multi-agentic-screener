"""Full-scale MR stop-width test on the 3Y Polygon cache — REJECTED result.

Kept in the repo because the conclusion is a NEGATIVE one and negative results
are the ones that get re-derived. The MR stop change looks compelling at low
selectivity and evaporates at the live gate; this script is the proof, so nobody
has to rebuild it a third time (see outputs/research/exit_layer_FINDINGS.md).

Mirrors the LIVE execution config from settings per CLAUDE.md (trail 0.5/0.3,
10bp slippage, gap-through fills, score-tiered stops) and varies ONLY the stop
width, at two selectivity levels:

  min_score=50  low selectivity — the population where a FALSE positive appears
  min_score=75  live selectivity (choppy_min_score) — where it vanishes

Always prints the PER-YEAR split beside the CI: the low-selectivity result is
carried entirely by 2025 and the pooled CI hides that.

Usage:
    python scripts/mr_stop_fullscale.py [--cache PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import get_settings
from src.research.signal_backtest import run_model_backtest

DEFAULT_CACHE = Path("outputs/research/ohlcv_polygon_3y.parquet")
DEFAULT_OUT = Path("outputs/research/mr_stop_fullscale.json")

# Live score-tiered stops (src/output/performance.py::_evaluate_position)
LIVE_TIERS = [(85, 1.25), (70, 0.85), (0, 0.50)]


def base_params() -> dict:
    s = get_settings()
    return dict(
        rsi2_threshold=10.0,        # live MR gate: RSI(2) <= 10
        stop_atr_mult=0.75,         # scan base; tiers override per trade
        target_atr_mult=1.5,
        holding_period=3,
        trail_activate_pct=s.trail_activate_pct,
        trail_distance_pct=s.trail_distance_pct,
        gap_through=True,
    )


def load_prices(cache: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(cache).rename(columns={"_ticker": "ticker"})
    return {t: g.drop(columns=["ticker"]).reset_index(drop=True)
            for t, g in df.groupby("ticker")}


def summarize(trades) -> dict:
    r = [t.pnl_pct for t in trades]
    if not r:
        return {}
    sd = st.pstdev(r) or 1e-9
    se = sd / (len(r) ** 0.5)
    mean = st.mean(r)
    return {"n": len(r), "wr": 100 * sum(1 for x in r if x > 0) / len(r),
            "avg": mean, "ci_lo": mean - 1.96 * se, "ci_hi": mean + 1.96 * se,
            "sum": sum(r)}


def by_year(trades) -> dict[str, dict]:
    return {str(y): summarize([t for t in trades if t.signal_date.year == y])
            for y in sorted({t.signal_date.year for t in trades})}


def configs() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = [
        ("LIVE tiers (1.25/0.85/0.50)", {"score_stop_tiers": LIVE_TIERS}),
    ]
    for k in (1.5, 2.0):
        out.append((f"tiers x{k}",
                    {"score_stop_tiers": [(s, round(m * k, 3)) for s, m in LIVE_TIERS]}))
    for flat in (0.75, 1.5, 2.0):
        out.append((f"flat {flat}xATR (no tiers)", {"score_stop_tiers": [(0, flat)]}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    prices = load_prices(args.cache)
    print(f"loaded {len(prices)} tickers from {args.cache}\n")

    results: dict[str, dict] = {}
    for min_score in (50.0, 75.0):
        label = "LOW selectivity — known artifact zone" if min_score < 75 else "LIVE selectivity"
        print("#" * 92)
        print(f"# min_score = {min_score}   ({label})")
        print("#" * 92)
        for name, override in configs():
            params = {**base_params(), "min_score": min_score, **override}
            res = run_model_backtest("mean_reversion", prices, params)
            s = summarize(res.trades)
            if not s:
                print(f"{name:30s} NO TRADES\n")
                continue
            yr = by_year(res.trades)
            results[f"ms{min_score}|{name}"] = {
                "overall": s, "by_year": yr, "exit_reasons": res.by_exit_reason}
            flag = "" if s["ci_lo"] < 0 < s["ci_hi"] else "  <-- CI excludes 0"
            print(f"{name:30s} n={s['n']:6d}  WR={s['wr']:5.2f}%  avg={s['avg']:+.4f}%  "
                  f"95%CI[{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]{flag}")
            print(f"{'':30s} by year: " + "  ".join(
                f"{y}:{v['avg']:+.3f}%(n={v['n']})" for y, v in yr.items()))
            print(f"{'':30s} exits: {res.by_exit_reason}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
