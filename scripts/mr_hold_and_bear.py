"""MR hold length x MARKET regime at full scale (Arm B of mr_hold_and_bear_FINDINGS).

Two live questions, one harness:
  1. Does a longer max hold (5 / 7 bars instead of the live 3) help MR's exit
     walk? FORWARD_DECAY_FINDINGS measured +1pp on LIVE picks from holding ~7
     calendar days (CI crossing zero); this asks the same thing on the full
     3Y backtest population through the unified exit engine.
  2. Is MR's bear cohort structurally worse? Live 90d: bear -0.42%/trade (n=21)
     vs choppy +0.99%. n=21 cannot settle that; ~3Y of signals can say whether
     bear is worse THAN THE REST of the population.

Caveat that governs both: the backtest population is edgeless at live
selectivity (MEMORY: MR raw -0.047%/trade), so nothing here is an expectancy
claim. It is a relative read only.

Regime is the SPY MARKET regime per entry date (scripts.choppy_sniper_regime_test
.spy_market_regime) — NEVER the trade's own `regime` field, which
run_model_backtest stamps once per TICKER from its full history.

Pre-registered bear rule (copied from scripts/pead_regime_stamp.py):
  bear >= +0.5%/trade @ n>=100 -> no gate; bear <= 0 @ n>=100 -> propose a
  bear-block (separate PR, through the validation card); otherwise no action.

Usage:
  python scripts/mr_hold_and_bear.py --cache-file outputs/research/ohlcv_polygon_3y.parquet \
      --json-out outputs/research/mr_hold_and_bear.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from scripts.choppy_sniper_regime_test import spy_market_regime  # noqa: E402
from scripts.gen_mr_trades import LIVE_MR  # noqa: E402
from src.backtest.portfolio import BookTrade, simulate_book  # noqa: E402
from src.research.signal_backtest import run_model_backtest  # noqa: E402

HOLDS = (3, 5, 7)
SEED = 20260918
MAX_CONCURRENT = 10          # mirrors export_dashboard_data.PORTFOLIO_MAX_CONCURRENT
START_CAPITAL = 100_000.0
BEAR_RULE_MIN_N = 100
BEAR_RULE_NO_GATE = 0.5      # %/trade


def boot_ci(x: list[float], n_boot: int = 10_000, seed: int = SEED) -> tuple[float, float]:
    """Seeded percentile bootstrap 95% CI of the mean (iid resample of trades)."""
    rng = random.Random(seed)
    k = len(x)
    if k == 0:
        return (float("nan"), float("nan"))
    means = sorted(sum(x[rng.randrange(k)] for _ in range(k)) / k for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def stamp_market_regime(trades: list[dict], regime_by_date: dict[str, str]) -> list[dict]:
    """Attach `mkt` = SPY market regime on the trade's entry_date (ISO string)."""
    out = []
    for t in trades:
        d = str(t["entry_date"])[:10]
        out.append({**t, "mkt": regime_by_date.get(d, "unknown")})
    return out


def _cell(rows: list[dict]) -> dict:
    p = [r["pnl_pct"] for r in rows]
    if not p:
        return {"n": 0}
    lo, hi = boot_ci(p)
    by_year: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_year[str(r["entry_date"])[:4]].append(r["pnl_pct"])
    return {
        "n": len(p),
        "wr": round(sum(1 for x in p if x > 0) / len(p), 4),
        "avg": round(st.mean(p), 4),
        "median": round(st.median(p), 4),
        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        "sum": round(sum(p), 2),
        "per_year": {y: {"n": len(v), "avg": round(st.mean(v), 4)}
                     for y, v in sorted(by_year.items()) if len(v) >= 10},
        "exit_reasons": dict(Counter(r["exit_reason"] for r in rows)),
        "avg_hold_days": round(st.mean(r["holding_days"] for r in rows), 2),
    }


def summarize(trades: list[dict]) -> dict:
    """{regime: cell} plus 'all' and 'ex_bear' for the relative read."""
    out = {"all": _cell(trades)}
    for reg in ("bull", "choppy", "bear", "unknown"):
        rows = [t for t in trades if t["mkt"] == reg]
        if rows:
            out[reg] = _cell(rows)
    out["ex_bear"] = _cell([t for t in trades if t["mkt"] != "bear"])
    return out


def equity(trades: list[dict]) -> dict:
    rows = sorted(trades, key=lambda r: (str(r["entry_date"]), r["ticker"]))
    res = simulate_book(
        [BookTrade(entry=_d(r["entry_date"]), exit=_d(r["exit_date"]), pnl_pct=r["pnl_pct"])
         for r in rows],
        max_concurrent=MAX_CONCURRENT, start_capital=START_CAPITAL,
    )
    return {k: res[k] for k in ("taken", "skipped", "peak_concurrent", "total_return_pct",
                                "max_drawdown_pct", "sharpe")}


def _d(x) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x)[:10])


def bear_rule(cell: dict) -> str:
    n, avg = cell.get("n", 0), cell.get("avg", float("nan"))
    if n < BEAR_RULE_MIN_N:
        return f"NOT EVALUABLE (bear n={n} < {BEAR_RULE_MIN_N})"
    if avg >= BEAR_RULE_NO_GATE:
        return f"NO GATE (bear avg {avg:+.3f}% >= +{BEAR_RULE_NO_GATE}% @ n={n})"
    if avg <= 0:
        return f"PROPOSE BEAR-BLOCK via validation card (bear avg {avg:+.3f}% <= 0 @ n={n})"
    return f"NO ACTION (bear avg {avg:+.3f}% between 0 and +{BEAR_RULE_NO_GATE}% @ n={n})"


def run(cache_file: Path) -> dict:
    combined = pd.read_parquet(cache_file)
    price = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
             for t, g in combined.groupby("_ticker")}
    regime_by_date = spy_market_regime(cache_file=cache_file)
    dates = pd.to_datetime(combined["date"])
    prov = {
        "cache_file": str(cache_file),
        "rows": int(len(combined)), "tickers": int(combined["_ticker"].nunique()),
        "date_min": str(dates.min().date()), "date_max": str(dates.max().date()),
        "provenance_manifest": "none beside cache (pre-dates the .provenance.json convention)",
        "regime_source": "SPY SMA20/50 per date (scripts.choppy_sniper_regime_test.spy_market_regime)",
    }
    print(f"Loaded {prov['tickers']} tickers, {prov['rows']} rows, "
          f"{prov['date_min']} -> {prov['date_max']}")

    results: dict = {"params_base": LIVE_MR, "holds": {}, "provenance": prov, "seed": SEED}
    for hold in HOLDS:
        params = {**LIVE_MR, "holding_period": hold}
        res = run_model_backtest("mean_reversion", price, params)
        trades = stamp_market_regime([{
            "ticker": t.ticker, "entry_date": t.entry_date, "exit_date": t.exit_date,
            "pnl_pct": t.pnl_pct, "exit_reason": t.exit_reason, "holding_days": t.holding_days,
        } for t in res.trades], regime_by_date)
        summ = summarize(trades)
        eq = equity(trades)
        results["holds"][str(hold)] = {"by_regime": summ, "equity": eq}
        _print_hold(hold, summ, eq)

    results["bear_rule"] = {h: bear_rule(v["by_regime"].get("bear", {"n": 0}))
                            for h, v in results["holds"].items()}
    print("\n=== Pre-registered bear rule (live hold=3 is the decisional row) ===")
    for h, verdict in results["bear_rule"].items():
        print(f"  hold={h}: {verdict}")
    return results


def _print_hold(hold: int, summ: dict, eq: dict) -> None:
    print(f"\n=== MR hold={hold} — by MARKET regime (SPY-based) ===")
    print(f"  {'regime':<8}{'n':>6}{'WR':>7}{'avg%':>9}{'95% CI':>18}{'median':>8}"
          f"{'hold':>6}   per-year / exits")
    for reg in ("all", "bull", "choppy", "bear", "ex_bear"):
        c = summ.get(reg)
        if not c or not c.get("n"):
            continue
        ys = "  ".join(f"{y}:{v['avg']:+.2f}(n={v['n']})" for y, v in c["per_year"].items())
        ex = " ".join(f"{k}={v}" for k, v in sorted(c["exit_reasons"].items()))
        print(f"  {reg:<8}{c['n']:>6}{100 * c['wr']:>6.1f}%{c['avg']:>+9.3f}"
              f"  [{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}]{c['median']:>+8.2f}{c['avg_hold_days']:>6.1f}"
              f"   {ys} | {ex}")
    print(f"  equity (cap {MAX_CONCURRENT}, ${START_CAPITAL:,.0f}): taken={eq['taken']} "
          f"skipped={eq['skipped']} peak={eq['peak_concurrent']} "
          f"ret={eq['total_return_pct']:+.2f}% maxDD={eq['max_drawdown_pct']:.2f}% "
          f"sharpe={eq['sharpe']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-file", default="outputs/research/ohlcv_polygon_3y.parquet")
    ap.add_argument("--json-out", default=None,
                    help="write the full result dict (aggregates only) as JSON")
    args = ap.parse_args()
    results = run(Path(args.cache_file))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
