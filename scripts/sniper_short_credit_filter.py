"""Stage-1 conditioned study: do short-activity or credit-spread AVOID filters
improve SNIPER expectancy?

The unconditional probes (short_volume_probe, days_to_cover_probe) showed heavily
shorted names drift DOWN monotonically. This asks the decisive question: sniper
goes LONG — does dropping its trades on high-short / crowded-short names, or during
credit-spread widening (risk-off), lift sniper's own expectancy past noise?

Inputs (all cached / free — no heavy re-pull):
  * sniper Run-E trades CSV (ticker, entry_date, exit_date, pnl_pct, regime, ...)
  * data/cache/short_volume/{ticker}.json   {date: short_volume_ratio}
  * data/cache/short_interest/{ticker}.json {settlement_date: days_to_cover}
  * FRED BAMLH0A0HYM2 (ICE BofA US HY OAS, free) for credit-spread state

For each trade we tag state AS OF entry_date (look-ahead-safe: short vol uses the
last obs strictly before entry; DTC uses the last settlement published >=8bd before
entry; HY uses the last daily obs on/before entry). Then we bucket sniper
expectancy by each, and simulate each avoid-filter's net effect (all trades and
the live-faithful non-bear subset), with a paired bootstrap on the expectancy delta.

Usage:
  python scripts/sniper_short_credit_filter.py \
      --trades outputs/research/sniper_truth_E_live_fixed_2026-07-26.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from src.config import get_settings
from src.data.fred_client import FREDClient

SV_CACHE = Path("data/cache/short_volume")
SI_CACHE = Path("data/cache/short_interest")
PUB_LAG_DAYS = 8  # FINRA short-interest publication lag (business days approx)


def _load_trades(path: str) -> list[dict]:
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append({
                    "ticker": r["ticker"].upper(),
                    "entry": date.fromisoformat(r["entry_date"]),
                    "pnl": float(r["pnl_pct"]),
                    "regime": r.get("regime", ""),
                    "score": float(r.get("score") or 0),
                })
            except (ValueError, KeyError):
                continue
    return out


def _load_json_series(path: Path) -> dict[date, float]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    out = {}
    for k, v in raw.items():
        try:
            out[datetime.strptime(k, "%Y-%m-%d").date()] = float(v)
        except (ValueError, TypeError):
            continue
    return out


def _latest_before(series: dict[date, float], asof: date, lag_days: int = 0) -> float | None:
    """Most recent value with key <= asof - lag_days. None if absent."""
    cutoff = asof - timedelta(days=lag_days)
    best_d, best_v = None, None
    for d, v in series.items():
        if d <= cutoff and (best_d is None or d > best_d):
            best_d, best_v = d, v
    return best_v


async def _hy_oas(start: date, end: date) -> dict[date, float]:
    s = get_settings()
    client = FREDClient(api_key=s.fred_api_key or None)
    df = await client.get_series("BAMLH0A0HYM2", start, end)
    if df.empty:
        return {}
    return {row["date"]: float(row["value"]) for _, row in df.iterrows()}


def _hy_state(hy: dict[date, float], asof: date) -> dict:
    """Credit-spread state as of entry: level, 20-trading-day change (widening>0),
    and whether above the 50-obs moving average (risk-off proxy)."""
    if not hy:
        return {"level": None, "chg20": None, "above_ma50": None}
    dates = sorted(d for d in hy if d <= asof)
    if not dates:
        return {"level": None, "chg20": None, "above_ma50": None}
    i = len(dates) - 1
    level = hy[dates[i]]
    chg20 = level - hy[dates[i - 20]] if i >= 20 else None
    ma50 = float(np.mean([hy[dates[j]] for j in range(max(0, i - 49), i + 1)]))
    return {"level": level, "chg20": chg20, "above_ma50": (level > ma50)}


def _summ(pnls: list[float]) -> str:
    if not pnls:
        return f"{'(none)':>44}"
    a = np.array(pnls)
    wr = (a > 0).mean() * 100
    exp = a.mean()
    wins, losses = a[a > 0], a[a <= 0]
    pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    worst10 = np.percentile(a, 10)
    return (f"N={len(a):>4}  WR={wr:>5.1f}%  exp={exp:>+6.3f}%  "
            f"PF={pf:>4.2f}  p10={worst10:>+6.2f}%")


def _boot_delta(kept: list[float], full: list[float], n: int = 2000) -> tuple[float, float]:
    """Bootstrap the expectancy delta (kept - full) 95% CI. Positive => filter helps."""
    if not kept or not full:
        return (0.0, 0.0)
    k, fl = np.array(kept), np.array(full)
    rng = np.random.default_rng(12345)
    deltas = []
    for _ in range(n):
        dk = rng.choice(k, size=len(k), replace=True).mean()
        df_ = rng.choice(fl, size=len(fl), replace=True).mean()
        deltas.append(dk - df_)
    return (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def _report_filter(name: str, trades: list[dict], keep_fn) -> None:
    full = [t["pnl"] for t in trades]
    kept = [t["pnl"] for t in trades if keep_fn(t)]
    dropped = [t["pnl"] for t in trades if not keep_fn(t)]
    lo, hi = _boot_delta(kept, full)
    print(f"  {name}")
    print(f"    full   : {_summ(full)}")
    print(f"    kept   : {_summ(kept)}")
    print(f"    dropped: {_summ(dropped)}")
    delta = (np.mean(kept) - np.mean(full)) if kept else 0.0
    verdict = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "noise (CI crosses 0)")
    print(f"    exp delta kept-full = {delta:+.3f}%  95%CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}\n")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    args = ap.parse_args()

    trades = _load_trades(args.trades)
    print(f"Loaded {len(trades)} sniper trades from {Path(args.trades).name}")
    if not trades:
        return

    start = min(t["entry"] for t in trades) - timedelta(days=90)
    end = max(t["entry"] for t in trades) + timedelta(days=5)
    hy = await _hy_oas(start, end)
    print(f"HY OAS obs: {len(hy)}  ({min(hy) if hy else '—'}..{max(hy) if hy else '—'})\n")

    # Tag every trade.
    sv_cache: dict[str, dict] = {}
    si_cache: dict[str, dict] = {}
    tagged = 0
    for t in trades:
        tk = t["ticker"]
        if tk not in sv_cache:
            sv_cache[tk] = _load_json_series(SV_CACHE / f"{tk}.json")
            si_cache[tk] = _load_json_series(SI_CACHE / f"{tk}.json")
        t["sv"] = _latest_before(sv_cache[tk], t["entry"], lag_days=1)
        t["dtc"] = _latest_before(si_cache[tk], t["entry"], lag_days=PUB_LAG_DAYS)
        st = _hy_state(hy, t["entry"])
        t["hy_level"], t["hy_chg20"], t["hy_above_ma50"] = st["level"], st["chg20"], st["above_ma50"]
        if t["sv"] is not None or t["dtc"] is not None:
            tagged += 1
    print(f"Tagged {tagged}/{len(trades)} trades with short data "
          f"(sv coverage={sum(t['sv'] is not None for t in trades)}, "
          f"dtc coverage={sum(t['dtc'] is not None for t in trades)})\n")

    non_bear = [t for t in trades if t["regime"] != "bear"]
    print(f"Baseline: ALL {_summ([t['pnl'] for t in trades])}")
    print(f"Baseline: non-bear (live-faithful) {_summ([t['pnl'] for t in non_bear])}\n")

    # --- Bucketed expectancy by short-volume ratio ---
    print("Sniper expectancy by short-volume ratio at entry:")
    for lo, hi in [(0, 40), (40, 50), (50, 60), (60, 1000)]:
        b = [t["pnl"] for t in trades if t["sv"] is not None and lo <= t["sv"] < hi]
        print(f"  sv [{lo},{hi}): {_summ(b)}")
    print()
    print("Sniper expectancy by days-to-cover at entry:")
    for lo, hi in [(0, 2), (2, 3), (3, 5), (5, 1e9)]:
        b = [t["pnl"] for t in trades if t["dtc"] is not None and lo <= t["dtc"] < hi]
        print(f"  dtc [{lo},{hi}): {_summ(b)}")
    print()
    print("Sniper expectancy by credit-spread state at entry:")
    wide = [t["pnl"] for t in trades if t["hy_chg20"] is not None and t["hy_chg20"] > 0]
    tight = [t["pnl"] for t in trades if t["hy_chg20"] is not None and t["hy_chg20"] <= 0]
    print(f"  HY widening (20d chg>0): {_summ(wide)}")
    print(f"  HY tightening (20d chg<=0): {_summ(tight)}")
    above = [t["pnl"] for t in trades if t["hy_above_ma50"] is True]
    below = [t["pnl"] for t in trades if t["hy_above_ma50"] is False]
    print(f"  HY above 50d MA (stress): {_summ(above)}")
    print(f"  HY below 50d MA (calm):   {_summ(below)}\n")

    # --- Candidate AVOID filters (net effect on the tradable book) ---
    print("=== Candidate avoid-filters (does dropping these lift sniper expectancy?) ===\n")
    _report_filter("drop sv>=55 (high short participation)", trades,
                   lambda t: not (t["sv"] is not None and t["sv"] >= 55))
    _report_filter("drop sv>=50", trades,
                   lambda t: not (t["sv"] is not None and t["sv"] >= 50))
    _report_filter("drop dtc>=3 (crowded short)", trades,
                   lambda t: not (t["dtc"] is not None and t["dtc"] >= 3))
    _report_filter("drop dtc>=5", trades,
                   lambda t: not (t["dtc"] is not None and t["dtc"] >= 5))
    _report_filter("drop HY widening (20d chg>0)", trades,
                   lambda t: not (t["hy_chg20"] is not None and t["hy_chg20"] > 0))
    _report_filter("drop HY above 50d MA (credit stress)", trades,
                   lambda t: not (t["hy_above_ma50"] is True))


if __name__ == "__main__":
    asyncio.run(main())
