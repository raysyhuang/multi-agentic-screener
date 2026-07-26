"""MR days-to-cover filter — selectivity robustness check BEFORE the validation card.

The short-credit study found dropping crowded shorts (days-to-cover >= 3) lifts raw
MR from +0.039% -> +0.097%/trade (bootstrap CI [+0.005, +0.110], barely clears 0) at
min_score=50. But the project's recurring trap is that thin MR edges appear at low
selectivity and VANISH at the live gate (the MR-stop "+flip at min_score=60" was a
low-selectivity artifact; MR's real live edge is the OFFICIAL selection funnel, not
the raw mechanics).

So gate the DTC filter on selectivity FIRST: does dropping dtc>=3 still help as the
MR score floor rises toward live? If the kept-full delta decays to noise at higher
min_score, it's an artifact — stop, don't build the full card. If it survives, it
earns the validation card.

Usage:
  python scripts/mr_dtc_selectivity.py --cache-file outputs/research/ohlcv_polygon_3y.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.sniper_short_credit_filter import (
    PUB_LAG_DAYS,
    SI_CACHE,
    _boot_delta,
    _latest_before,
    _load_json_series,
)
from src.research.signal_backtest import run_model_backtest

LIVE_MR = {
    "rsi2_threshold": 10.0,
    "stop_atr_mult": 0.75,
    "target_atr_mult": 1.5,
    "holding_period": 3,
    "trail_activate_pct": 0.5,
    "trail_distance_pct": 0.3,
    "gap_through": True,
}
DTC_THRESHOLD = 3.0


def _summ(pnls) -> str:
    if not pnls:
        return "(none)"
    a = np.array(pnls)
    return (f"N={len(a):>5}  WR={np.mean(a > 0) * 100:>5.1f}%  "
            f"exp={a.mean():>+6.3f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default="outputs/research/ohlcv_polygon_3y.parquet")
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}
    print(f"Loaded {len(price_data)} tickers\n")

    # Cache DTC series per ticker once.
    si_cache: dict[str, dict] = {}

    def _dtc_at(ticker, d):
        if ticker not in si_cache:
            si_cache[ticker] = _load_json_series(SI_CACHE / f"{ticker}.json")
        return _latest_before(si_cache[ticker], d, lag_days=PUB_LAG_DAYS)

    print(f"MR 'drop days-to-cover >= {DTC_THRESHOLD:.0f}' filter vs MR score floor "
          f"(live floor is 50 base / 75 choppy):")
    print(f"{'min_score':>10}{'full':>34}{'kept (dtc<3)':>34}{'delta 95%CI':>26}")
    print("-" * 104)

    for min_score in (50, 60, 70, 75):
        params = {**LIVE_MR, "min_score": float(min_score)}
        result = run_model_backtest("mean_reversion", price_data, params)
        rows = []
        for t in result.trades:
            dtc = _dtc_at(t.ticker, t.signal_date)
            rows.append((t.pnl_pct, dtc))
        full = [p for p, _ in rows]
        kept = [p for p, dtc in rows if not (dtc is not None and dtc >= DTC_THRESHOLD)]
        if not full:
            print(f"{min_score:>10}  (no trades)")
            continue
        lo, hi = _boot_delta(kept, full)
        delta = np.mean(kept) - np.mean(full) if kept else 0.0
        verdict = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "noise")
        print(f"{min_score:>10}  {_summ(full):>32}  {_summ(kept):>32}  "
              f"{delta:>+6.3f} [{lo:>+.3f},{hi:>+.3f}] {verdict}")

    print("\nRead: if the kept-full delta decays to noise (CI crosses 0) as min_score "
          "rises toward the live floor, the DTC edge is a low-selectivity artifact "
          "(same pattern as the rejected MR-stop) and does NOT earn the validation card.")


if __name__ == "__main__":
    main()
