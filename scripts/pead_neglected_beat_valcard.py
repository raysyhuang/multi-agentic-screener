"""Validation-card gate for the "neglected beat" PEAD candidate.

Byproduct of the guidance-raise test (pead_revaccel_test): the DECELERATING-growth
beat cohort was the better PEAD subset (>10% EPS beat AND YoY revenue growth
decelerating: +2.42%/Sharpe 2.02). That was ONE split found in-sample, so before it
can be taken seriously it must clear the 8-check validation card (deflated Sharpe
corrected for the variants searched, slippage/dispersion/regime robustness).

This runs the cohort through:
  * generate_validation_card (fragility + deflated Sharpe) at several variants_tested
    counts (the deflated Sharpe is sensitive to how much we searched — reported, not
    hidden).
  * run_validation_checks (the pipeline's 8-check NoSilentPass gate).

Cohort trades come from the same unified engine / cost as pead_backtest. Regime per
trade = SPY market regime at the announcement day (PEAD is allowed in every regime,
so regime diversity is a fair check). 2x-slippage returns feed the slippage check.

Usage:
  python scripts/pead_neglected_beat_valcard.py --cache-file outputs/research/ohlcv_polygon_3y.parquet
"""
from __future__ import annotations

import argparse
import asyncio

import numpy as np
import pandas as pd

from scripts.pead_backtest import build_events, run_config
from scripts.pead_revaccel_test import _accel_by_ticker_date
from src.backtest.validation_card import generate_validation_card, run_validation_checks
from src.research.signal_backtest import classify_regime


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default="outputs/research/ohlcv_polygon_3y.parquet")
    ap.add_argument("--min-surprise", type=float, default=10.0)
    ap.add_argument("--stop-atr", type=float, default=3.0)
    ap.add_argument("--target-atr", type=float, default=6.0)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--cost-bps", type=float, default=7.5)
    args = ap.parse_args()

    combined = pd.read_parquet(args.cache_file)
    price_data = {t: g.drop(columns=["_ticker"]).reset_index(drop=True)
                  for t, g in combined.groupby("_ticker")}
    cost = args.cost_bps / 10000.0

    spy = price_data.get("SPY")
    if spy is not None:
        spy = spy.sort_values("date").reset_index(drop=True)

    print(f"Building >{args.min_surprise:.0f}% beat events + revenue-accel tags...")
    events = await build_events(price_data, args.min_surprise)
    accel_cache: dict = {}
    for ev in events:
        tk = ev["ticker"]
        if tk not in accel_cache:
            accel_cache[tk] = await _accel_by_ticker_date(tk)
        sd = ev["signal_date"]
        cand = [d for d in accel_cache[tk] if d <= sd]
        ev["accel"] = accel_cache[tk][max(cand)] if cand else None

    decel = [e for e in events if e["accel"] is not None and e["accel"] <= 0]
    print(f"Neglected-beat cohort (decelerating growth): {len(decel)} events\n")

    base_trades, _ = run_config(decel, stop_atr=args.stop_atr, target_atr=args.target_atr,
                                hold=args.hold, cost=cost)
    slip_trades, _ = run_config(decel, stop_atr=args.stop_atr, target_atr=args.target_atr,
                                hold=args.hold, cost=2 * cost)  # 2x slippage

    returns = [t["pnl_pct"] for t in base_trades]
    slippage_returns = [t["pnl_pct"] for t in slip_trades]

    # Regime per trade = SPY market regime at the announcement day.
    def _regime_at(sd) -> str:
        if spy is None:
            return "unknown"
        window = spy[spy["date"] <= sd]
        return classify_regime(window) if len(window) >= 50 else "unknown"

    by_regime: dict[str, list[float]] = {}
    for t in base_trades:
        by_regime.setdefault(_regime_at(t["signal_date"]), []).append(t["pnl_pct"])

    arr = np.array(returns)
    print(f"cohort: N={len(arr)}  WR={np.mean(arr > 0):.1%}  avg={arr.mean():+.3f}%  "
          f"raw_sharpe(x50)={arr.mean() / arr.std(ddof=1) * np.sqrt(50):.2f}")
    print("regime counts:", {k: len(v) for k, v in by_regime.items()})
    print("regime WR:", {k: round(float(np.mean(np.array(v) > 0)), 3) for k, v in by_regime.items()})
    print()

    # --- Deflated Sharpe / fragility at several search sizes ---
    print("Validation card at several variants_tested (deflated Sharpe is search-sensitive):")
    print(f"{'variants':>9}{'deflSharpe':>12}{'fragility':>11}{'robust?':>9}")
    print("-" * 41)
    cards = {}
    for v in (1, 6, 12, 20):
        card = generate_validation_card("pead_neglected_beat", returns, by_regime,
                                        slippage_returns, variants_tested=v)
        cards[v] = card
        print(f"{v:>9}{card.deflated_sharpe:>12.3f}{card.fragility_score:>11.1f}"
              f"{str(card.is_robust):>9}")
    print()

    # Primary card = honest search size for THIS candidate (revaccel splits x
    # thresholds + the E1/PEAD quality sweep it came out of ~= 12).
    card = cards[12]
    print("Primary card (variants_tested=12):")
    print(f"  total_trades={card.total_trades}  win_rate={card.win_rate:.3f}  "
          f"avg_pnl={card.avg_pnl_pct:+.3f}%")
    print(f"  dispersion={card.performance_dispersion:.3f}  "
          f"slippage_sensitivity={card.slippage_sensitivity:.3f}  "
          f"deflated_sharpe={card.deflated_sharpe:.3f}")
    print(f"  fragility_score={card.fragility_score:.1f}  is_robust={card.is_robust}")
    for n in card.notes:
        print(f"    - {n}")
    print()

    # --- Pipeline 8-check gate (NoSilentPass) ---
    run_date = max(t["entry_date"] for t in base_trades)
    run_date = run_date.date() if hasattr(run_date, "date") else run_date
    signal_dates = [t["signal_date"] for t in base_trades]
    execution_dates = [t["entry_date"] for t in base_trades]
    rr = (args.target_atr / args.stop_atr)  # 6/3 = 2.0 for every trade
    payload = run_validation_checks(
        run_date=run_date,
        signal_dates=signal_dates,
        execution_dates=execution_dates,
        feature_columns=["eps_surprise_pct", "revenue_yoy_accel", "atr_14", "close"],
        validation_card=card,
        slippage_bps=args.cost_bps,
        risk_reward_ratios=[rr] * len(base_trades),
        min_risk_reward=1.0,
        allowed_regimes={"bull", "bear", "choppy"},
    )
    print("Pipeline 8-check gate (NoSilentPass):")
    for k, val in payload.checks.items():
        print(f"  {val.upper():>4}  {k}")
    print(f"\n  validation_status = {payload.validation_status.upper()}")
    if payload.key_risks:
        print("  key_risks:")
        for r in payload.key_risks:
            print(f"    - {r}")
    print(f"  notes: {payload.notes}")

    print("\nVERDICT:", "PASS — clears the card + gate" if
          (card.is_robust and payload.validation_status == "pass")
          else "FAIL — does not clear the validation bar")


if __name__ == "__main__":
    asyncio.run(main())
