# Sniper Truth Matrix — Findings (2026-07-19)

Data: 503-ticker S&P 500 universe, 2023-05 → 2026-04 (2.74Y), from the cached
`ohlcv_b788de7c4eb3_2023-04-30_2026-04-29.parquet`. All runs go through the
unified exit engine (`src/backtest/exit_engine.py`). Equity columns are a
concurrency-capped account (fixed-fraction 20%, max 3 concurrent = sniper_max_positions).

`eqDD%` = real account-equity max drawdown. (The old headline `total_return_pct`
is a SUM of per-trade returns and its "drawdown" can exceed 100% — not an account
number; deliberately not used here.)

| Run | config | N | WR | avg% | payoff | Sharpe | equity× | CAGR% | eqDD% |
|-----|--------|---|-----|------|--------|--------|---------|-------|-------|
| A baseline | spy on, min70, no-timestop, **no-gap** | 1090 | 92.0% | 4.11 | 0.63 | 5.08 | 796× | 1042% | 7.6% |
| B +fill | spy on, min70, no-timestop, **gap** | 1090 | 60.5% | 0.78 | 0.98 | 1.03 | 3.62× | 59.8% | 17.9% |
| C +timestop | spy on, min70, **1d-timestop**, gap | 1090 | 54.3% | 0.54 | 1.12 | 0.75 | 2.39× | 37.4% | 18.4% |
| D live-as-was | **spy OFF, min60**, 1d-timestop, gap | 2029 | 53.6% | 0.33 | 1.04 | 0.47 | 1.45× | 14.4% | 39.0% |
| **E live-fixed** | spy on, min70, 1d-timestop, gap | 1090 | **54.3%** | **0.54** | **1.12** | **0.75** | **2.39×** | **37.4%** | **18.4%** |

## The headline: the spectacular backtest was a fill-realism artifact

The frozen sniper baseline (~82% WR, +4.13% avg, Sharpe 3.9) was reproduced by
Run A (92% WR here — see caveat below). **Modeling realistic gap-through-open
fills alone (A→B) collapses win rate 92% → 60.5% and the capped-account multiple
from 796× to 3.62×.** The exact-stop-fill assumption — filling every stop at the
exact stop price even when the bar gapped through it — was manufacturing almost
the entire edge. Run A's exit mix is 999/1090 trailing stops with only 81 real
stops and near-identical 90%+ WR in *every* regime including bear, which is not
plausible for a real breakout strategy. This is the root cause of the
backtest-vs-live gap.

The 1-day time_stop (B→C) costs another ~6pp WR and a third of the average
return, confirming it truncates would-be winners.

## The true go-forward strategy (Run E)

After the Phase-1a production fixes (spy_df wired, min_score=70 enforced) and
realistic fills + the live time_stop, the honest expectancy is:

- **54.3% win rate, +0.54%/trade, payoff 1.12, per-trade Sharpe 0.75.**
- Account level (20% fraction, 3 concurrent): **2.39× over 2.74Y = 37.4% CAGR
  with 18.4% max drawdown** (MAR ≈ 2.0). 873 of 1090 signals taken; 217 dropped
  for the position cap.

Positive and, at the account level, respectable — but a different universe from
the 82–92% WR / Sharpe 3.9 fantasy.

## The 1a fixes materially help (E vs D)

Run D (live-as-was: spy off, min60) → Run E (fixed: spy on, min70) improves
CAGR 14.4% → 37.4%, cuts max drawdown 39.0% → 18.4%, and lifts Sharpe 0.47 →
0.75, on roughly half the trades (fewer, better). This validates shipping the
1a fixes to production regardless of the keep/kill decision.

## Sanity check: Run D vs assumed live ~69%

Run D reconstructs the pre-fix production strategy at **53.6% WR** — well below
the ~0.69 working assumption. But 0.69 was never a *measured* live win rate; it
was the equity-curve haircut default. The live-faithful sim landing at ~54% is
consistent with "forward tracking not very good." Most likely the 69% assumption
was optimistic on a small live sample. **Confirm against the measured live
scorecard (`scripts/sniper_scorecard.py`) before final judgment.**

## Caveats

1. **Window is bull-heavy** (2023-05 → 2026-04). Run E bear-regime rows are
   51.9% WR / +0.0% avg — bear contributes nothing (consistent with the existing
   bear block). The 37% CAGR is bull-flattered; expect less in a real bear.
2. **Run A over-reproduces** (92% vs published 82%): the unified engine's
   canonical semantics (entry bar eligible for exits, MFE/MAE ordering) plus a
   different 3Y window. The robust, methodology-independent finding is the
   *relative* collapse A→B→C, not Run A's absolute number.
3. Minute-bar fills (Phase 3) would resolve the remaining same-bar stop-vs-target
   ambiguity and are the natural next refinement of Run E.

## Decision (gate)

The plan's GO bar was per-trade expectancy > ~1% and capped-equity Sharpe > ~1.5.
Run E clears neither on a per-trade basis (0.54%, 0.75) but compounds to a
37% CAGR / 18% DD account. This is a **judgment call, not a clean GO**:

- **Ship the 1a fixes** (done — they are correctness fixes and measurably help).
- **Retire the 82% baseline** as an artifact; hold sniper to the realistic bar
  (~54% WR, +0.54%/trade) in forward tracking.
- **Recommend**: keep sniper live but risk-capped, and re-decide after minute-bar
  fills (Phase 3) and a measured live-scorecard comparison. Do not scale it up on
  the strength of the old backtest.
