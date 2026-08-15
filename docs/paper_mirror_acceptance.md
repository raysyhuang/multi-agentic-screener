# VPS Paper-Mirror Acceptance Criteria

## Purpose

The VPS paper-mirror launcher (`scripts/mas_vps_paper_mirror.py`) enables the afternoon `--check-now` lane to stamp fills (`pnl_pct`) on open paper positions. Until this PR, the VPS mirror had only a morning lane, leaving all paper positions with `pnl_pct: null` because `_evaluate_position` never ran.

This document pre-registers the acceptance criteria for evaluating the paper trial.

## What Changed

- **Before**: VPS mirror ran only the morning lane (`--run-now`). Open paper positions never received `pnl_pct` because the afternoon check (`--check-now`) was not scheduled.
- **After**: The in-repo launcher supports `--phase {morning,afternoon}`. The afternoon phase runs `worker --check-now` + export (NO alembic, NO briefs), stamping `pnl_pct` on open positions.

## What Did NOT Change

- Gate logic (sniper gates, MR gates, PEAD gates)
- Ranker scoring
- Signal models
- Pick count (`max_final_picks`, `sniper_max_positions`, etc.)
- Entry/exit engine parameters (trail, stops, slippage)

The launcher is pure orchestration. All alpha-bearing code is unchanged.

## Pre-Registration: Do Not Cite WR/Hit/Return Until Afternoon Has Stamped pnl_pct

The retracted 85.7% WR claim (2026-07 issue) was built on `null` pnl_pct rows that had never resolved. Any performance claim made before the afternoon lane stamps fills is premature.

**Minimum sample before any WR/hit/return claim:**

- 40 resolved paper trades with non-null `pnl_pct`, OR
- 20 closed trades over ≥15 trading days,
- **whichever comes second**.

Split sniper vs MR. PEAD paper is a separate quarantined stream — track it independently. Never pool.

## What to Measure

Once the minimum sample is met:

1. **Win rate** (trades with `pnl_pct > 0` / total closed)
2. **Average P&L per trade** (mean `pnl_pct` across closed trades)
3. **Sharpe ratio** (if ≥30 trades with distinct entry dates)
4. **Max drawdown** (portfolio sim at 10-concurrent, $100k start, sniper + MR only)

## Comparison: Published Live Books, Not Backtest

Compare against the already-published live performance, **not** truth-matrix or backtest bands:

- **MAS-GH sniper** (GitHub-hosted prod): ~50% WR / −0.97% avg (reconciled through 2026-08-13)
- **IBKR sniper** (live paper): ~42% WR / −0.14% avg (reconciled)

A new number that ignores these live results is backtest theater. The question is whether the VPS paper mirror reproduces the GHA/IBKR results, not whether it matches a backtest.

## What Is NOT Expectancy

- **Retracted**: 85.7% WR — fill artifact from `pnl_pct: null` rows. Never cite it.
- `scripts/sniper_component_ic.py` `trade_pnl_pct` is a component IC measurement (Spearman rank correlation), not a per-trade expectancy.
- Ranker scoring captures 2–7% of selection value (Spearman ≈ 0). The gates (quality filters, capacity, regime, blackout) are the alpha. Do not retune scores in this trial.

## After the Bar Is Met

Once the minimum sample is reached and the results are measured:

1. **Decide whether to keep the VPS paper mirror running**, tighten gates, or test sniper pick-count adjustments.
2. **Do NOT say the VPS "promotes to a live executor."** Live promotion is a separate human decision and out of scope for this trial.
3. Record findings in `outputs/research/` with full provenance (sample size, date range, reconciled vs un-reconciled, data source).

## Operational Health

The paper trial also validates the launcher's operational reliability:

- **No multi-day unresolved positions** caused by a broken afternoon lane (check logs for persistent `--check-now` failures)
- **`run-meta.json` attestations** show both phases completing successfully on ≥90% of trading days
- **Less than 10% of trades** have `pnl_pct: null` after 5 days from entry (data quality)

## Related

- MAS-GH vs IBKR reconciliation: internal records (2026-08-13)
- Truth-matrix reconciliation: `outputs/research/HANDOFF_gap_through_diagnosis.md`
- PEAD validation: `outputs/research/pead_FINDINGS.md`
- Manual sleeve forensic: `outputs/research/manual_sleeve_reconciliation_FINDINGS.md` (2026-07-24)
- VPS Boston mirror review: `outputs/research/REVIEW_vps_boston_mirror_2026-08-10.md` (F2 dotenv-precedence fix)

## Changelog

- **2026-08-15**: Pre-registered acceptance criteria before the afternoon lane goes live. Compare against published live books (MAS-GH, IBKR), not backtest. Do not revise bands after the trial starts.
