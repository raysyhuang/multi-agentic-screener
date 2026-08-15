# VPS Paper-Mirror Acceptance Criteria

## Purpose

The VPS paper-mirror launcher (`scripts/mas_vps_paper_mirror.py`) enables the afternoon `--check-now` lane to stamp fills (`pnl_pct`) on open paper positions. Until this PR, the VPS mirror had only a morning lane, leaving all paper positions with `pnl_pct: null` because `_evaluate_position` never ran.

This document pre-registers the acceptance criteria for declaring the paper trial successful or failed.

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

Split sniper vs MR. PEAD paper is a separate quarantined stream — track it independently.

If the house rule is stricter (e.g., 60 trades or 30 days), follow that.

## What to Measure

Once the minimum sample is met:

1. **Win rate** (trades with `pnl_pct > 0` / total closed)
2. **Average P&L per trade** (mean `pnl_pct` across closed trades)
3. **Sharpe ratio** (if ≥30 trades with distinct entry dates)
4. **Max drawdown** (portfolio sim at 10-concurrent, $100k start, sniper + MR only)

Compare these to the honest expectation bands from `scripts/export_dashboard_data.py`:

- **Sniper (official)**: ~54.3% WR / +0.54%/trade (truth-matrix Run E, 2026-07-19)
- **MR (official)**: ~52.2% WR / +0.46%/trade (90d reconciliation, provisional n=23)

PEAD paper expectation: ~57% WR / +1.80%/trade (backtest target from `pead_FINDINGS.md`), **unproven live**.

## What Is NOT Expectancy

- `sniper_component_ic.py` `trade_pnl_pct` is a component IC measurement (Spearman rank correlation), not a per-trade expectancy.
- Ranker scoring is noise. The gates (quality filters, capacity, regime, blackout) are the alpha. Do not conflate rank position with expected return.

## Success Criteria

The paper trial succeeds if, after the minimum sample:

- Sniper official: WR ≥ 50%, avg P&L ≥ +0.40%/trade
- MR official: WR ≥ 48%, avg P&L ≥ +0.30%/trade
- PEAD paper: WR ≥ 52%, avg P&L ≥ +1.00%/trade

AND:

- No multi-day unresolved positions caused by a broken afternoon lane (check logs for persistent `--check-now` failures)
- `run-meta.json` attestations show both phases completing successfully on ≥90% of trading days

## Failure Criteria

The paper trial fails if:

- Any official stream (sniper or MR) falls below breakeven (avg P&L < 0) after the minimum sample
- Any official stream's WR falls below 45% after 40 trades
- The afternoon lane does not run for ≥3 consecutive trading days (operational failure)
- More than 10% of trades have `pnl_pct: null` after 5 days from entry (data quality failure)

PEAD paper is quarantined — its failure does not fail the paper trial, but it blocks PEAD from production promotion.

## What Happens After the Trial

- **If successful**: the VPS can promote the paper instance to a live executor after the 30-day paper gate clears. The acceptance band becomes the live SLA.
- **If failed**: diagnose the gap (data feed? execution sim? gate miscalibration?). Do not promote to live. Record findings in `outputs/research/`.

## Related

- Truth-matrix reconciliation: `outputs/research/HANDOFF_gap_through_diagnosis.md`
- PEAD validation: `outputs/research/pead_FINDINGS.md`
- Manual sleeve forensic: `outputs/research/manual_sleeve_reconciliation_FINDINGS.md` (2026-07-24)

## Changelog

- **2026-08-15**: Pre-registered acceptance criteria before the afternoon lane goes live. Do not revise bands after the trial starts.
