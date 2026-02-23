# Multi-Engine Tuning Decision (2026-02-23)

## Scope

Config-first robustness tuning for multi-engine synthesis with a 1Y/3Y/5Y evaluation gate.

Primary optimization target:
- `cred_synth` (long-horizon robustness first)

Acceptance requirements:
- 5Y improvement threshold (PF/Sharpe/drawdown criteria)
- No material 1Y/3Y regressions beyond guardrails

## Candidates Evaluated

### `cred_long_memory_conservative`

Status:
- Rejected early (pruned after 1Y + 3Y)

Reason:
- `cred_synth` degraded materially on 3Y (negative trade-PnL points), failing balanced guardrails before 5Y.

### `quality_floor_40`

Config changes (vs baseline):
- `min_confidence: 40`
- `top_n_per_day: 4`
- longer rolling credibility memory (`window: 40`)
- tighter credibility cap (`2.0`)
- softer diversity multipliers
- backtest regime-gate aggregation mode `average`

Result:
- Improves 1Y `cred_synth` PF/Sharpe
- Regresses 3Y `cred_synth` PF/Sharpe and drawdown
- 5Y `cred_synth` does not clear the required improvement threshold

Evaluator outcome:
- `5Y_ok = False`
- Guardrails failed (`4`)

## Final Evaluation Update (5Y completed)

### `cred_mild_window30_cap22`

Final 1Y/3Y/5Y evaluator outcome:
- `guardrails = 0` (1Y/3Y guardrails pass)
- `5Y_ok = False`
- Final note: `5Y improvement threshold not met`

Key final 5Y `cred_synth` (candidate):
- PF `1.044`
- Sharpe `0.118`
- trade-PnL points `+512.08`
- drawdown points `1040.73`

Interpretation:
- Strong 1Y improvement and acceptable 3Y tradeoffs were not enough.
- The candidate did not improve long-horizon robustness enough to justify promotion under the plan's acceptance criteria.

## Decision

Do not promote any tuning candidate from this pass.

Recommended action:
- Keep current baseline configuration (`configs/multi_engine_backtest.yaml`) unchanged as production/backtest default.

## Why

The tested candidates produced improvements in isolated windows (mainly 1Y), but not a balanced improvement across 1Y/3Y/5Y. The plan explicitly prioritized long-horizon robustness without material medium-horizon regression, and neither candidate satisfied that.

## Artifacts

Baseline (existing):
- `backtest_results/multi_engine/multi_engine_2025-02-20_2026-02-20.json`
- `backtest_results/multi_engine/multi_engine_2023-02-20_2026-02-20.json`
- `backtest_results/multi_engine/multi_engine_2021-02-20_2026-02-20.json`

Candidates:
- `backtest_results/multi_engine/multi_engine_2025-02-20_2026-02-20_cred_long_memory_conservative.json`
- `backtest_results/multi_engine/multi_engine_2023-02-20_2026-02-20_cred_long_memory_conservative.json`
- `backtest_results/multi_engine/multi_engine_2025-02-20_2026-02-20_quality_floor_40.json`
- `backtest_results/multi_engine/multi_engine_2023-02-20_2026-02-20_quality_floor_40.json`
- `backtest_results/multi_engine/multi_engine_2021-02-20_2026-02-20_quality_floor_40.json`
- `backtest_results/multi_engine/multi_engine_2025-02-20_2026-02-20_cred_mild_window30_cap22.json`
- `backtest_results/multi_engine/multi_engine_2023-02-20_2026-02-20_cred_mild_window30_cap22.json`
- `backtest_results/multi_engine/multi_engine_2021-02-20_2026-02-20_cred_mild_window30_cap22.json`
- `backtest_results/multi_engine/multi_engine_2025-02-20_2026-02-20_cred_mild_window30_cap22_conv12.json`
- `backtest_results/multi_engine/multi_engine_2023-02-20_2026-02-20_cred_mild_window30_cap22_conv12.json`

Evaluator:
- `scripts/evaluate_multi_engine_robustness.py`

## Next Candidate Suggestions (if continuing)

Prioritize candidates that target 5Y without tightening quality filters as aggressively:
- keep `top_n_per_day=5`
- keep `min_confidence=35`
- test regime-gate aggregation mode + moderate credibility cap changes only
- test mild credibility smoothing (Phase 6 `rolling_credibility_blend_alpha`) if config-only changes continue to fail
