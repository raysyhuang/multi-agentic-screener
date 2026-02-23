# Multi-Engine Backtest Tuning Configs

Candidate configs for the 5Y robustness pass. These are starting points for the
matrix in the tuning plan, not guaranteed winners.

Suggested usage:

```bash
./.venv/bin/python -m src.backtest.multi_engine.orchestrator \
  --config configs/backtest_tuning/cred_long_memory_conservative.yaml \
  --start 2021-02-20 --end 2026-02-20
```

Compare outputs with:

```bash
./.venv/bin/python scripts/evaluate_multi_engine_robustness.py \
  --baseline backtest_results/multi_engine/multi_engine_2021-02-20_2026-02-20.json \
  backtest_results/multi_engine/*.json
```
