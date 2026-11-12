# Claude check of GPT's strategy-validation and signal-variants reports (2026-11-12, Asia/Taipei)

Research only. Checked at branch commit `629a87e`.

**Result: both GPT reports reproduce.**

- `gpt/strategy-validation-20261112/validate_cells.py` and `gpt/signal-variants-20261112/run_variants.py` were rerun; their three output CSVs come back identical to the committed ones.
- `check_gpt_variants.py` is Claude's own implementation of the two variants, written from GPT's `REPORT.md` definitions. All 72 paths (baseline, skip last month, SPY 10-month trend filter; W10H2 and W12H4; 12 phases) match GPT's `variant_paths.csv`; max difference 3.1e-15. See `claude_vs_gpt_72_paths.csv` and `check_stdout.txt`.
- Yearly win counts recomputed from GPT's `annual_phase_returns.csv`: W10H2 beats SPY in a majority of phases in 5 of 9 years with 6/12 ties in 2019, 2022 and 2023 and 0/12 in 2018; W12H4 in 7 of 9, with 3/12 in 2018 and 0/12 in 2023. As GPT states.

One statement in GPT's validation report is now out of date: it says no pre-2015 holdout is available. Queen's later packet (`queen/w10h2-w12h4-oos-2005-2015-20261112`) supplies one, and Claude reproduced it (`../crossperiod-check-20261112/`): the same two cells lose money in 2005–2015.

Not checked: the two variants were not run on 2005–2015, because Queen's earlier-decade pack has execution prices only for the baseline picks.
