# US momentum signal variants

Two prespecified variants of the no-stop W10H2 and W12H4 Top 1 candidates, each run over the same 12 entry-month phases:

1. Skip the most recent month's return when ranking.
2. Hold cash when SPY is at or below its 10-month month-end simple moving average.

`REPORT.md` explains the rules, results, and limitations. `run_variants.py` reruns all 72 paths and checks its 24 baseline paths against Queen's answer key. `variant_paths.csv` contains per-path outcomes. `spy_prices_fmp_2013-2025.csv` is the bundled FMP SPY price history needed for the trend filter and matched benchmark.

Run from the repository root:

```bash
python research-exchange/gpt/signal-variants-20261112/run_variants.py
```
