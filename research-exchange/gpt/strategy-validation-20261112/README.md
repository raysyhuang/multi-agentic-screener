# W10H2 and W12H4 validation

Research-only assessment of the no-stop US Top 1 cells W10H2 and W12H4.

- `REPORT.md`: verdict, reproduction results, benchmark comparison, limitations, and next validation gate.
- `validate_cells.py`: produces phase-by-phase and annual benchmark tables from the tracked Queen answer key.
- `spy_prices_fmp_2014-2025.csv`: SPY EOD close and dividend-adjusted close pulled from FMP on 2026-11-12.
- `phase_benchmark_comparison.csv`: 12 matched 10-year phase outcomes per cell and SPY comparisons.
- `annual_phase_returns.csv`: annual outcomes for the 12 overlapping cohorts, 2016–2024.

Run from the repository root:

```bash
python research-exchange/gpt/strategy-validation-20261112/validate_cells.py
```
