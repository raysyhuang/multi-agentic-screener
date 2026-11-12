# Two additional US momentum tests

Date: 2026-11-12  
Branch: `research/momentum-exchange`  
Base commit: `cae048efa326317852eaea17826f51942e970a91`  
Scope: W10H2 and W12H4, Top 1, 12 monthly start phases, 10 years; no stop-loss.

## Result

Neither variant improves median CAGR over the baseline. The one-month skip modestly improves W12H4's drawdown while reducing its CAGR; the SPY 10-month trend filter reduces CAGR substantially and does not improve drawdown. Both are worth retaining as robustness comparisons, not as replacements for the baseline.

| Cell and signal | Median CAGR | Range across 12 phases | Median max drawdown | Worst phase drawdown | Median time in cash | Phases beating matched SPY total return |
|---|---:|---:|---:|---:|---:|---:|
| W10H2 baseline | 45.24% | 33.85%–50.29% | −51.78% | −54.26% | 0.0% | 12/12 |
| W10H2, skip last month | 32.65% | 18.27%–50.43% | −55.19% | −55.35% | 0.0% | 12/12 |
| W10H2, SPY above 10-month SMA | 36.06% | 25.59%–43.70% | −53.62% | −58.73% | 19.0% | 12/12 |
| W12H4 baseline | 43.03% | 30.74%–48.77% | −52.00% | −58.09% | 0.2% | 12/12 |
| W12H4, skip last month | 39.40% | 34.43%–52.33% | −49.35% | −53.48% | 0.2% | 12/12 |
| W12H4, SPY above 10-month SMA | 33.24% | 25.45%–48.65% | −55.39% | −58.09% | 21.6% | 12/12 |

For reference, the W12H12 January-start Top 1 annual champion remains 60.54% CAGR with −36.61% max drawdown on its single January path. It is not directly comparable to the 12-phase medians above: it is one start phase and was selected from the same historical search.

## Exact test definitions

- **Baseline:** existing frozen no-stop Top 1 signal. W10 or W12 month-end total price return, rank on the prior month-end signal, rebalance every H months. Buy/sell at the first session of the month close. One-way turnover cost is 0.25%.
- **Skip last month:** W10-1 or W12-1 formation return. At signal month-end, rank from the close W months earlier through the prior month-end; omit the return in the most recent month. Eligibility is still measured on the current signal date. Rebalance and execution are unchanged.
- **SPY trend filter:** use the baseline stock ranking only when SPY's signal-date month-end close is above its 10-month simple moving average, calculated from the latest ten month-end closes including that signal month. Otherwise sell into cash at the next scheduled rebalance. Cash earns zero. The stock ranking and H schedule remain unchanged.
- **Benchmark:** matched start/end dates, SPY dividend-adjusted close series from FMP, retrieved 2026-11-12. The strategy itself follows the frozen pack convention and does not credit ordinary dividends, so this comparison is conservative for the strategy. SPY trend decisions use unadjusted SPY closes.

## Reproducibility and data limits

`run_variants.py` reruns all 72 path variants. The 24 baseline paths first reproduce Queen's frozen answer key within `1e-10` for final multiple, CAGR, and daily max drawdown. The code uses the full pack daily close table for alternate selections. When the sparse `execution_availability` table has no explicit mapping for an alternate name/date/window, it accepts only a direct `fmp:TICKER` or `acq:TICKER` series that has close observations at both the trade date and next rebalance date; it leaves the path UNKNOWN if neither is present. All 48 modified paths were complete under this rule. Corporate actions and post-hoc untradable deal checks are applied from the same frozen pack.

This is a research replay on the same provisional 2015–2025 sample used to select the original candidates. It is not out-of-sample validation. It also does not credit ordinary dividends to the strategy. Do not interpret 12/12 full-period wins as annual consistency; paths overlap and each result covers the same decade.

## Reproduction

From the repository root:

```bash
python research-exchange/gpt/signal-variants-20261112/run_variants.py
```

The script writes `variant_paths.csv`. Inputs are the frozen pack, answer key, and bundled FMP SPY daily closes.

No shared input, answer key, main branch, or strategy rule was changed.
