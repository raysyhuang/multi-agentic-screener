# W10H2 and W12H4 validation status

Date: 2026-11-12  
Branch reviewed: `research/momentum-exchange`  
Base commit: `938ac9bbbac26a54bc3f083ab0aa345db4efb076`  
Scope: US equities, Top 1, no stop-loss. Research only.

## Verdict

**Both cells are strong historical research candidates, but neither can yet be certified as an effective strategy.** The arithmetic reproduces, all 12 start-month paths beat the date-matched SPY price and dividend-adjusted benchmarks in this 2015–2025 sample, and the known price corrections do not change either cell's picks or returns. This is still an in-sample result from a grid of 432 configurations. The 12 start-month paths overlap heavily, and the only month-end price history in the frozen pack starts in November 2013, so it cannot provide a pre-2015 holdout. Selection bias and the absence of out-of-sample evidence remain unresolved.

No 30% stop was applied. The written baseline tested here is the no-stop strategy.

## Reproduction and benchmark

I reran `research-exchange/gpt/independent_ranker.py` and `independent_execution.py --verify` against the tracked frozen input pack and Queen answer key. The ranker rebuilt 1,584 signal/lookback tables; the executor processed 5,184 paths (4,718 known, 466 UNKNOWN). Verification reported **zero discrepancies** against the frozen answer key. The input-audit replay with all currently identified fixes also reports no Top 3 changes for W10 or W12; the corrected selections and outcome concerns found so far are confined to W4–W7.

For the two target cells, I independently summarized the 12 Top 1 phase paths and compared each path's exact entry and end dates with SPY. The SPY closes are from FMP's historical EOD full chart; the dividend-adjusted closes are from FMP's historical EOD dividend-adjusted chart, retrieved 2026-11-12 and preserved in `spy_prices_fmp_2014-2025.csv`. SPY is not charged the strategy's transaction costs. The SPY dividend-adjusted comparison is the more economically relevant benchmark, while noting the strategy itself does not credit ordinary dividends.

| Cell | Median 10-year CAGR | Phase range | Median max drawdown | Phases beating SPY price | Phases beating SPY dividend-adjusted |
|---|---:|---:|---:|---:|---:|
| W10H2 Top 1 | 45.24% | 33.85%–50.29% | −51.78% | 12/12 | 12/12 |
| W12H4 Top 1 | 43.03% | 30.74%–48.77% | −52.00% | 12/12 | 12/12 |

SPY's median matched-period CAGR is 11.47% on price and 13.41% including distributions. The strategy drawdowns are substantial: worst phase drawdown is −54.26% for W10H2 and −58.09% for W12H4.

Annual checks are less uniformly favorable than the full-period result. Across the 12 overlapping phase paths, W10H2 beat SPY in a majority of phases in 5 of 9 calendar years from 2016–2024, with ties in 2019, 2022, and 2023; W12H4 did so in 7 of 9. Neither beat SPY in every year: W10H2 lost all 12 comparisons in 2018; W12H4 lost all 12 in 2023 and 2018 was 3/12. These are descriptive counts, not independent statistical trials.

The adjacent-window check shows both selected cells sit in a generally strong W10–W12 region, but the exact cells are not obviously robust optima. For W10, H2's 45.24% median is above neighboring H1/H3/H4/H5/H6 results (34.65%, 39.99%, 40.69%, 36.92%, 40.40%). For W12, H4's 43.03% is above H1/H2/H3/H5/H6 (32.24%, 39.00%, 40.43%, 34.73%, 38.95%). Because these were selected after scanning the grid, that local strength cannot remove selection bias.

## What is and is not validated

- **Validated:** the independent Python ranker/executor reproduces the frozen answer key; the two selected cells have 12/12 complete paths; benchmark comparisons use matching dates; known audited ranking corrections to date do not alter W10H2 or W12H4.
- **Not validated:** future performance; an unseen historical period; significance adjusted for choosing among 432 configurations; robustness to membership reconstruction, ordinary-dividend accounting, market regimes, and vendor data errors beyond the identified audit. The pack itself remains marked provisional, and a fully corrected versioned pack is not yet available.
- **Operational risk:** a roughly 52% median maximum drawdown (and almost 58% in one W12H4 phase) is large. A high CAGR alone does not establish that the strategy is tolerable or investable.

## Next validation gate

The current pack has no usable pre-2015 price history for a holdout. The recommended next gate is a prospective paper test with rules locked before the first signal: run W10H2 and W12H4 as separate no-stop, Top 1 portfolios; use the frozen ranking, trade timing, and 25 bp one-way turnover cost; credit ordinary dividends consistently; log every signal, selected ticker, fill assumption, cash flow, and daily NAV; compare against SPY total return on the same dates. Track 12 staggered monthly-start cohorts from Dec 2026 through Nov 2027, each for 60 months. Do not change a rule during the test.

For a practical preregistered pass rule, require **each** cell to meet all three: (1) median annualized net return across the 12 cohorts exceeds SPY total return by at least 3 percentage points; (2) at least 9 of 12 cohorts beat SPY total return; and (3) no cohort's maximum drawdown is worse than −60%. Report 36-month results as an interim checkpoint; assess the formal gate after all 60-month cohorts finish. These thresholds are an operational standard for this study, not a universal definition of a valid strategy.

## Reproduction

From the repository root:

```bash
python research-exchange/gpt/independent_ranker.py
python research-exchange/gpt/independent_execution.py --verify
python research-exchange/gpt/strategy-validation-20261112/validate_cells.py
```

The last command writes `phase_benchmark_comparison.csv` and `annual_phase_returns.csv`. Inputs are the tracked answer key and bundled SPY observations. The first two commands also regenerate output files in the existing GPT folder; keep or restore those working-tree outputs as appropriate. This report was added only under the GPT research folder. No main change, PR, strategy freeze change, or frozen-pack/answer-key edit was made.
