# Queen independent recheck: US W10–12 × H2–4 (2026-11-12)

Queen independently recomputed a third-party analysis of the US momentum band W10–12 × H2–4 (Top1/Top2/Top3, 12 entry months) from the frozen Queen v18 answer key. Research only. All figures remain PROVISIONAL. No 30% trailing stop was applied. This does not replace the locked annual-champion rule, and it is not a live-trade or official MAS conclusion.

This packet does **not** include the answer key or the frozen 2026-11-09 inputs pack, and it does not modify them.

## Verdict

The third-party direction holds: this band is a research candidate, not a verified strategy. Top1’s ~40% CAGR is largely NVDA. Quantitative claims match to ordinary rounding, or differ only by median-of-even and switch-day conventions.

## Queen numbers (standard median of 108 paths = average of 54th and 55th)

- Top1: 108/108 computable. CAGR min 20.54%, median **40.42%**, max 64.96%. Median max drawdown −54.26%, worst −69.45%.
- Their 40.1% is the lower-middle value (40.12%); their 2020→end 50.6% is the upper-middle value (50.59%). Queen standard medians: 40.42% and 50.44%. First half (entry to last session of 2019): median 30.92%, worst 0.96%.
- Top2 / Top3 medians: 29.46% / 21.39%.
- SPY (FMP dividend-adjusted close, no fees): 12.97% CAGR from 2015-01-02 to 2025-01-02; 2018 calendar return −4.57%. In 7 of 9 years 2016–2024, a majority of the 108 paths beat SPY; they lagged only in 2018 (0/108) and 2023 (35/108).
- NVDA: 36.36% of holding days (pooled); 79.44% of pooled log return (their 77% differs by switch-day fee/return split). Treating NVDA periods as cash: median CAGR 6.77% with all fees kept (their figure was ~8%).
- W9 H2–4 Top1 cell medians: H2 16.74%, H3 27.46%, H4 28.33%. Calendar-year median contrast: 2016 W9 +11.29% vs W10–12 +85.59%; 2022 −20.57% vs +34.30%. The 2016 gap is mainly NVDA (W10–12) vs NEM/FCX (W9); 2022 is mainly DVN/OXY vs NVDA/MOS/ENPH.
- H2–4 is only slightly above other holds (cell-median average 40.9% vs 36.3%); H6 (39.6%) and H8 (40.5%) are similar, so the edge is weak. 35 of 36 W10–12 × H1–12 K1 cell medians sit in 30%–45%; W10H2 is 45.24%.

See `REPORT.md` and `comparison.csv` for the claim-by-claim table.

## How to read the outputs

`recompute.py` still hard-codes the local answer-key path `/workspace/momentum432-answer-key-full-20261109/`. Independent reruns should point that `AK` variable at a local copy of the same v18 answer key (paths_summary, nav_daily, rebalances, trades). `spy_daily.csv` is already in this folder; the script does not fetch the network. The CSVs and `REPORT.md` in this directory are the official results of the 2026-11-12 run and can be reviewed without rerunning.
