# W10–12 × H2–4 第三方分析复算（Queen v18 冻结答案表，2026-11-12 台北时间）

All figures are PROVISIONAL (frozen v18 inputs; membership dates and corporate-action audit not re-done). Research only, not a trading instruction. No 30% trailing stop applied. Taiwan out of scope.

## Verdict
- **Every quantitative claim reproduces to rounding or near-rounding. Nothing is materially wrong.**
- Exact/rounding matches: 108/108 computable; min 20.5% / max 65.0%; Top2 29.5%, Top3 21.4%; 7 of 9 years beat SPY, lagging 2018 & 2023; NVDA time ≈36%; W9 H2–4 17%–28%; 2016 (W9 +11.3% vs +85.6%) and 2022 (−20.6% vs +34.3%); halves 30.9% / 1.0%; MDD −54% / −69%; 2018 median −29.6%; SPY ≈13%.
- Near-misses that come down to convention:
  - Top1 median: the standard median of the 108 paths is **40.42%**, not 40.1%. Their 40.1% is the 54th sorted value (40.12%).
  - 2020→end median: the standard median is **50.44%**, not 50.6%. Their 50.6% is the 55th sorted value (50.59%).
  - NVDA log-return share: **79.4%** pooled (80.5% per-path median), vs their 77%. Alternative timing gives 78.6%.
  - NVDA-as-cash median CAGR: **6.8%–7.4%** depending on convention, vs their ~8%.
  - Claim 2: 35 of 36 cells fall in 30–45%. W10H2 is 45.24%, just above the upper bound.
  - SPY 2018: −4.57%, vs their "about −4%".
- **H2–4 is mildly above average, not "not special."** The H2–4 cell medians average 40.9%, against 36.3% for the other H values. They include the #1 cell (W10H2) and the #2 cell (W12H4). Still, H6 and H8 sit at the same level, so the edge is weak.
- **The qualitative conclusion holds under Queen's numbers.** This is a research candidate, not a verified strategy. Top1's ~40% is mostly NVDA: about 79% of log return comes from NVDA, and the cash-flat median CAGR is about 7%.

## Comparison table
| # | Claim | Theirs | Queen recompute | Result |
|---|---|---|---|---|
| 1 | Top1 computable | 108/108 | 108/108 (all PROVISIONAL_C) | MATCH |
| 1 | Top1 CAGR min/median/max | 20.5 / 40.1 / 65.0% | 20.54 / **40.42** / 64.96% | min, max MATCH; median NEAR (their value = lower-middle 40.12%) |
| 2 | W10–12×H1–12 K1 cell medians in 30–45% | 36/36 | 35/36; range 30.19% (W11H12) – 45.24% (W10H2) | NEAR |
| 2 | H2–4 not special | — | H-column means: H2 41.6, H3 39.9, H4 41.2, H6 39.6, H8 40.5, H1 34.7, H9 33.4, H12 32.7% | mild edge only |
| 3 | Majority beat SPY 2016–24 | 7/9, lag 2018 & 2023 | 7/9, lag 2018 (0/108) & 2023 (35/108) | MATCH |
| 4 | Top2 / Top3 median | 29.5 / 21.4% | 29.46 / 21.39% | MATCH |
| 4 | SPY CAGR | ~13% (memory) | 12.97% (2015-01-02→2025-01-02); per-path own window median 13.41% (12.16–15.20%) | MATCH |
| 5 | NVDA log-return share | 77% | 79.44% pooled; 80.53% per-path median | NEAR (+2.4pp) |
| 5 | NVDA holding time | 36% | 36.36% pooled sessions; 36.71% median; equity-weighted median 45.59% | MATCH |
| 5 | NVDA periods as cash, median CAGR | ~8% | 6.77% (all fees kept); 7.08% (NVDA fees removed); 7.43% (alt timing) | NEAR |
| 6 | W9 H2–4 cell medians | 17–28% | H2 16.74, H3 27.46, H4 28.33% | MATCH |
| 6 | 2016 W9 vs W10–12 | +11 vs +86% | +11.29 vs +85.59% | MATCH |
| 6 | 2022 W9 vs W10–12 | −21 vs +34% | −20.57 vs +34.30% | MATCH |
| 7 | Top1→Top3 | 40→21% | 40.42→29.46→21.39% | MATCH |
| 8 | Entry→2019-12-31 median / worst | 30.9 / 1.0% | 30.92 / 0.96% | MATCH |
| 8 | 2020→end median | 50.6% | 50.44% (their value = upper-middle 50.59%) | NEAR |
| 9 | Top1 MDD median / worst | −54 / −69% | −54.26 / −69.45% | MATCH |
| 9 | 2018 median loss / SPY 2018 | −30% / ~−4% | −29.56% calendar return (median in-2018 MDD −49.46%) / SPY −4.57% | MATCH / NEAR |
| 10 | Post-hoc selection | qualitative | not recomputable; the band was chosen after seeing 432 cells | N/A |

## Conventions
- **Universe.** W∈{10,11,12}, H∈{2,3,4}, all 12 phases: 108 paths each for K=1, 2 and 3, all known. The W9 contrast uses 36 paths per K.
- **CAGR.** Taken from `paths_summary.csv`, with years = days/365.25. This was verified against final_multiple: max difference 2e-16.
- **Medians.** A standard 108-path median averages the 54th and 55th sorted values.
- **NAV.** Pre-entry NAV = 1.0. The engine trades at the close.
- **Daily log-return attribution.**
  - A day's return belongs to the holding carried over (t−1, t].
  - On a rebalance day, the market part ln(equity_before/NAV_{t−1}) goes to the outgoing name.
  - The fee is split by leg from `trades.csv`: sell fees go to the outgoing name, buy fees to the incoming name.
  - Alternative timing ("pick as of trade_date ≤ t") is also reported.
- **Cash-flat counterfactual.** NAV is flat on sessions where NVDA is held. All other daily returns are unchanged, and there is no reinvestment into another pick.
- **NVDA identity.** NVDA has a single execution key, `fmp:NVDA`. The band has no NVDA corporate-action events; its only CA events are LB and CVC, listed in `k1_band_corporate_action_events.csv`.
- **Calendar year.** NAV at the last session on/before 12-31 of the prior year, to NAV at the last session on/before 12-31 of the year. SPY is measured over the same dates.
- **Halves.** Split at the last session on/before 2019-12-31. Half 1 runs from 1.0 at entry; half 2 runs to the final multiple. An alternative using full calendar 2016–2019 vs 2020–2024 gives 34.89% / 4.47% / 54.46%; the primary convention matches their figures better.
- **SPY.** FMP `historical-price-eod/dividend-adjusted` adjClose, 2014-12-01 to 2026-02-27, no fees (see `spy_source.txt`).

## 2016 / 2022 selection events (K1 signal picks; identical pick across H at each signal date)
- **2016.**
  - W10–12 picked NVDA from 2016-05 onward, and every month 2016-08 to 2016-12.
  - W9 picked NEM at 2016-04 and 2016-06 to 2016-09, and FCX at 2016-10.
  - Mean NVDA log contribution in 2016: +0.72 per path for W10–12 vs +0.27 for W9.
  - NEM contributed −0.11 for W9.
- **2022.**
  - W10–12 held DVN from 2021-11 to 2022-05, then OXY.
  - W9 picked NVDA at 2021-11/12 (−0.16 log contribution in 2022), MOS at 2022-03, OXY from 2022-05, and ENPH from 2022-10.
  - DVN contributed +0.31 for W10–12 vs +0.16 for W9.
- See `w9_vs_w10_12_ticker_contrib_2016_2022.csv`.

## Files
- `recompute.py`: reads only the answer key and `spy_daily.csv`; no network.
- `comparison.csv`
- `top1_108_path_stats.csv`
- `year_by_year_vs_spy.csv`
- `year_by_year_path_returns.csv`
- `w9_contrast.csv`
- `w9_k1_36_path_stats.csv`
- `nvda_attribution_per_path.csv`
- `half_period_cagr.csv`
- `w10_12_h1_12_k1_median_cagr.csv`
- `w9_vs_w10_12_ticker_contrib_2016_2022.csv`
- `spy_daily.csv` and `spy_source.txt`
