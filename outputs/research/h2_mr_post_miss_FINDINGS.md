# H2 — MR avoid-filter after an earnings miss: REJECTED at G1 (2026-09-19)

**Verdict: REJECTED.** At the live threshold (min_score 75), MR trades entered within 10 sessions of a ≤ −10% EPS miss do lose money: **−0.30% per trade** (n=1,167). They underperform the other MR trades by **−0.28 percentage points**, and the difference is negative in all four years.

The pre-registered test also required the difference to be established: the date-cluster 95% CI upper bound had to be below 0. It is **+0.08**. The rule says in advance that "right sign, CI crosses zero" is a fail, because that is exactly how the short-interest and days-to-cover filters looked before they reversed. So the verdict is REJECTED.

## Pre-registration

The criteria are in the docstring of `scripts/h2_mr_post_miss.py`, committed in `d44fc9c` and pushed at **2026-09-19T05:14:02Z**. The first run started after that; `generated_at` in `outputs/research/h2_mr_post_miss.json` is later. Unlike H1 and H4, the wide price parquet already existed at commit time, and the docstring says so. What the timestamp proves is that no H2 computation had run.

**Trades.** MR is taken from `scripts/gen_mr_trades.LIVE_MR` at **min_score 75** (the live selectivity) and run through the unified exit engine with gap-through fills. A trade is kept only if the name was liquid as of the prior close. That leaves 48,815 of 56,587 trades on 4,722 names.

**POST_MISS.** The ticker's most recent report on or before the MR signal had an EPS surprise ≤ −10%, and the MR signal came ≤ 10 sessions after that report's signal bar.

## Result

| Test | Value | Pass? |
|---|---|---|
| (a) n(POST_MISS) ≥ 30 | 1,167 | ✓ |
| (b) post-miss − rest, cluster CI upper bound < 0 | −0.28pp, CI **[−0.64, +0.08]** | ✗ |
| (c) mean pnl(POST_MISS) < 0 | −0.30%/trade (win rate 48.7% vs 50.9%) | ✓ |
| (d) negative in ≥ 2 of the years with n ≥ 10 | 4/4 (2023 n=10: −5.66; 2024: −0.17; 2025: −0.23; 2026: −0.32) | ✓ |
| **PASS** | | **false** |

## Descriptive cells

These cells are counted as variants. None of them can rescue the verdict.

| Cell | n flagged | Flagged mean | Difference | Cluster 95% CI |
|---|---|---|---|---|
| window 5 sessions | 596 | −0.32% | −0.30pp | [−0.86, +0.25] |
| **window 20 sessions** | 2,439 | −0.27% | −0.27pp | **[−0.50, −0.04]** |
| miss ≤ −5% | 1,433 | −0.23% | −0.22pp | [−0.56, +0.12] |
| mirror: post-**beat** ≥ +10% | 2,598 | +0.11% | +0.14pp | [−0.13, +0.41] |
| min_score 50 (not live) | 8,424 | +0.02% | −0.03pp | [−0.27, +0.20] |

## Reading

- **The direction is consistent everywhere.** Post-miss MR trades are worse at every window and threshold, and post-beat trades are better. That is what PEAD's short leg predicts.
- **The effect is small next to its noise.** About −0.28pp per trade, spread across a few hundred entry dates.
- **The 20-session window clears zero, but only as a post-hoc result.** It was one of five descriptive cuts, and picking the best of five after seeing them is how the gap-continuation "winner" was manufactured. It is registered as a **lead** to be tested on data this study did not use.
- **The low-selectivity population shows nothing.** At min_score 50 the difference vanishes (−0.03pp). This is the reverse of the usual pattern, where an effect lives only at low selectivity and dies at live selectivity. Here, whatever there is appears only in the live-selected trades. It is the one respect in which H2 looks better than the filters that died before it.
- **Live MR has no raw edge on this universe either.** At live selectivity, the non-flagged MR trades average −0.01%. That is consistent with the registry's standing finding that MR's thin live edge is selection, not mechanics.

## What this changes

Nothing in production. The standing lesson from the short-interest and days-to-cover filters is not to ship a filter whose CI crosses zero. The effect is logged as a lead: a post-miss exclusion window of about 20 sessions. It can be re-tested on MR trades after 2026-09-18, which this study never saw. At ~400 flagged trades a year, a decisive forward read needs about a year.

## Caveats

- **Stage-0 on backtested MR, not live MR.** The trades come from the unified exit engine at live parameters, not from the live book. The live book has n=49 and is too small to condition on.
- **Timing and universe.** The after-close timing issue and the universe construction are the same as H1 (`h1_pead_wide_FINDINGS.md` § 6).
- **Ordering.** A report's signal bar on the same day as the MR signal counts as 0 sessions and is flagged. This is correct only if the report came before the MR decision; the backtest decides MR from the same bar's close, so the ordering holds.

## Provenance

- **Prices.** `outputs/research/ohlcv_polygon_wide_3y.parquet`, sha256 `2d04af2db42ab0519614522a7e8cbd0f38d9532c01b26c34f4e3933e07d25e0c`. Provider `polygon`, no fallback; 2 of 4,724 requested tickers are missing.
- **Earnings.** FMP per-ticker cache.

## Reproduce

```bash
python scripts/h2_mr_post_miss.py --json-out outputs/research/h2_mr_post_miss.json
```
