# H1 — PEAD outside the S&P 500: REJECTED at G1 (2026-09-19)

**Verdict: REJECTED.** On the survivorship-free universe, the pre-registered primary cohort (E1-gated beats on names outside the S&P 500) has a 20-session excess return of **+0.19%**, with a date-cluster 95% CI of **[−0.58, +0.95]**. The pass rule required at least +0.50% and a CI lower bound above 0. The neglected-beat variant is **−0.22%**, CI [−1.46, +1.02].

**The consequence goes further than H1.** Live PEAD trades mostly outside the S&P 500. Measured on the whole universe live samples from, E1 is **+0.36%**, CI [−0.32, +1.03], and neglected-beat is **+0.08%**, CI [−1.04, +1.22] (post-hoc, § 4). The +1.8 to +2.4%/trade figures that justified the paper sleeve came from a current-S&P-500 sample replayed backwards. They do not describe the population the sleeve trades.

## 1. Pre-registration

The criteria are in the docstring of `scripts/h1_pead_wide.py`. That file was committed in `5f6235a` and pushed at **2026-09-19T04:45:51Z**. At that time the wide price parquet did not exist; its provenance shows `generated_at` 2026-09-19T05:06:08Z, and the run itself happened after that. Verify with `git log --format='%h %ad' --date=iso -- scripts/h1_pead_wide.py`.

- **Primary cohort.** E1-gated beats: EPS surprise ≥ 10%, revenue surprise ≥ 2%, and a signal-bar reaction in [+2%, +12%]. Only names not in the S&P 500 list are included, and each must be liquid as of the prior close (price ≥ $5, 20-session mean dollar volume ≥ $2M). These are the live gates, unchanged.
- **Metric.** Excess return from the entry-bar open to the close 20 sessions later, measured against the same-day, same-liquidity-tercile mean of all eligible names.
- **G1 pass condition.** All three of the following:
  - (a) mean excess ≥ +0.50%;
  - (b) date-cluster CI lower bound > 0;
  - (c) positive in ≥ 2 of the 3 years with n ≥ 30.

## 2. Result (`outputs/research/h1_pead_wide.json`)

| Universe | Cohort | n | Entry dates | Excess 20d | Cluster 95% CI | Hit |
|---|---|---|---|---|---|---|
| **non-S&P** | **E1 (primary)** | **1,797** | 414 | **+0.19%** | **[−0.58, +0.95]** | 48.4% |
| non-S&P | E1 + neglected | 614 | 264 | −0.22% | [−1.46, +1.02] | 45.1% |
| non-S&P | raw beat ≥10% | 11,776 | 654 | +0.23% | [−0.09, +0.55] | 48.0% |
| S&P 500 | E1 | 308 | 166 | +1.36% | [+0.32, +2.47] | 52.6% |
| S&P 500 | E1 + neglected | 88 | 69 | +2.15% | [+0.12, +4.69] | 53.4% |
| S&P 500 | raw beat ≥10% | 1,587 | 350 | +0.77% | [+0.27, +1.29] | 51.1% |

G1 on the primary cohort:

| Condition | Result |
|---|---|
| (a) mean excess ≥ +0.50% | ✗ |
| (b) cluster CI lower bound > 0 | ✗ |
| (c) positive in ≥ 2 of 3 years | ✓ (3 of 4 years positive: 2023 −0.88, 2024 +0.42, 2025 +0.29, 2026 +0.22) |
| **PASS** | **false** |

Supply: the non-S&P E1 cohort produces about 500–590 events a year, roughly 5× the S&P rate. Supply is not the constraint; edge is.

**Descriptive only — none of these can rescue the verdict:**
- By liquidity tercile, non-S&P E1 is:
  - least liquid: +1.25%, CI [−0.07, +2.59];
  - middle: +0.27%;
  - most liquid: **−1.71%**, CI [−3.10, −0.28].
- The 60-session horizon is stronger than the 20-session horizon in most cells.

## 3. Reproduction check

The S&P 500 subset reproduces the published event counts.
- On the 504-name S&P cache: raw 1,550 vs the published 1,558–1,574; E1 301 vs the published 306.
- On the wide panel the counts are 1,587 and 308. The price window runs to 2026-09-18, and the base rate now includes every eligible name.

The toolkit is therefore counting the same events the engine backtests counted. The difference is the comparison: a same-day, same-bucket base rate rather than absolute return. The bull tape alone was worth +1.8 to +2.2% per 20 sessions to an average name in this window.

## 4. Post-hoc decomposition (`scripts/h1_decomposition.py`, `outputs/research/h1_decomposition.json`)

This was written after the verdict. It is descriptive, and every cut counts as a variant.

1. **Current S&P 500 membership is look-ahead.** With no event at all, today's S&P 500 members beat the same-day matched base by **+0.19% per 20 sessions** (1.21M name-days; positive in 2023, 2024 and 2026). Non-members come in at −0.04%. Replaying today's member list backwards selects the names that went on to do well, so every S&P-only backtest in this repo carries this tailwind.
2. **The population live trades.** On the whole survivorship-free universe:

   | Cohort | n | Excess 20d | Cluster 95% CI |
   |---|---|---|---|
   | E1 | 2,105 | +0.36% | [−0.32, +1.03] |
   | E1 + neglected | 702 | +0.08% | [−1.04, +1.22] |
   | raw beat ≥10% | 13,363 | +0.29% | [+0.01, +0.57] |

   Raw beat's 60-session excess is +0.93%, CI [+0.39, +1.46].
3. **Member vs member.** Measured against an S&P-only same-day base, S&P E1 is +1.11% with CI [−0.00, +2.29], and neglected is +1.79% with CI [−0.22, +4.36]. Neither CI clears zero.
4. **Same liquidity tercile, opposite signs.** In the most liquid tercile, S&P E1 is +1.31% and non-S&P E1 is −1.71% with a CI entirely below zero. The large names that are *not* current members are mostly names that fell out of, or never reached, the index. The sign flip is consistent with membership look-ahead. It cannot be proven without a point-in-time constituent list, which the halted PIT-universe project was meant to provide.

**Leads, to be re-registered only as new hypotheses and tested on data not used here:**
- The least-liquid tercile: E1 +1.25%; raw beat +0.83%, CI [+0.32, +1.34].
- The 60-session horizon.

## 5. What this changes

- **PEAD paper sleeve.** The honest expectation for the live-traded population is roughly **+0.2 to +0.4% excess per 20 sessions, not distinguishable from zero**. It is not the +0.9% forward expectation or the +1.8 to +2.4% backtest figure. The measurement window opened 2026-09-19 under `docs/paper_sleeve_acceptance_criteria.md`, and its Tier-2 bar (cluster CI lower bound > 0 on alpha) is unchanged. This finding makes clearing that bar unlikely, and says so in advance.
- **Every S&P-only backtest in this repo is survivorship-biased by about +0.2% per 20 sessions**, before any strategy effect.

## 6. Caveats

- **Timing.** FMP does not say whether a report came pre-market or after the close. For an after-close report, the signal bar predates the news and its "reaction" is the wrong day. This discards some real E1 beats; it cannot manufacture them.
- **Universe.** The candidate universe comes from quarter-start liquidity sampling (types CS/ADRC, 4,723 names, S&P 500 coverage 100%). A name that was liquid only between sample dates is missing. The precise eligibility screen is the per-date, prior-close screen in `src/research/event_study.build_panel`.
- **Earnings coverage.** 4,696 of 4,723 names are cached. 104 are empty (funds with no earnings). 27 fetch failures are recorded and were not cached (`outputs/research/event_universe_earnings_manifest.json`).
- **No execution model.** This is Stage-0 excess return: open to close, no stops, no costs. G2 was never reached.

## Provenance

- **Prices.** `outputs/research/ohlcv_polygon_wide_3y.parquet`:
  - sha256 `2d04af2db42ab0519614522a7e8cbd0f38d9532c01b26c34f4e3933e07d25e0c`
  - 4,722 tickers, 3,219,830 rows, 2023-07-10 → 2026-09-18
  - provider `polygon`, fallback `null`, requested 4,724, returned 4,722
  - missing: `AXIAPC`, `GEVW` (both "empty_or_malformed_response")
  - Sidecar: `.provenance.json`
- **Earnings.** FMP `/stable/earnings?symbol=` per ticker, cached in `data/cache/earnings/`.
- **Candidates.** `outputs/research/event_universe_candidates.json`, built by `scripts/build_event_universe.py`.

## Reproduce

```bash
python scripts/build_event_universe.py --stage all
python scripts/h1_pead_wide.py --json-out outputs/research/h1_pead_wide.json
python scripts/h1_decomposition.py --json-out outputs/research/h1_decomposition.json
```
