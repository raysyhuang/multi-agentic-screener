# H1 — PEAD outside the S&P 500: REJECTED at G1 (2026-09-19, v2 after Codex review)

**Verdict: REJECTED.** The pre-registered primary cohort was E1-gated beats on names outside the S&P 500. It fails under every variant we checked:

| Run | Non-S&P E1, 20-session excess | Date-cluster CI | Moving-block CI |
|---|---|---|---|
| Registered run (v1, `h1_pead_wide.json`) | +0.19% (n=1,797) | [−0.58, +0.95] | — |
| Refreshed earnings, registered timing (v2) | +0.21% (n=1,810) | [−0.56, +0.97] | [−0.66, +1.11] |
| Refreshed earnings, look-ahead-free timing (v2) | +0.38% (n=2,191) | [−0.12, +0.89] | [−0.32, +1.08] |

G1 required an excess of at least +0.50% and a cluster CI lower bound above 0. No row meets either.

**What this does and does not say about the live PEAD sleeve.** At the 20-session hold the sleeve uses, PEAD on the broad research universe is small and not established. With a live-like liquidity floor it is essentially zero. A 60-session horizon looked robust in v2 (§ 4), but **v3 found a look-ahead in the price screen, and once it is corrected the 60-session lead no longer clears zero either (§ 7).**

The v1 version of this document said the live-traded population earns +0.2–0.4% and that S&P-only evidence carries a measured +0.2% survivorship penalty. The Codex review showed neither is established, and both are withdrawn (§ 5).

## 1. Pre-registration

The criteria are in the docstring of `scripts/h1_pead_wide.py`. It was committed in `5f6235a` and pushed at 2026-09-19T04:45:51Z, before the wide price parquet was generated at 05:06:08Z. That ordering covers the recorded artifacts only. It is not proof that no earlier computation occurred.

- **Primary cohort.** E1 beats (EPS surprise ≥10%, revenue surprise ≥2%, reaction in [+2%, +12%]), non-S&P, liquid as of the prior close. Liquid means price ≥ $5 and 20-session mean dollar volume ≥ $2M.
- **Metric.** 20-session excess over the same-day, same-liquidity-tercile base rate.
- **G1 conditions.**
  - (a) mean ≥ +0.50%;
  - (b) date-cluster CI lower bound > 0;
  - (c) positive in at least 2 years with n ≥ 30. The docstring says "3 calendar years"; the code evaluates the 4 observed years.

## 2. Changes after the Codex review, and why

Each change is recorded as an amendment in the script's docstring. None of them changes the registered criteria.

| Defect found | Fix | Effect |
|---|---|---|
| FMP gives a report date but no release time, and neither data plan has release times (Polygon's Benzinga feed returns 403). For an after-close report, the registered "reaction" is the wrong day. That can pass E1 on noise as well as reject real beats. | `--timing volume`: the reaction bar is whichever of the report-date session and the next one traded more volume. Entry is always the session after that pair, which makes a pre-market report enter one day late. This is look-ahead-free. | Primary rises from +0.21% to +0.38%; still fails |
| The earnings cache was never refreshed. The S&P files dated from July, so no S&P name had a report after 07-19. | Refetched 687 stale files (`--refresh-before 2026-09-19`). Output JSONs now carry an earnings fingerprint (4,721 files, sha `5262c738…`). | S&P E1 goes from n=308 to n=335 |
| Adjacent entry dates share most of a 20-session path, and the date-cluster bootstrap resamples them as independent. | Added a circular moving-block CI with block length equal to the horizon, reported next to the cluster CI. | Wider intervals, most visibly for raw beats |
| Events without a forward return were dropped silently. | Each cell now reports `n_no_forward_return`, e.g. 25 for whole-universe E1 at 20 sessions, 20 of which are simply windows that run past the end of the data. The count does not separate end-of-data, missing entry, and a premature missing exit such as a delisting. | "Survivorship-free" is now "survivorship-reduced" |

## 3. Results, refreshed earnings (`h1_pead_wide_v2_{registered,volume}.json`)

The E1 rows are the ones the paper sleeve depends on.

| Cell, excess_20 | Registered timing | Look-ahead-free timing |
|---|---|---|
| non-S&P E1 (primary) | +0.21%, cluster [−0.56, +0.97] | +0.38%, cluster [−0.12, +0.89] |
| non-S&P E1 + neglected | −0.21%, [−1.45, +1.01] | +0.22%, [−0.61, +1.03] |
| S&P E1 | +1.36%, [+0.36, +2.38], block [+0.27, +2.47] | +1.35%, [+0.44, +2.30], block [+0.24, +2.64] |
| **All names, E1** | +0.39%, [−0.29, +1.07], block [−0.35, +1.14] | +0.53%, [+0.10, +0.96], **block [−0.06, +1.13]** |
| All names, E1, live-like liquidity (sensitivity) | **−0.14%**, [−0.94, +0.66] | +0.25%, [−0.29, +0.78] |
| All names, raw beat ≥10% | +0.29%, [+0.01, +0.56], block [−0.18, +0.72] | +0.38%, [+0.12, +0.63], block [−0.05, +0.77] |

"Live-like liquidity" means 20-session mean share volume ≥ 500k and mean dollar volume ≥ $10M, as of the prior close. It does **not** reproduce the live universe's cap and tiering. It only asks whether the result depends on the thinnest names.

## 4. Post-hoc leads

These were found after the verdict. They are registered as leads, not results.

| Lead | Registered timing | Look-ahead-free timing |
|---|---|---|
| **All names E1, 60-session horizon** | +1.28%, cluster [+0.23, +2.34], block [+0.40, +2.27] | +1.32%, cluster [+0.48, +2.15], block [+0.44, +2.19] |
| All names raw beat, 60-session | +0.89%, block [+0.15, +1.62] | +0.87%, block [+0.20, +1.56] |
| All names E1, least-liquid tercile, 20-session | +1.22%, cluster [−0.08, +2.52] | +1.32%, cluster [+0.53, +2.12] |

In v2 the 60-session result looked like the one that held up: positive under both timing rules and both bootstraps, with both CIs excluding zero. **It does not survive the v3 raw-price screen (§ 7): +0.80% / +0.70%, with block CIs spanning zero.**

It is still a lead. It was one of several horizons looked at after the verdict. It also comes from the same 3-year window as everything else here, so it has had no out-of-sample test. The paper sleeve holds 20 sessions. Changing the hold would be a live-config change that needs its own registration and forward data.

## 5. Withdrawn from v1

- **"The live-traded population earns +0.2–0.4%."** The study averages every name passing a $5 / $2M screen. The live universe additionally applies a 500k-share floor, a 1,000-name cap and dollar-volume tiers. With a live-like floor, E1 is −0.14% under registered timing and +0.25% under look-ahead-free timing. No historical reconstruction of live selection exists, so no live expectancy is claimed.
- **"S&P-only backtests carry a measured +0.2% survivorship bias."** With no event at all, current S&P members beat the same-day matched base by +0.19% per 20 sessions (379,030 valid observations; v1 reported 1.21M because it counted NaNs). The equal-weight mean of the daily member means is also +0.19%, and its moving-block CI is **[−0.14, +0.53]**. That interval is for the daily-mean estimand; the observation-weighted mean is almost identical. In addition, 98.4% of member observations sit in the top liquidity tercile, and a tercile is not a size or factor match. The warning stands: replaying today's membership backwards uses future information. The magnitude of the bias is not identified.

## 6. Caveats

- **Timing.** Neither timing rule is the truth; release timestamps would be. The look-ahead-free rule gives up a day on pre-market reports. The registered rule mislabels after-close reports.
- **Universe.** 4,723 CS/ADRC names, sampled at quarter-start liquidity using Polygon's point-in-time listing with `date=`. S&P coverage is 100%. SIVB, FRC and SBNY failed before the window started. Names that were liquid only between sample dates are missing.
- **Earnings.** The study read 4,721 files, of which 103 are empty (funds). The manifest's 105 includes AXIAPC and GEVW, which have no prices. 4 refetch failures are recorded in `event_universe_earnings_manifest.json`; the retained older files for HRL and USB are among them. After the refresh, 453 of 503 S&P names and 3,168 of 4,218 non-members have an actual report after 07-19, so a smaller coverage gap remains. The manifest does not record how many files the v2 refresh attempted (687, taken from the run log); later runs record it as `attempted_this_run`.
- **Scope.** This is Stage 0: open-to-close excess with no stops and no costs. G2 was never reached.

## Provenance

- **Prices.** `ohlcv_polygon_wide_3y.parquet`, sha256 `2d04af2db42ab0519614522a7e8cbd0f38d9532c01b26c34f4e3933e07d25e0c`, provider `polygon` with no fallback. It contains 4,722 of the 4,724 requested names; AXIAPC and GEVW are missing.
- **Earnings.** FMP `/stable/earnings?symbol=`. v2 fingerprint `5262c738a5f4…`.

## Reproduce

```bash
python scripts/build_event_universe.py --stage all --refresh-before 2026-09-19
python scripts/h1_pead_wide.py --timing registered --json-out outputs/research/h1_pead_wide_v2_registered.json
python scripts/h1_pead_wide.py --timing volume     --json-out outputs/research/h1_pead_wide_v2_volume.json
python scripts/h1_decomposition.py --timing volume --json-out outputs/research/h1_decomposition_v2_volume.json
```

## 7. v3 — the price screen looked ahead (found while running H3)

The $5 floor was applied to **split-adjusted** prices. A penny stock that later reverse-splits (MULN, HUBC and the like) shows up far above $5 in its own adjusted history, so the screen admitted it using information about a split that had not happened yet. That affected 2.1% of eligible name-days across 436 tickers, and they are the collapsing names.

Because they sat in the same-day base rates, they pushed most excess returns **up**. `--raw-price-screen` rebuilds the traded price from the split history (`event_study.split_price_multiplier`); details are in `h3_dividends_splits_FINDINGS.md` § 3. The table compares the v2 JSONs with the v3 raw-screen JSONs (`h1_pead_wide_v3raw_{registered,volume}.json`, `h1_decomposition_v3raw_*.json`):

| Cell | v2, adjusted screen (registered / look-ahead-free timing) | **v3, raw screen** |
|---|---|---|
| non-S&P E1, 20 sessions (primary) | +0.21% / +0.38% | **−0.05% / +0.03%** |
| All names E1, 20 sessions | +0.39% / +0.53% | **+0.16% / +0.22%** |
| **All names E1, 60 sessions** | +1.28% / +1.32%, block CIs > 0 | **+0.80%, block [−0.08, +1.78] / +0.70%, block [−0.12, +1.54]** |
| All names raw beat, 60 sessions | +0.89% / +0.87%, block CIs > 0 | +0.51% / +0.50%, block CIs span 0 |
| Least-liquid tercile E1, 20 sessions | +1.22% / +1.32% | +0.86%, cluster [−0.33, +2.09] / +0.70%, cluster [−0.09, +1.47] |
| S&P E1, 20 sessions | +1.36% / +1.35% | +1.30%, block [+0.23, +2.40] / +1.29%, block [+0.18, +2.57] |
| All names E1, live-like liquidity (share volume also un-adjusted in v3) | −0.14% / +0.25% | −0.29%, block [−1.12, +0.50] / +0.11%, block [−0.59, +0.80] |
| Current S&P members, no event | +0.19% | +0.15%, block CI of daily means [−0.19, +0.48] |

**Reading.**

- On the survivorship-reduced universe, with a price screen that no longer looks ahead, PEAD at the sleeve's 20-session hold is about **+0.2%** of excess and not distinguishable from zero.
- The 60-session horizon and the least-liquid tercile, the two leads from § 4, **no longer clear zero** either. They are downgraded from leads to observations.
- The only cell still clearly positive is S&P E1. Its sample is today's index members replayed backwards, and that membership is look-ahead of unknown size (§ 5).
- The H1 verdict is unchanged: REJECTED.

## 8. Final review (Codex, gpt-5.6-sol, on `main` after #123)

Every headline number was reproduced exactly from the scripts. Two caveats bound what this study can support.

- **Membership look-ahead cuts both ways.** "Non-S&P" uses today's S&P 500 list for every date. A name deleted from the index during the window is counted as non-S&P throughout, and a name added later is counted as S&P before it joined. The *non-member* cohort therefore carries look-ahead just as the member cohort does. The H1 rejection is a statement about "names not in today's index", not about point-in-time non-members. Making the literal claim would need historical constituents. The all-names cells are unaffected by this.
- **Delisting returns: the magnitude is not settled.** A name with no bar at the horizon end has no forward return. It is dropped from both the event cohorts and the same-day base rate, instead of being counted at its delisting value. Before the final 20 sessions this covers 11,082 of 2,231,544 eligible name-days (0.5%, 615 tickers), a mix of delistings, suspensions and data gaps. As a stress test, treating every such mid-window disappearance as a loss moves all-names E1 (volume timing, raw screen) as follows:

  | Loss assumed for missing returns | All-names E1 | Non-S&P E1 |
  |---|---|---|
  | none (dropped, as reported) | +0.22% | +0.03% |
  | −50% | +0.49% | +0.30% |
  | −100% | +0.74% | +0.57% |

  These are bounds, not estimates. None of them establishes condition (b), and the verdict stands. What they do mean is that "+0.2%" is not a number to recalibrate the paper sleeve to. **The supported conclusion is directional:** the broad-universe evidence is far weaker than the +1.8–2.4% the S&P-only backtest showed.
- **Fundamentals are not point-in-time vintages.** The session lag keeps each report out until it is public. The report *contents*, however, come from today's FMP cache, so any later restatement of actuals, estimates or report dates would leak in. This is unquantified.
