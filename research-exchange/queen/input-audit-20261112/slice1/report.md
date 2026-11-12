# Momentum 432 frozen inputs: external truth audit, slice 1
Prepared 2026-11-12 (Asia/Taipei). Research only. Nothing was committed or pushed, and the frozen pack, engine and answer key were not changed. I did not touch Cat, MAS, Bee or any live book.

Scope: this audit checks whether the frozen 2026-11-09 inputs (`/workspace/momentum432-inputs-pack-full-20261109/`) agree with sources outside Queen's pack and engine. The code-vs-answer-key match (4,718 known paths, 466 UNKNOWN, 0 discrepancies) was already settled, and I did not rerun it.

## 1. Verdict
I found **no contradiction that changes any Top1 pick or any headline number.** The headlines are annual champion 60.5383% CAGR / −36.6137% MDD, W10H2 45.24% and W12H4 43.03%.
- Every Top1 lookback return in W10H2, W12H4 and W12H12 (the annual champion) matches a second vendor (FMP) to a median difference of 0. No rank flips.
- Holding-period returns for the same K1 paths match Massive daily bars to within 1.2 percentage points (pp). The pack's execution series are mostly from FMP, so Massive is the independent side here.
- None of the 466 UNKNOWN paths touches the K1 anchor paths W10H2, W12H4 or W12H12.
- Every recent S&P 500 addition behind a Top1 pick (23 names) matches an official S&P DJI release or a prior grade-A v14 record, to the effective day.
- One genuine data contradiction was found: the FOXA/FOX identity switch in March 2019. It changes no pick (section 4).

**Membership as a whole stays PROVISIONAL.** Its source is a community reconstruction, not S&P. 25 of the 48 pick names have only weak, Wikipedia-derived support. I did not audit, name by name, whether a real member is missing from the pack (one that might have beaten a pick). That is the biggest open risk.

### Counts
| Check | Confirmed externally | Contradicted | Weak / unverifiable |
|---|---|---|---|
| B. Membership of the 48 Top1/annual names | 23 official (17 grade A, 6 A−) | 0 | 25 weak (long-tenure names, years of seasoning slack) |
| B. Effective-date conventions vs Wikipedia | Pack matches S&P effective-open dates (UAL, AVGO, AAP, RCL, PYPL, CPRT, LW, AMD, ALGN, AXON, ENPH, TSLA) | 0 (Wikipedia often lists the prior day, but S&P supports the pack) | — |
| C. Unique UNKNOWN events (11 holding/pick + 1 ranker) | 12/12 event facts confirmed (dates, consideration) | 1 identity contradiction (FOXA/FOX), no pick impact | — |
| C. UNKNOWN label itself | 3 pick_not_tradable (417 paths): justified. SIVB, WBA, FOXA: justified | 0 | 5 events (VIAC, SYMC, CELG, STJ, CTLT; 24 paths) are pack gaps that could be resolved, not data errors |
| D. Top1 lookback returns vs FMP | 256/258 (2 HSP rows have no FMP data) | 0 rank-changing | 2 unverifiable (HSP) |
| D. K1 holding legs vs Massive | 362/378 | 0 return-changing | 16 not comparable (deal/spin legs: HSP, CVC, LB/BBWI) |
| Grade-C corporate actions in the anchor paths | PFE→VTRS ratio and child close confirmed; PPL→TLN ratio confirmed | 0 | TLN first close unverified |

## 2. A: membership file inventory (`membership_inventory.txt`)
- 778 intervals, 765 engine tickers, 1996-01-02 to 2026-08-18.
- **Source, as the pack describes it:** "sp500_hist_upd.csv (public reconstruction)". This is github fja05680/sp500, the "Updated" CSV, with its last snapshot on 2026-08-18. It is largely built from Wikipedia's changes table. **Do not call it official.**
- 4 v14 official-date patches are applied: GMCR removal, CEG addition, and the ARNC/HWM parent rename.
- v14 evidence: 11 transitions (10 grade A, 1 grade B) and 15 pure renames ("A issuer text").
- The pack itself still labels membership dates and corporate-action (CA) evidence PROVISIONAL.
- KORS and CPRI overlap in the source (KORS 2013-11-13..2018-09-19, CPRI 2016-01-04..2020-05-12). This is an artifact of the reconstruction, neutralised by the v18 dedupe.
- Sanity check: the pack has 499–506 members per signal date (`member_counts_by_signal.csv`), which is consistent with a ~500-security index with dual classes.
- **Circularity warning:** agreement between the pack and Wikipedia is not independent confirmation, because the pack was built from Wikipedia. Only S&P DJI releases, issuer releases and SEC filings count as external.

## 3. B: membership of the names that mattered (`membership_top1_external_check.csv`, `top1_membership_pack.csv`)
Universe checked:
- 258 unique (W, signal, Top1) events across W10H2, W12H4 and W12H12, covering 47 names.
- The annual picks for signals 2014-12-31 to 2024-12-31.
- The 2025-12-31 WDC vs MU case.

That is 48 names in total.

**Officially confirmed (23).** Each name's interval start (or relevant rename or removal) matches the S&P effective open to the day:

| Name | S&P 500 effective open (or event) |
|---|---|
| AVGO | 2014-05-08 |
| ALGN | 2017-06-19 |
| CPRT | 2018-07-02 |
| AMD (re-add) | 2017-03-20 |
| AXON | 2023-05-04 |
| ENPH | 2021-01-07 |
| PYPL | 2015-07-20 |
| TSLA | 2020-12-21 |
| LVS | 2019-10-03 |
| UAL | 2015-09-03 |
| LW | 2018-12-03 |
| AAP | 2015-07-09 |
| RCL | 2014-12-05 |
| MNST | 2012-06 (S&P release located; day not read) |
| NFLX, CVC | 2010-12 (S&P release located; day not read) |
| META (FB added) | 2013-12-23 |
| HSP | removal 2015-09-03 |
| PLTR | v14, grade A |
| GEV | v14, grade A |
| CEG | v14, grade A |
| BBWI (LB rename) | v14, grade A |
| HWM | v14, grade A |

URLs are in the CSV. Six of these are A− because the effective wording came from a search extract of the S&P page rather than a full read.

**Tightest 365-day seasoning margins**, all officially confirmed:

| Name | Slack beyond 365 days |
|---|---|
| PLTR | 7 days |
| AVGO | 21 days |
| CEG | 25 days |
| GEV | 28 days |
| ALGN | 42 days |

BBWI is eligible only through the rename bridge. That bridge rests on the v14 grade-A S&P text: "L Brands will remain in the S&P 500 … name and symbol change".

**Weak (25):** AMZN, BLL, CMG, DVN, EA, EW, FCX, GWW, LB, LRCX, LUV, MPC, MRO, MU, NEM, NRG, NTAP, NVDA, OXY, TPR, TRIP, TSN, WDC, WYNN and XLNX.
- All are long-tenure members, with years of slack at their pick dates.
- The only support is the reconstruction plus Wikipedia, which shows no removal in the window.
- I fetched no S&P release for these names. They are not contradicted, but they are not officially confirmed.

**Annual champion picks** (`annual_champion_check.csv`):

| Year (signal) | Pick |
|---|---|
| 2015 | LUV |
| 2016 | NFLX |
| 2017 | NVDA |
| 2018 | NRG |
| 2019 | AMD |
| 2020 | AMD |
| 2021 | NVDA |
| 2022 | DVN |
| 2023 | OXY |
| 2024 | NVDA |
| 2025 (outside the 10-year CAGR window) | NVDA |
| 2026 | WDC |

- All picks are seasoned members under the rule. AMD's 2017-03-20 re-addition is confirmed by S&P.
- The smallest margin over the runner-up is 5.4 pp: NVDA over PYPL at 2020-12-31. FMP returns for both names agree with the pack (NVDA +122.1% vs pack 121.9%; PYPL identical), so the pick does not flip.
- Notable excluded names were excluded correctly under the 365-day rule. TSLA in 2020 (added 2020-12-21), PLTR and VST in 2024 (added 2024-09-23 and 2024-05) were all not yet seasoned.

**Contradictions in B:** none for any pick name.

## 4. C: the 466 UNKNOWN paths (`unknown_unique_events.csv`)
The 466 path rows reduce to 11 unique events plus 1 ranker event:
- **pick_not_tradable, 417 paths, 3 events.**
  - ANDV, 76 paths: MPC closed the deal 2018-10-01 and ANDV stopped trading before the open.
  - SCG, 288 paths: Dominion closed 2019-01-01 and SCG stopped trading 2019-01-02.
  - TWTR, 53 paths: the deal closed 2022-10-27. S&P removed TWTR prior to the open on 2022-11-01, matching the pack's membership end, so TWTR was still a member on the 10-31 signal with a stale price.
  - All three are confirmed: the stock could not be bought on the trade date. Turning these paths into known results is a **rule choice** (skip to the next rank or hold cash), not a data fix.
  - Minor note: FMP carries an ANDV bar dated 2018-10-01, which cannot be a real trade. It is harmless because the engine marks ANDV not_tradable.
- **corporate_action_unresolved, 49 paths, 8 events.** All are externally documented:
  - SIVB (15 paths): halted 2023-03-10; S&P removal effective 2023-03-15 matches the pack. The label is justified. Any resolution would make these paths worse.
  - CELG (9 paths): $50 cash + 1 BMY share + 1 CVR, which later expired worthless. A pack gap; resolvable.
  - VIAC (7 paths): a **pure rename** to PARA on 2022-02-17. A pack gap; resolvable.
  - STJ (6 paths): $46.75 cash + 0.8708 ABT. Resolvable.
  - WBA (6 paths): $11.45 cash plus a non-transferable DAP right. Only bounds are possible.
  - FOXA (4 paths): justified, but see the contradiction below.
  - SYMC (1 path): a **pure rename** to NLOK on 2019-11-05, plus a $12 special dividend. Resolvable.
  - CTLT (1 path): $63.50 cash on 2024-12-18. Resolvable.
- **Ranker, 12 UNKNOWN tables at 2025-11-28:** DuPont completed the Qnity spin on 2025-11-01 at 1 Q per 2 DD, with Q trading regular-way from 2025-11-03, matching the pack's ex date. This is resolvable with Q's first close. It affects no 10-year anchor window.
- **None of the 466 touches the K1 anchors.** The only W10 or W12 rows are VIAC paths at H12, K2/K3.

**Contradiction found: FOXA/FOX identity in March 2019.**
- The pack's `massive:FOXA` series and its FOXA/FOX month-end rank prices switch with no deal leg from old 21st Century Fox (~51.36 on 2019-03-18) to the new Fox Corp (40.34 on 2019-03-19).
- Membership treats new Fox Corp as continuously seasoned since 2004. In fact it is a new issuer (CIK 1754301) from 2019-03-19, and the pack's own rule says spin children never inherit seasoning.
- **Impact:**
  - The affected rank windows (signals 2019-03 to about 2020-02) show a fake drop of about −27% for FOXA/FOX, so the error could only push them out of the top, never into it.
  - Checked: there is no FOX/FOXA pick in any rebalance between 2019-03 and 2020-03.
  - The 4 held paths are already UNKNOWN.
  - No pick or headline change. Recommendation for a later patch (not applied): split the identity and add the 21CF→DIS/FOX deal leg.

## 5. D: price spot-check
The rank prices are Massive grouped closes, and the execution series are mostly FMP. So I compared each side against the other vendor.
- **Lookback, Top1 (`price_check_lookback_top1.csv`):** 256 of 258 events compared with FMP split-adjusted closes.
  - Median absolute difference 0. Only 2 events exceed 1 pp:
    - NVDA 2017-06-30 W12: −1.6 pp, from rounding of split-adjusted sub-$1 prices.
    - BBWI 2021-08-31 W12: a spin-convention difference. The pack applies F = 0.7727 using VSCO's first regular-way close of 58.23, which FMP confirms. FMP's own vendor adjustment gives +184.0% against the pack's +197.0%. BBWI stays Top1 either way, since the runner-up TPR is at +173.7%.
  - **0 rank flips.**
- **Tight margins (`tight_margin_rank_check.csv`):** I rechecked the 20 Top1-vs-runner-up margins under 2 pp, using FMP for the runner-up where available. 0 flips.
  - The knife-edge case is **CMG over BLL at 2019-07-31 W12, by 0.02 pp** (0.08 pp under FMP). Price-return basis keeps CMG on top. A total-return basis was not tested and could flip it.
- **Holding legs (`price_check_holding_k1.csv`):** 362 of 378 K1 spell legs compared with Massive daily adjusted closes.
  - Max difference 1.18 pp (NVDA 2016-07-01→2017-07-03). Only 3 legs exceed 0.5 pp, all NVDA split-scale rounding.
  - The 16 legs not comparable are cash deals and spin legs (HSP, CVC, LB/BBWI).
- **Annual champion second source:** this reuses `annual-champion-original-audit-20261107/holding_second_source_jan.csv`, which found all Jan holdings within 0.1% between FMP and Massive. It is now extended to every W10H2, W12H4 and W12H12 leg.
- **2026 annual (WDC vs MU, outside the pack's window):** the earlier SanDisk finding is confirmed.
  - WDC +284.4% (pack-style F on Massive) or +282.3% (FMP vendor-adjusted), against MU +239.1%. So WDC is Top1.
  - Without the spin adjustment WDC shows +188.9%, which would wrongly pick MU.
  - The ratio (1 SNDK per 3 WDC, WDC 8-K) and both closes on 2025-02-24 (SNDK 48.60, WDC 49.02) match in both vendors.
- **Caveat:** FMP and Massive agree to ~1e-6 on most closes. They probably draw on the same consolidated tape, so agreement shows correct extraction and split handling, not two independent histories.

## 6. Grade-C corporate actions inside the anchor paths (`anchor_gradeC_events_check.csv`)
All 36 K1 anchor paths are PROVISIONAL_C only because of two rank-side spin factors:
- **SPIN_PFE_VTRS:** ratio 0.124079 (Pfizer completion release) and VTRS first close of 16.34 (FMP) are both confirmed.
- **SPIN_PPL_TLN:** ratio 0.124906 (PPL release) against the pack's 0.1249, a rounding difference that changes F by under 0.001%. TLN's first close of 18.76 is unverified because TLN is delisted.

Both events only adjust PFE and PPL lookbacks, and neither name is near the top. Upgrading them would not change any number.

## 7. Still unverifiable / residual risk
1. **Members missing from the pack** (a real member absent from the reconstruction that would have outranked a pick) was only sanity-checked by member counts, not audited name by name. This is the largest open item.
2. 25 long-tenure pick names rest on weak (Wikipedia-derived) membership evidence.
3. Vendor agreement does not mean the history is independent (see section 5).
4. Dividends are excluded by design, and the CMG/BLL 0.02 pp knife-edge could flip on a total-return basis. The rule is price-only, so this is a robustness question, not an error.
5. HSP's price history has no second vendor on the current FMP plan.
6. The 2026 rotation (WDC) and the 2025 NVDA year are outside the pack's 10-year CAGR window. The headline 60.54% covers 2015-01-02 to 2025-01-02 only.
7. Code-level agreement plus this slice still say nothing about out-of-sample robustness. Jan is the best of 12 entry months, chosen after the fact (already disclosed).

## 8. Recommended next slice
1. **Members missing from the pack:** for the 10 annual signal dates (and then the W10/W12 signal dates), rebuild the eligible set from an independent snapshot. Options are SEC N-Q/N-PORT holdings of an S&P 500 index fund (e.g., SPDR S&P 500 ETF Trust) near each date, or S&P DJI quarterly rebalance releases. Diff it against the pack and rank any extra names.
2. **Patch proposal, drafted only, not applied:** VIAC→PARA and SYMC→NLOK bridges; CELG, STJ and CTLT deal legs; WBA bounds; FOXA/FOX identity split; DD/Qnity factor. Then report how many of the 49 corporate_action_unresolved paths and 12 ranker tables would become known. The anchors are not expected to change.
3. Run a total-return sensitivity on the tight-margin Top1 cases, CMG/BLL first.

## Files (all in /workspace/momentum-input-audit-20261112/)
- `report.md`, `executive.txt`, `membership_inventory.txt`
- `top1_membership_pack.csv`: pack intervals and seasoning for every Top1 (signal, name)
- `membership_top1_external_check.csv`: external evidence per name, with URL and grade
- `wikipedia_changes_parsed.csv` and `sources/`: Wikipedia raw fetch 2026-11-12 (weak lead only); S&P PDFs and text for LW, CPRT, AMD, UAL, AAP and RCL
- `member_counts_by_signal.csv`
- `unknown_unique_events.csv`
- `anchor_gradeC_events_check.csv`
- `annual_champion_check.csv`
- `price_check_lookback_top1.csv`, `tight_margin_rank_check.csv`, `price_check_holding_k1.csv`
- `massive_spot_closes.csv`, `fmp_prices/` (FMP pulls 2026-11-12)
- Scripts: `top1_membership.py`, `parse_wiki.py`, `price_check.py`, `holding_check.py`, `massive_query.sql`
