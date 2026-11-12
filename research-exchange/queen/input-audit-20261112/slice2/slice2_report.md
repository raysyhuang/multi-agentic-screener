# Slice 2: fund-filing membership audit (IVV holdings vs. frozen pack)

Prepared 2026-11-12 (Asia/Taipei). Research only. The frozen pack, the engine, the repo and main were not changed.

## Verdict
**No missing or false member changes any annual Top1, and none changes the 60.5383% annual-champion headline.**
- The independent IVV holdings snapshots match the pack's membership **one-to-one** at 2016-12-30, 2017-12-29 and 2019-12-31 through 2024-12-31. The only residuals at those dates are explained by a dual-key representation and fund lag.
- One **real gap** was found: the reconstruction drops three acquirers that were S&P 500 members before their mergers. All three were seasoned members (held in the 2013-12-31 IVV N-Q).
  - **Tyco (TYC):** missing for signals 2014-12-31 to 2016-08-31.
  - **ACE Ltd (ACE):** missing for 2014-12-31 to 2015-12-31.
  - **Actavis (ACT):** missing for 2014-12-31 to 2015-02-27.
- None of the three would rank Top1 or even Top3:
  - Annual dates: ACT +53.2% vs LUV +124.6% in 2014 (71.4pp short). ACE and TYC are even further behind.
  - Every monthly signal in their gap windows, all W1–W12 (444 checks): the closest any came was 17.2pp below Top1 (TYC, 2016-01-29, W1). None enters Top3, so no K1–K3 path can change.
- Every annual Top1 and runner-up was in the fund both at S and one year before S (365-day seasoning holds independently).

## Source series and why
- **Fund:** iShares Core S&P 500 ETF (IVV), series S000004310, iShares Trust CIK 1100663.
- **Why IVV:**
  - It has a Dec-31 fiscal-quarter schedule in every year, so period-ends land exactly on the year-end signals.
  - The 2019+ filings are NPORT-P (XML with CUSIP/ISIN).
  - The 2013–2018 filings are N-Q full schedules.
  - It is a full-replication fund, and the holdings carry no sampling.
- 
- Filings, all verified in-document as "December 31" schedules: see `slice2_filings_cited.csv` (accession, filing date and URL for each).

| Fund period-end | Form | Accession | Filed | Compared pack signal |
|---|---|---|---|---|
| 2013-12-31 | N-Q | 0001193125-14-076476 | 2014-02-28 | (seasoning reference for 2014) |
| 2014-12-31 | N-Q | 0001193125-15-065725 | 2015-02-26 | 2014-12-31 |
| 2015-12-31 | N-Q | 0001193125-16-480933 | 2016-02-26 | 2015-12-31 |
| 2016-12-31 | N-Q | 0001193125-17-065125 | 2017-03-01 | 2016-12-30 |
| 2017-12-31 | N-Q | 0001193125-18-063612 | 2018-02-28 | 2017-12-29 |
| 2018-12-31 | N-Q | 0001193125-19-059323 | 2019-03-01 | 2018-12-31 |
| 2019-12-31 | NPORT-P | 0001752724-20-038725 | 2020-02-27 | 2019-12-31 |
| 2020-12-31 | NPORT-P | 0001752724-21-040719 | 2021-02-25 | 2020-12-31 |
| 2021-12-31 | NPORT-P | 0001752724-22-046281 | 2022-02-25 | 2021-12-31 |
| 2022-12-31 | NPORT-P | 0001752724-23-039243 | 2023-02-24 | 2022-12-30 |
| 2023-12-31 | NPORT-P | 0001752724-24-043113 | 2024-02-27 | 2023-12-29 |
| 2024-12-31 | NPORT-P | 0001752724-25-043800 | 2025-02-27 | 2024-12-31 (outside window) |
| 2025-12-31 | NPORT-P | 0002071691-26-004238 | 2026-02-25 | pack 2025-11-28 (pack has no 2025-12-31 signal; supplement) |

When Dec 31 fell on a weekend (2016, 2017, 2022, 2023), no trading day separates the pack signal from the fund date.

## Method
1. **Parse.**
   - N-PORT: equity lines (assetCat EC) with name, CUSIP and ISIN. I excluded the BlackRock Cash Funds (STIV) and the E-mini futures (DE).
   - N-Q: the IVV "Schedule of Investments" table rows from COMMON STOCKS up to the first cash-fund line. Footnote markers were stripped. This excludes cash, securities-lending collateral and futures.
2. **Map to pack tickers.**
   - N-PORT: CUSIP/ISIN went through OpenFIGI. Remaining names were resolved with a prefix alias table (`alias_prefix.json`).
   - N-Q: normalised-name dictionary built from the N-PORT matches, the Wikipedia current and changes tables, and a manual alias table (`nq_alias.json`).
   - A match counts if the ticker equals the pack engine_ticker or the ticker_at_signal.
   - Dual-class lines (GOOG/GOOGL, FOX/FOXA, NWS/NWSA, DISCA/DISCK, UA/UAA, BF.B, BRK.B, LEN) were checked by hand.
3. **Diff.** Each snapshot was diffed against `eligibility_by_signal.csv.gz` (member_at_signal) at the matched signal.
4. **Seasoning.**
   - For every fund name the pack marks eligible_final=False, I checked whether the fund held it one year earlier.
   - For every pack-eligible name, I checked whether the fund held it one year earlier under the same engine key.
5. **Missing names.** For each, I computed the trailing price return on Massive grouped closes (the pack basis) and compared it with the pack Top1 and runner-up.

## Results by signal (`slice2_fund_vs_pack_by_signal.csv`)

| Signal | Fund equity lines | Pack members | Matched 1:1 | Fund-only | Pack-only |
|---|---|---|---|---|---|
| 2014-12-31 | 502 | 499 | 499 | TYC, ACE, ACT (real gap) | – |
| 2015-12-31 | 504 | 502 | 502 | TYC, ACE (real gap) | – |
| 2016-12-30 | 505 | 506 | 505 | – | CPRI (dual key of KORS; dedupe) |
| 2017-12-29 | 504 | 505 | 504 | – | CPRI (dual key of KORS; dedupe) |
| 2018-12-31 | 506 | 505 | 505 | FRC (fund pre-positioned; index add effective 2019-01-02) | – |
| 2019-12-31 | 505 | 505 | 505 | – | – |
| 2020-12-31 | 505 | 505 | 505 | – | – |
| 2021-12-31 | 505 | 505 | 505 | – (Meta = pack FB, same CUSIP) | – |
| 2022-12-30 | 503 | 503 | 503 | – | – |
| 2023-12-29 | 503 | 503 | 503 | – | – |
| 2024-12-31 | 503 | 503 | 503 | – | – |
| 2025 supp. | 503 | 503 (11-28) | 499 | ARES, CVNA, FIX, CRH (Dec-2025 adds) | K, LKQ, MHK, SOLS (Dec-2025 removals) |

**False members: none.** CPRI/KORS is one company carried under two engine keys. The v18 dedupe ensures it ranks once:
- 2016: KORS ranks, because CPRI is not yet 365-day eligible.
- 2017: KORS is removed by dedupe and CPRI ranks.

## Root cause of the real gap
The fja05680/Wikipedia reconstruction treats the surviving entity of an acquirer-takes-target's-ticker merger as the continuing target ticker:
- Wikipedia logs "TYC removed 2016-09-06 (TYC acquired JCI and retained JCI ticker)".
- It also logs "ACE removed 2016-01-19 (ACE acquired Chubb and retained CB ticker)".

So before each merger, the pack holds only the target (JCI Inc, Chubb Corp, Allergan Inc) and omits the acquirer. Two survivor price series are already handled in the pack:
- AGN, through rank_price_overrides (issuer routing to Actavis/Allergan plc from 2015-03-17).
- CB, through the CB_ACE deal leg.

**Unhandled, low impact:** I found no JCI cash-leg / identity entry in corporate_actions_applied for the Tyco/JCI merger (old JCI received 1 JCI plc share + cash). JCI lookbacks spanning 2016-09 may be slightly understated. JCI was never near Top1, so I have not quantified it.

## Rank threats (`slice2_rank_threats_annual.csv`, `missing_member_rank_threats_monthly.csv`)

| Signal | Candidate | 12m price return | Pack Top1 (return) | Gap | Flips? |
|---|---|---|---|---|---|
| 2014-12-31 | ACT | +53.22% | LUV (+124.63%) | 71.4pp | No (not even #2; EA +104.9%) |
| 2014-12-31 | ACE | +10.96% | LUV | 113.7pp | No |
| 2014-12-31 | TYC | +6.87% | LUV | 117.8pp | No |
| 2015-12-31 | ACE | +1.71% | NFLX (+134.38%) | 132.7pp | No |
| 2015-12-31 | TYC | −27.29% | NFLX | 161.7pp | No |

- Monthly check, all W1–W12 signals inside each gap window (444 rows):
  - 0 cases would be Top1 and 0 would enter Top3.
  - Closest gaps to Top1: TYC 17.2pp (2016-01-29 W1), ACE 19.5pp (2015-06-30 W1), ACT 24.7pp (2015-01-30 W3).
- Prices:
  - Massive daily/grouped adjusted closes (`massive_missing_member_daily.csv`).
  - Massive ACE equals FMP `CB` (ACE lineage) at 103.53 / 114.88 / 116.85.
  - FMP has no TYC history (empty), so TYC rests on Massive alone. TYC therefore has a single source, but it is >17pp from any flip.

## Seasoning cross-check
- **Pack-ineligible fund names** (`pack_ineligible_vs_fund_seasoning.csv`): in 2015–2024, every one was absent from the prior-year fund snapshot. The pack's 365-day exclusions agree with the fund.
- **2014 cross-check against the 2013-12-31 N-Q:** the 16 ineligible names were absent in 2013, with two exceptions that are new share classes:
  - GOOG Class C (created 2014-04).
  - DISCK (added 2014-08).
  - Both classes' parent lines were present.
- **Pack-eligible names not held a year earlier under the same key** (`pack_eligible_not_in_prior_fund.csv`): 16 cases, all pure renames under the rename bridge. Examples: FB→META, NLOK→GEN, ANTM→ELV, BLL→BALL, PKI→RVTY, ABC→COR, CDAY→DAY, FLT→CPAY, PEAK→DOC. Also HWM (Arconic Inc legal successor) and CPRI/KORS. None is a Top1 or runner-up.
- **LUV and EA at 2014-12-31:** both are present in the 2013-12-31 IVV N-Q ("Southwest Airlines Co.", "Electronic Arts Inc.").

## 2025-12-31 supplement (outside the 10-year window)
- The pack has no 2025-12-31 signal, so I diffed against 2025-11-28.
- The differences are exactly the December 2025 index changes:
  - ARES in for K (2025-12-11).
  - CRH, CVNA and FIX in for LKQ, SOLS and MHK (2025-12-22).
  - None of the adds is seasoned.
- APO, LII and WDAY (added 2024-12-23) become seasoned by 2025-12-31. Their 2025 returns are −12.4%, −20.3% and −16.8%.
- **WDC (spin-adjusted 282.3%) vs MU (239.1%) is unchanged.**

## Optional item 3: CMG over BLL, 2019-07-31 W12 (total-return sensitivity)
- Price basis (pack): CMG 15.9106 / 8.6732 = +83.45%; BLL 71.48 / 38.97 = +83.42%. CMG leads by 0.02pp.
- BLL ex-dividends in (2018-07-31, 2019-07-31] per FMP: 2018-08-31 $0.10, 2018-11-30 $0.10, 2019-02-28 $0.10, 2019-05-31 $0.15, total $0.45. CMG paid no dividends (FMP returns none).
- BLL total return (unreinvested): (71.48 + 0.45) / 38.97 − 1 = **+84.58%**.
- **On a total-return basis BLL overtakes CMG by about 1.1pp and the W12 Top1 at 2019-07-31 flips to BLL.**
- The spec is price-return, so the pack and the headline do not change. This is a convention sensitivity only, and it is not in the annual champion path. I did not compute the impact on paths that use the W12 2019-07-31 signal.

## Not done / unverifiable
- **Optional item 2 (UNKNOWN patch-list draft and path counts): not done this slice.** I prioritised the primary diff.
- TYC prices are single-source (Massive). FMP has no TYC series.
- The N-Q mapping is name-based; the N-Q schedules carry no tickers or CUSIPs. It is validated by the exact 1:1 counts and by a hand check of the dual classes.
- Massive's adjusted CVNA 2024-12-31 close (40.672) looks inconsistent. It is irrelevant here because CVNA was not seasoned.

## Files (this folder)
- `slice2_executive.txt`: 10–20 line verdict.
- `slice2_fund_vs_pack_by_signal.csv`: per-signal counts, residual names, and Top1/runner-up presence at S and S−1y.
- `slice2_missing_extra_names.csv`: every discrepancy with classification and evidence.
- `slice2_rank_threats_annual.csv` and `missing_member_rank_threats_monthly.csv`: returns vs Top1.
- `slice2_fund_holdings_matched_all.csv`: every fund equity line mapped to its pack ticker.
- `pack_ineligible_vs_fund_seasoning.csv` and `pack_eligible_not_in_prior_fund.csv`: seasoning cross-checks.
- `slice2_filings_cited.csv`: filing citations.
- `massive_missing_member_daily.csv`: Massive closes for TYC, ACE and ACT.
- Scripts: `parse_nport.py`, `parse_nq.py`, `figi.py`, `figi2.py`, `match_nport.py`, `match_nq.py`, `seasoning.py`, `missing_rank_threat.py`, `build_outputs.py`, `summary.py`.
