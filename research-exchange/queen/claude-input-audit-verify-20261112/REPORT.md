# Queen verification of Claude's input-data audit (commit 54a6b13), 2026-11-12 (Asia/Taipei)

Research only. Nothing in main, the frozen 2026-11-09 packs or the answer key was changed. No PR, nothing pushed.
Primary price source: Massive (`/v2/aggs/ticker/...` adjusted=true and adjusted=false; `/stocks/v1/splits`). Claude's FMP numbers were not used as confirmation.
Ranking: Queen's own code (`queen_rerank.py`), written from SPEC.md §1–5. Pack baseline reproduces the answer key exactly: 4,699 (signal, W, K) pick lists from `rebalances.csv.gz`, 0 mismatches.

## Headline
- Claude's nine corrections hold. Massive's adjusted bars are unadjusted for all of them, and with only those nine fixed, Top1/2/3/5 change in 0 tables and Top10 in 3. That matches Claude exactly.
- **New error Claude missed: ARNC 2016-11-01 (Alcoa split-up) is adjusted twice.** The pack's pre-ex ARNC prices are Massive's AA series, which Massive had already scaled by 0.801 for the spin. The engine then also applies F=0.7116, and no QADJ de-adjust row exists for ARNC. After the fix, **Top1 changes in 3 tables, Top3 in 5**. ARNC was an answer-key pick in those tables, so **364 answer-key paths rebalance on a changed table**. Claim 4's "results are identical" holds only for Claude's nine fixes, not for a fully corrected pack.
- Same double adjustment for EQT 2018-11-13, but EQT left the index on the ex date, so it has no ranking effect (latent).

## Claim-by-claim
| Claim | Verdict | Queen numbers / source |
|---|---|---|
| 1 nine errors | **CONFIRMED** (all 9) | Massive raw vs adjusted, see the table below and `massive_event_comparison.csv` |
| 2 674 wrong cells, all understate | **CONFIRMED with a caveat** | Own diff: 674 cells, same set as Claude's (0 only-Claude, 0 only-Queen): 386 price-corrected cells (all understated in the pack), 144 IR removed, 144 TT added. Caveat: 66 of the 144 IR cells have lookbacks entirely after the issuer switch (59 positive). They are wrong because of eligibility, not understatement, so "all understate" is literally true only for the 386 + 78 IR cells that cross March 2020. |
| 3 Top1–5 unchanged; Top10 3 tables; TT 4.8pp | **CONFIRMED** (for the nine fixes) | 1,572 ranked tables (the other 12 are 2025-11-28 UNKNOWN in both): Top1/2/3/5 changes = 0; Top10 = 3 (2018-04-30 W4 BKNG 483→9; 2020-07-31 W1 TT new at 7; 2020-10-30 W4 TT new at 8). TT 2020-07-31 W1 is 4.75pp short of Top3 (25.72% vs 30.48%). BKNG best rank 9, and BKNG's closest gap is 10.45pp (2018-03-29 W2, rank 17). Claude's corrected Top3 matches mine in 1,572/1,572 tables. **Not true once ARNC is also fixed** (see C). |
| 4 5,184 + 324 results identical | **CONFIRMED as an artifact check, for the nine fixes only** | SHA-256: paths_5184 pack = corrected `f49fd307…`; stoploss_324 pack = corrected `955cd30e…`; Claude's SHA256SUMS.txt all OK. This follows from Top3 being unchanged. With the ARNC fix, 364 answer-key paths (357 PROVISIONAL_C, 6 CONDITIONAL, 1 UNKNOWN) in W4–W7 cells rebalance on a changed table. They were not rerun. No anchor (W9–W12) or W10–12×H2–4 stop-loss path is touched. |
| 5a 6,801 vendor pairs, 0 >0.5% | **CONFIRMED** | Own code on the pack's FMP execution closes vs rank prices: 6,801 pairs, 0 >0.5% (6,771 if both rank prices must be Massive grouped closes; still 0). `vendor_crosscheck_fmp_exec_vs_massive_rank.csv` |
| 5b member_at_signal = fja05680 on 132 dates | **CONFIRMED** | Downloaded `S&P 500 Historical Components & Changes (Updated).csv` at fja05680/sp500@a2430f2. It is byte-identical (sha256 36326709…) to the pack's source `sp500_hist_upd.csv`. engine_ticker sets equal on 132/132 signal dates (`membership_vs_fja05680.csv`). As Claude says, this shows faithful copying, not that the source is correct (PROVISIONAL). |
| 5c 160/173 join dates within 10 days | **CONFIRMED in pattern; my counts are 157/170** | Used the local Wikipedia snapshot `/workspace/prior-quarter-winner-check-2026-10-20/sp500_wiki.html` (older snapshot, 3 fewer qualifying names): 170 added 2013-06..2024-12, 146 exact, 157 within 10 days. The 13 outliers are the same kind Claude lists (BKR, CPAY, D, EG, FOXA, FOX, HWM, IR, META, NWSA, SRE, TROW, WTW). |

## A. The nine corrections against Massive (primary source)
Rows marked raw==adj: Massive returned byte-identical responses for adjusted=true and adjusted=false over the window.

| Event | Massive evidence | Pack value | Pack state | Why the adjustment is missing | Verdict |
|---|---|---|---|---|---|
| GOOGL 2014-04-03 class C dividend | Old GOOG (class A) adj 2014-03-31 55.7255 = raw 1114.51/20; GOOGL 2014-04-03 raw 571.50, GOOG(C) 569.74; Massive split table has **no** 2014-04-03 record for GOOG/GOOGL (only 2022-07-18 20:1) | 55.7255 (alias GOOG) | Only the 2022 20:1 split is applied; the 2014 class C dividend is not | The share-class dividend is missing from Massive's split calendar | **CONFIRMED.** Factor from Massive closes 571.50/(571.50+569.74) = 0.50077 (Claude 0.50050) |
| GOOG same event | GOOG engine membership starts 2014-04-03, so pre-04-03 prices are never inside an eligible lookback (0 affected cells) | 55.7255 | unadjusted | same | **CONFIRMED, harmless** |
| TMK 2014-07-02 3:2 | TMK raw==adj: 2014-06-30 81.92, 2014-07-01 82.39, 2014-07-02 54.83; split recorded only under **GL** (2:3, factor 0.666667) | 81.92 | unadjusted | Split stored under the successor ticker GL; bars under the retired symbol TMK are not adjusted | **CONFIRMED.** Factor 2/3 |
| DISCA 2014-08-07 class C dividend | DISCA raw==adj: 08-06 83.02, 08-07 40.975; DISCK 08-07 39.78; recorded only under **WBD** (stock_dividend 1:2) | 85.21 (07-31) | unadjusted | Stored under the successor WBD; bars under the retired DISCA are not adjusted | **CONFIRMED.** Massive-implied factor 40.975/(40.975+39.78) = 0.5074 (Massive table 0.5; Claude 0.5110). Ranking effect identical |
| BF.B 2016-08-19 2:1 | BF.B raw==adj: 08-18 99.65, 08-19 50.40; Massive table lists 2016-08-19 1:2 stock_dividend under BF.B | 98.19 (07-29) | unadjusted | Massive lists the event but its adjusted aggregates do not apply it for this dotted share class. Vendor-side; the pack build had no independent split check | **CONFIRMED.** Factor 0.5 |
| BLL 2017-05-17 2:1 | BLL raw==adj: 05-16 80.81, 05-17 39.98; listed under BLL (stock_dividend 1:2); BALL has no 2017 bars | 76.89 (04-28) | unadjusted | Same vendor behaviour (event typed stock_dividend, retired symbol BLL→BALL) | **CONFIRMED.** Factor 0.5 |
| BF.B 2018-03-01 5:4 | BF.B raw==adj: 02-28 69.79, 03-01 55.37; listed (stock_dividend 4:5) | 69.79 | unadjusted | same as BF.B 2016 | **CONFIRMED.** Factor 0.8 |
| BKNG 25:1 (2026-04-06) on PCLN bars | PCLN raw==adj, 2018-01-31 1912.05; BKNG bars start 2018-02-27, adj 76.2004 = raw 1905.01/25; 2018-02-28 adj 81.3616 = 2034.04/25; split recorded under BKNG only | 1912.05 (01-31), 81.3616 (02-28) | Partially adjusted: BKNG-symbol bars yes, PCLN-symbol bars no | Split stored under the current ticker; bars under the old symbol PCLN are not adjusted | **CONFIRMED.** Factor 1/25 (1912.05/25 = 76.482) |
| IR/TT 2020-03-02 | IR 2020-02-28 129.04 (Ingersoll-Rand plc), 03-02 32.80; GDI last bar 02-28 32.79; TT bars start 03-02 at 100.55; no Massive split/spin record | IR 129.04 → 24.80; TT first 82.59 (03-31) | Issuer switch inside one ticker | Massive keeps the symbol IR across issuers and starts TT fresh. The public membership list keeps IR continuous from 2010 and starts TT on 2020-03-03 | **CONFIRMED** (see D) |

## B. Own rerank
`run_rerank.py` uses `queen_rerank.py`. Variants: pack; Claude's nine (Massive factors above, TT = old IR / 1.28784, IR removed and TT added for 2020-03-31..2021-02-26); the nine plus the EQT/ARNC fixes; EQT/ARNC only.

| Variant (1,572 ranked tables) | Top1 | Top2 | Top3 | Top5 | Top10 |
|---|---|---|---|---|---|
| Claude's nine | 0 | 0 | 0 | 0 | 3 |
| nine + EQT + ARNC | **3** | **5** | **5** | **8** | 21 |
| EQT + ARNC only | 3 | 5 | 5 | 8 | 18 |

Top3 changes, all ARNC (`rerank_changes.csv`):
- 2017-02-28 W4: pack ARNC|URI|CSX (ARNC +75.86%) → URI|CSX|NRG (ARNC +40.86%, rank 13)
- 2017-02-28 W5: ARNC|URI|CSX → URI|CSX|BAC (ARNC 66.03% → 32.99%, rank 28)
- 2017-02-28 W6: CSX|ARNC|NVDA → CSX|NVDA|WDC (ARNC rank 2 → 34)
- 2017-04-28 W6: ARNC|CSX|MU → CSX|MU|NRG (ARNC 66.94% → 33.72%, rank 26)
- 2017-04-28 W7: CSX|ARNC|MU → CSX|MU|NFLX (ARNC rank 2 → 57)
- Top5-only changes: 2017-03-31 W5, 2017-04-28 W8, 2017-05-31 W7.

Plausibility check: an old Alcoa holder's 3 shares, worth 3 × 28.72 on 2016-10-31, became 3 ARNC + 1 AA. The documented F applied to the raw start gives +40.9% to 2017-02-28. The pack's +75.9% comes from scaling the start price twice (0.801 × 0.7116).

## C. Errors Claude missed
Method: (1) I pulled the full Massive split/stock-dividend calendar from 2013-10-01 to 2026-11-12 (15,131 records, all US symbols; `massive_splits_all_2013_2026.csv`). I matched each record to each of the 725 rank-price engine tickers through their own symbol, the Massive aliases the pack used, and a hand-built successor map for renamed or retired names (TMK→GL, DISCA→WBD, BLL→BALL, PCLN→BKNG, FB→META, and about 60 more in `split_diff.py`). I then compared each record with the pack's month-end ratio (`split_calendar_diff.csv`, 159 matches; 21 look unapplied, classified in `split_diff_unapplied_classified.csv`). (2) I cross-checked each of the 84 pack spin/special-dividend events for a Massive vendor adjustment without a QADJ row. (3) I scanned every pack monthly move with |ln r| ≥ 0.15 inside a usable lookback (5,234 moves) against any Massive split with a matching factor in the same month (`pattern_scan_moves_15pct.csv`), and reviewed by hand the 380 moves with |ln r| > 0.33 that have no pack CA.

Findings (`new_errors.csv`):
1. **ARNC, prices on or before 2016-10-31: double spin adjustment.** Massive AA 2016-10-31 raw 28.72, adjusted 23.0047 = pack. Massive record AA 2016-11-01 801:1000. Pack CA AA_ARNC F=0.7116, no QADJ. 78 cells overstated, 5 Top3 tables changed. **Material.**
2. **EQT, prices on or before 2018-10-31: double spin adjustment.** Massive raw 33.97, adjusted 18.4921 = pack; record 1000:1837; F=0.5262; no QADJ. Latent: EQT is not eligible across the ex date.
3. HWM and BALL copy the ARNC and BLL series. Latent.
4. Every other Massive-recorded spin adjustment (A, DHR, EBAY, FTV×2, GE×2, HON 2025, HSIC, IBM, LH, MMM, XRX, YUM) has its QADJ row. UAA's class C dividend is handled once, as a spin. SLG's 2020–21 stock dividend and reverse split are in the Massive-adjusted pack values (verified raw vs adjusted).
5. For 16 spins with no Massive record, a Massive raw close equals the pack value on the month-end before ex, so there is no hidden vendor adjustment: BAX, HPQ, NI, PNR, DWDP, WYND(WYN), HPE, T, WDC, EXC, KDP(DPS), DXC(CSC), APTV(DLPH), BBWI(LB), CAG, DOV 2018 (`spin_vendor_spotcheck.csv`).
6. Apart from the known nine and ARNC, none of the 380 large unexplained moves lines up with a split. They are real events (March 2020, earnings crashes, FRC, NFLX 2022-04 and similar). No new split-calendar error was found.

## D. IR/TT
- Transaction: S&P DJI release (2020-02-27): Ingersoll-Rand plc stays in the S&P 500 as Trane Technologies (TT), and Gardner Denver, renamed Ingersoll Rand Inc. (new IR), joins prior to the open on Tuesday 2020-03-03. Trane 8-K Ex-99.1 and the Gardner Denver 8-K (CIK 1699150): spin on 2020-02-29, 0.8824 GDI shares per IR plc share, record date 2020-02-24.
- Price series: Massive IR through 2020-02-28 is IR plc (129.04); from 2020-03-02 it is Gardner Denver (32.80; GDI 02-28 32.79). TT starts 2020-03-02.
- Ratio: (TT 100.55 + 0.8824 × new IR 32.80) / 100.55 = **1.2878**. Claude's 1.289 (FMP 1289:1000) agrees within 0.1%. Correct.
- Eligibility: new IR joined on 2020-03-03, so under the 365-day rule it is first eligible on 2021-03-31 (2021-02-26 − 365 days = 2020-02-27, before it joined). Removing IR for 2020-03-31..2021-02-26 (12 signals) and adding TT, the continuing constituent, is **correct, neither too aggressive nor too loose**. Wikipedia also lists IR's date added as 2020-03-03.
- Top3 effect: none. TT's best rank is 7 (2020-07-31 W1, 4.75pp short); new IR's removal changes no Top10.

## E. New versioned pack (not built; see `NEW_PACK_NOTE.md`)

## Not checked
- I did not rerun the 5,184-path grid or the 324 stop-loss paths with the ARNC fix. The 364-path count is an exposure count from the answer key, not a count of changed results.
- Execution-side prices (FMP holding series) were not audited beyond the 6,801-pair return cross-check.
- The successor map for renamed tickers is hand-built. A rename I did not map, whose split Massive files only under the new symbol, would be missed by the calendar diff. The move scan (|ln r| ≥ 0.33 reviewed by hand) is the backstop.
- Small (<~5%) stock dividends: the classifier is ambiguous. I checked these only through the "pack is Massive-adjusted, not raw" heuristic, verified directly for SLG.
- 16 of about 70 no-record spins were spot-checked with raw closes. The rest were inferred from Massive's calendar having no record.
- FOXA (earlier Queen finding), Taiwan, membership correctness beyond the public list, and dividends (price return only, by spec) were not checked.
- Wikipedia check used an older local snapshot, not today's page.
