# Independent GPT review of Claude's US input audit

Review date: 2026-11-12  
Frozen data/code ref reviewed: `54a6b13101c982d124d9bd4358fbdfdd49f39bc7`; Queen's independent report was later committed at `30812438d7f5b22d284b25b81453556998fe2faf`.  
Scope: US frozen pack only. No edits were made to `main`, the frozen input pack, or Queen's answer key.

## Verdict

| Claim | Verdict | Finding |
|---|---|---|
| 1. Nine listed price/identity corrections | **CANNOT VERIFY** as an exact nine-factor specification; the underlying split/dividend and issuer events are broadly supported | Independent primary-source documents and Massive daily prices support the broad events and the need to repair the histories. Exact vendor factors for GOOGL/GOOG, DISCA and IR/TT could not all be independently established. |
| 2. 674 lookbacks affected, all understated | **CONFIRMED with a caveat** | GPT's separately written ranker finds the same 674 changed cells for the nine fixes. Price-scale changes understate returns, but 288 IR/TT rows reflect eligibility remapping, not understated prices. |
| 3. Top1–3/Top5 unchanged; three Top10 changes | **CONFIRMED** | Across 1,584 `(signal, W)` tables, Top3 changes: 0; Top5: 0; Top10: 3. |
| 4. 5,184 grid and 324 stop-loss results unchanged | **CONFIRMED for Claude's nine corrections only; REFUTED for all known corrections** | Adding ARNC's double-adjustment correction changes Top1 in 3 tables, Top2/Top3 in 5 tables, and picks in 364 answer-key paths that rebalance on those tables. GPT reran the grid, but the frozen execution-availability pack produces 111 `NOT_IN_PACK` path failures for newly selected names; 86 paths that were known under the nine-fix run become UNKNOWN. Therefore the corrected full-grid outcome is not yet validated. The 324 W10–12 × H2–4 stop-loss paths remain unchanged. |
| 5. Supporting input checks | **CANNOT VERIFY** as a complete claim | GPT reproduced the 6,801 Massive/FMP return comparison and the 9/132 intervals-vs-eligibility mismatch. The fja05680 membership comparison and Wikipedia join-date sample were not independently reconstructable from preserved source snapshots. The Massive event calendar also proved incomplete against official filings. |

## 1. The nine corrections

The following table separates confirmation of an event from confirmation of Claude's exact scaling factor. “Massive ratio” is close-to-close raw daily close ratio around the event; corporate-action ex-dates can make this differ slightly from a month-end correction factor.

| Engine ticker/event | Independent evidence | Finding |
|---|---|---|
| GOOG/GOOGL, Apr 2014 | Massive GOOG raw close was $1,135.10 on Apr 2 and $569.74 on Apr 3 (0.5019). SEC/Alphabet materials document the Class C stock distribution. The Massive adjusted close still shows the near-halving. | Broad 2:1 scale error supported. Exact 999:500 factor not independently established. [SEC/Alphabet](https://www.sec.gov/Archives/edgar/data/1652044/000119312514125769/d699468dex991.htm) |
| TMK, Jul 2014 | Massive close $82.39 on Jul 1 and $54.83 on Jul 2 (0.6657); Torchmark disclosed a 50% stock dividend / 3-for-2 event. | 3-for-2 correction supported. [Company announcement](https://investors.globelifeinsurance.com/news-releases/news-release-details/torchmark-announces-50-stock-dividend) |
| DISCA, Aug 2014 | Massive close $83.02 on Aug 6 and $40.975 on Aug 7 (0.4936); Discovery announced one Class C share per existing share, and Nasdaq records the ex-date. | Broad 2:1 correction supported. Exact 1957:1000 factor not independently established. [SEC filing](https://www.sec.gov/Archives/edgar/data/1437107/000119312514263045/d758612dex991.htm) |
| BF.B, Aug 2016 | Massive close $99.65 on Aug 18 and $50.40 on Aug 19 (0.506); Brown-Forman's filing confirms 2-for-1. | Supported. [SEC filing](https://www.sec.gov/Archives/edgar/data/14693/000001469316000109/bf-b-20160819x8k.htm) |
| BLL, May 2017 | Massive close $80.81 on May 16 and $39.98 on May 17 (0.495); Ball announced a 2-for-1 stock split. | Supported. [Company announcement](https://www.ball.com/newswire/article/118193/ball-corporation-announces-two-for-one-stock-split) |
| BF.B, Mar 2018 | Massive close $69.87 on Feb 28 and $55.37 on Mar 1 (0.7925); Brown-Forman disclosed one additional B share for each four existing shares. | 5-for-4 correction supported. [SEC filing](https://www.sec.gov/Archives/edgar/data/14693/000001469318000018/bf-b-20180301x8k.htm) |
| BKNG/PCLN, Feb 2018 | SEC confirms BKNG's 25-for-1 split effective Apr 6, 2026. Massive's PCLN-era close on Jan 31, 2018 is $1,912.05; divided by 25 it is $76.482, in the later BKNG price scale. | 25-for-1 scale discontinuity supported. [SEC filing](https://www.sec.gov/Archives/edgar/data/1075531/000107553126000018/bkng-20260223.htm) |
| IR/TT, Mar 2020 | SEC documents the separation: legacy Ingersoll-Rand plc became Trane Technologies (TT); Gardner Denver took the IR ticker and began trading Mar 2, 2020. Massive has old IR at $129.04 on Feb 28, TT at $100.55 on Mar 2, and new IR at $32.80. | Issuer-identity break confirmed. The 1/1.289 bridge is plausible from the price scale but is not an officially defined corporate-action conversion factor. [SEC filing](https://www.sec.gov/Archives/edgar/data/1699150/000119312520060271/d864688dex991.htm) |

The adjustment factor of each record is implemented as stated in the Claude audit for the nine-fix replay. This verifies ranking sensitivity to those assumed factors; it does not independently establish every exact factor from primary-source documents.

## 2–4. Independent ranking and execution reruns

`independent_input_audit.py` is a standalone ranker written from the frozen specification and input pack; it does not import Claude's implementation. It recomputes eligibility, lookback returns, ranking, and applies the eight price-scale adjustments plus the IR/TT identity remap.

```text
tables 1584
Top3 changes 0
Top5 changes 0
Top10 changes 3
affected cells 674
base/corrected UNKNOWN rank tables 12 / 12
```

The three Top10 changes are:

| Signal | W | Name newly in corrected Top10 | Corrected rank |
|---|---:|---|---:|
| 2018-04-30 | 4 | BKNG | 9 |
| 2020-07-31 | 1 | TT | 7 |
| 2020-10-30 | 4 | TT | 8 |

The nearest corrected Top3 miss is TT on 2020-07-31 W1, 4.8 percentage points behind the third-place return.

The nine-fix rankings were passed to GPT's independent Python executor and fixed-30% stop implementation. The executor compared each path to Queen's answer key:

```text
total=5184 known=4718 UNKNOWN=466; verification discrepancies=0
stop paths=324 known=324 UNKNOWN=0
```

For 4,718 computable paths, maximum absolute differences versus Queen were `1.36e-13` for CAGR, `3.69e-13` for daily max drawdown, and `1.26e-10` for final multiple. Claude's pack-input and corrected-input path files are byte-identical, as are its paired stop-loss files. These comparisons apply only to the nine fixes. After also correcting ARNC, GPT's independent ranker found 5 Top3 changes (same count as Queen), 8 Top5 changes, and 21 Top10 changes. The changed tables are 2017-02-28 W4/W5/W6 and 2017-04-28 W6/W7; the altered Top3 selections match Queen's published rerank. Matching those tables to Queen's rebalance ledger yields 364 unique paths that change picks.

GPT's executor ran on the additional corrected rankings and emitted 4,632 KNOWN and 552 UNKNOWN. Compared with the nine-fix output, 363 path final multiples differ and 86 previously KNOWN paths become UNKNOWN. The verifier reports 787 field discrepancies against Queen's frozen answer key, which is expected because that key contains the old picks. In the all-fixes grid the UNKNOWN causes are 111 `NOT_IN_PACK`, 393 `pick_not_tradable`, and 48 `corporate_action_unresolved`; the frozen execution-availability table does not contain every needed ticker/date/window row for newly selected names. The daily close table may contain prices, but availability cannot be inferred safely from that alone. Thus this is a rerun and exposure diagnosis, not a verified corrected outcome grid.

### Untradable-path rebalance-count presentation

The 417 `pick_not_tradable` UNKNOWN paths have different rebalance-count displays. Queen records zero executed rebalances on all 417, because the path is invalidated after detecting an untradable selected name. GPT records 3–94 attempted rebalances before invalidating the whole path. The paths have the same UNKNOWN status and fail reason; the difference is whether the counter means “completed/executed after path validity” or “attempts made before the later failure.” It is a reporting-semantics difference, not evidence that one path completed those rebalances. `corporate_action_unresolved` accounts for the other 49 UNKNOWN paths.

## Additional issues found outside the nine corrections

| Ticker(s) | Date | Finding | Limit |
|---|---|---|---|
| ARNC; HWM alias | 2016-11-01 | **Confirmed additional double adjustment.** The pack's ARNC rank-price override on 2016-10-31 is 23.0047, matching Massive AA adjusted close (raw 28.72 × 0.801). The pack also applies spin factor F=0.7116349 to the lookback start. For a consistent basis, use the raw pre-spin price before applying that spin factor. GPT independently rescales the pre-event ARNC rank prices by 1/0.801; this lowers ARNC's 2017-02-28 W4 return from 75.86% to 40.86% and reproduces all five changed Top3 tables in Queen's rerank. [SEC spin terms](https://www.sec.gov/Archives/edgar/data/4281/000119312516731663/d249430dex991.htm) | 78 affected ARNC lookback cells; correction changes five ranking tables. HWM's copied prices are latent in our replay because its eligibility/action treatment blocks those lookbacks. |
| EQT | 2018-11-13 | Queen's split-calendar review identifies the same potential vendor+pack double adjustment: pack pre-ex price 18.4921 equals Massive raw 33.97 × historical factor 0.54437, and the pack also applies its own spin factor. | EQT was removed from the index on the ex-date; GPT's own corrected ranking has zero changed EQT cells, so it is latent for this universe. [SEC spin terms](https://www.sec.gov/Archives/edgar/data/33213/000104746918007182/a2237122zex-99_3.htm) |

My initial AIV and SLG “missing action row” leads are not counted as newly confirmed errors: absence of a pack action row alone does not prove a price-history defect, and vendor-adjusted prices may already incorporate the event. Queen's report specifically verifies the SLG stock distribution/reverse split in the Massive-adjusted pack values. FOX/FOXA has a real issuer-identity transition, but I did not establish a bad rank-price bridge or rerun its affected rankings, so I do not classify it as a confirmed input error here.

Massive's split endpoint was queried for 725 current engine tickers and historical aliases, yielding 173 event records over the reviewed date span; 145 event/ticker pairs could be aligned to pack prices. It is not a complete stock-dividend calendar: Massive omits Google's 2014 event despite primary evidence and a visible daily price discontinuity. Queen's broader scan later found the ARNC double-adjustment, which GPT's smaller split-ratio screen had missed. Many flagged spin-like price ratios are already represented by pack spin/action overrides. Accordingly, I cannot certify the full corporate-action list for all 725 histories.

## Supporting checks and scope limits

- Recomputed comparison: 6,801 matched Massive rank-price/FMP execution-close monthly returns; 0 exceeded 0.5% disagreement, maximum relative difference 0.40266%. This checks price-return consistency for matched observations, not the correctness of either vendor's corporate-action history.
- `membership_intervals.csv` and `eligibility_by_signal.csv.gz` disagree on 9 of 132 signal dates (123 agree), consistent with the reported interval-vs-eligibility discrepancy.
- I did not independently reproduce the `fja05680/sp500` membership check because the exact source snapshot was not present in the review inputs. I also did not reconstruct the 173 Wikipedia join-date sample from a frozen snapshot. Those two Claude subclaims remain unchecked.
- No adjusted/raw event ratio can by itself prove the vendor's historical basis: Massive adjusted bars also preserve the near-half steps for the Google event. SEC/company documents were used to establish event existence; independent daily data was used to inspect the observed price scale.
- These are arithmetic/data-quality checks of supplied histories, not evidence that the input universe or vendor prices equal authoritative historical index membership and executable market prices.

## Reproduce locally

Run from the repository root with Python 3.10+ and only the standard library. First verify the nine-fix replay, then run the expanded ranking/exposure replay:

```bash
python research-exchange/gpt/input-audit-review-20261112/independent_input_audit.py
cp research-exchange/gpt/input-audit-review-20261112/corrected_rankings_1584.csv research-exchange/gpt/input-audit-review-20261112/independent_rankings_1584.csv
python research-exchange/gpt/input-audit-review-20261112/independent_execution_corrected.py --verify
python research-exchange/gpt/input-audit-review-20261112/independent_input_audit_all_fixes.py
cp research-exchange/gpt/input-audit-review-20261112/corrected_rankings_all_fixes_1584.csv research-exchange/gpt/input-audit-review-20261112/independent_rankings_1584.csv
python research-exchange/gpt/input-audit-review-20261112/independent_execution_corrected.py
python research-exchange/gpt/input-audit-review-20261112/stoploss_corrected.py
```

Expected: the first executor run yields 4,718 known / 466 UNKNOWN and zero discrepancies. The expanded ranking changes five Top3 tables; its executor run is an exposure replay against the old answer key, not a verified results comparison.

The input pack manifest verifies 19/19 files; Claude's audit manifest verifies 20/20 files when paths are resolved relative to its audit directory. The scripts read the frozen shared pack and answer key and write outputs only to this GPT review directory.

## Artifacts

- `independent_input_audit.py`: independent ranking replay for Claude's nine fixes.
- `independent_input_audit_all_fixes.py`: reranks with those nine plus ARNC/HWM/EQT vendor-scale checks.
- `independent_execution_corrected.py`, `independent_execution.py`: executor; `independent_rankings_1584.csv` is set to the all-fixes rankings for the latest grid output.
- `stoploss_corrected.py`: 30% fixed-stop rerun for W10–12 × H2–4 × Top1–3 × 12 entry phases.
- `corrected_rankings_1584.csv`, `rank_comparison_1584.csv`, `affected_rank_rows_independent.csv`: nine fixes.
- `corrected_rankings_all_fixes_1584.csv`, `rank_comparison_all_fixes_1584.csv`, `affected_rank_rows_all_fixes.csv`, `execution_delta_nine_vs_all_fixes.csv`: full correction candidates.
- `independent_full_grid_nine_fixes_5184.csv`, `independent_full_grid_5184.csv`, `stoploss_w10_12_h2_4_30pct_paths.csv`, `stoploss_w10_12_h2_4_30pct_summary.csv`.
- `massive_splits_2013_2025.csv`, `massive_event_pack_ratio_screen.csv`, `market_data_spots.csv`, `additional_action_candidates.csv`, `review_support_checks.csv`.

On GitHub, CSV artifacts larger than 60 KB are also stored as `.csv.gz` mirrors to keep the research commit compact. `GITHUB_SHA256SUMS.txt` verifies the exact subset and compressed files committed there; `SHA256SUMS.txt` verifies the full local artifact set.
