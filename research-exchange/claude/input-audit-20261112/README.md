# Claude input-data audit of the frozen US pack (2026-11-12, Asia/Taipei)

Research only. Reads `shared/inputs-pack-full-20261109`; does not modify it, the answer key, `main`, or any other agent's folder. Checked at branch commit `04932e0`.

## Verdict

The frozen pack contains **eight price-scale errors on seven tickers plus one issuer-identity error**. Each is confirmed by an FMP split record. After correcting all of them and re-ranking every signal table, **no Top1, Top2 or Top3 pick changes in any of the 1,584 (signal, W) tables**, so none of the 5,184 path results change. The headline numbers stand; the pack itself should get a versioned fix.

## Errors found (all make the affected name look worse than it was, never better)

| Engine ticker | Month where the pack shows a fake drop | Cause | Signals affected |
|---|---|---|---|
| GOOGL, GOOG | 2014-04 (−52%) | class C share dividend 2014-04-03 (999:500) not applied | 2014-12 to 2015-03 |
| TMK | 2014-07 (−36%) | 3-for-2 split 2014-07-02 not applied | 2014-12 to 2015-06 |
| DISCA | 2014-08 (−49%) | class C share dividend 2014-08-07 (1957:1000) not applied | 2014-12 to 2015-07 |
| BF.B | 2016-08 (−51%) | 2-for-1 split 2016-08-19 not applied | 2016-08 to 2017-07 |
| BLL | 2017-05 (−47%) | 2-for-1 split 2017-05-17 not applied | 2017-05 to 2018-04 |
| BF.B | 2018-03 (−22%) | 5-for-4 split 2018-03-01 not applied | 2018-03 to 2019-02 |
| BKNG | 2018-02 (−96%) | 25-for-1 split of 2026-04-06 applied to BKNG-era prices but not to PCLN-era prices | 2018-02 to 2019-01 |
| IR / TT | 2020-03 (−81%) | engine ticker IR switches issuer on 2020-03-02 (Ingersoll-Rand plc, today's TT, becomes Gardner Denver). New IR is treated as seasoned; TT is treated as a new member until 2021-03 | 2020-03 to 2021-02 |

Common pattern: the split belongs to a ticker that was later renamed, retired or is a dotted share class (TMK→GL, BLL→BALL, PCLN→BKNG, DISCA→WBD, BF.B), which is where the vendor's adjustment seems to be missing. The IR case is the same kind of issue Queen already found for FOXA.

## Effect of correcting them

- 674 (signal, W, ticker) lookback returns were wrong (`affected_rank_rows.csv`).
- Top3 changed: **0 of 1,584 tables**. Top10 changed in 3 tables (`top3_pack_vs_corrected_1584.csv`).
- Closest calls after correction: TT reaches rank 7, 4.8pp short of Top3; BKNG rank 9, 10.5pp short. Everything else is rank 21 or lower.

## Other checks

- **Two vendors inside the pack:** 6,801 month-to-month returns where both a Massive rank price and an FMP execution close exist for the same name. Zero disagree by more than 0.5%. (Price *levels* differ for GE, WDC, ILMN and others by a constant spin factor; returns agree.) This only covers names that were actually held.
- **Membership vs its source:** the pack's `member_at_signal` equals the public list `fja05680/sp500` (commit `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7`) on all 132 signal dates. That list is Andreas Clenow's file for 1996–2019 and Wikipedia after 2019, so this shows the pack copied its source faithfully, not that the source is right.
- **Join dates vs Wikipedia "date added":** of 173 current members added 2013-06 to 2024-12, 149 match the pack's interval start exactly and 160 within 10 days. The 13 outliers are renames (META/FB, EG/RE, WTW/WLTW, CPAY/FLT, HWM/ARNC, BKR/BHGE) or restructurings where Wikipedia restarts the date (D, TROW, SRE, IR, FOXA, FOX). For post-2019 additions this is circular with the source.
- **A second public list (`hanshof/sp500_constituents`) is not usable as a reference:** it has 464 names at 2014-12-31 against 499, because delisted names are missing.
- **Intervals vs eligibility:** 9 of 132 dates differ, as GPT and Queen already reported.
- **Outside price spots:** `fmp_price_spots_fetched.csv` holds the 24 FMP values pulled by hand (annual-champion path, SPY, BKNG). The champion path rebuilds to 113.7806x.

## Backtests rerun on the corrected data (added 18:20, Asia/Taipei)

Question asked: do the backtests change once the nine corrections are applied? **No.**

- `results/paths_5184_pack_inputs.csv` and `results/paths_5184_corrected_inputs.csv` — all 5,184 paths from Claude's executor, on the frozen pack and on the corrected rankings, each row next to Queen's answer. The two files are byte-identical: 4,718 known, 466 UNKNOWN, same fail reasons; max difference vs Queen 1.0e-12 (final), 1.4e-13 (CAGR), 3.7e-13 (drawdown).
- `results/stoploss_324_pack_inputs.csv` and `results/stoploss_324_corrected_inputs.csv` — the 324 W10-12 x H2-4 paths with no stop, GPT's 30% fixed stop, and a 30%-from-peak trailing stop. Also byte-identical between the two inputs; the fixed-stop columns match GPT's file to 3e-15.
- Reason: Top5 is unchanged in all 1,584 tables, the backtests only buy Top1-3, and none of the nine names was ever held, so execution prices are untouched.
- Where it would matter: Top10 changes in 3 tables (BKNG rank 483 to 9 at 2018-04-30 W4; TT enters at rank 7 at 2020-07-31 W1 and rank 8 at 2020-10-30 W4), and any rule that uses the bottom of the ranking.

To rerun on corrected data: run `audit_inputs.py` (writes `rank_corrected.pkl`), then set `RANK_PKL=rank_corrected.pkl` and optionally `OUT_CSV=<file>` before the executor or stop script.

## What this audit does NOT cover

- The scan only catches moves of about −20% or worse in one month. A small unadjusted split (for example 10%) or a stock dividend would not stand out.
- No full comparison against a complete split calendar: the FMP plan hit its rate limit after the calls recorded here. **Suggested next step for another model:** pull `splits-company` for all 725 engine tickers (using today's symbol for renamed ones) and diff against the month-end price ratios.
- Membership between year-ends still rests on the public list. No official S&P file or fund-holdings file was available here.
- FMP and Massive may share an upstream price tape.
- Taiwan is out of scope.

## Files

- `audit_inputs.py` — reruns everything above from the frozen pack. Standard library only. Writes the three CSVs and `audit_stdout.txt`.
- `price_corrections.csv` — the nine corrections with evidence. `fmp_splits_fetched.csv` — the raw FMP split records behind them.
- `scan_large_monthly_moves.csv` — all 932 large monthly moves that sit inside a usable lookback, for others to review. Most are real (March 2020, earnings crashes).
- `independent_ranker_claude.py`, `independent_execution_claude.py`, `stoploss_claude.py` — Claude's own ranker, executor and 30% stop, written from `SPEC.md`. Their run logs are `run_execution_stdout.txt` (5,184 paths: 4,718 known, 466 UNKNOWN, same reasons, CAGR diff 1.4e-13 vs Queen) and `run_stoploss_stdout.txt` (324 paths match GPT to 3e-15). The "TRAIL" columns in the stop log are a 30%-from-peak variant with no second implementation behind it.

## Rerun

```bash
cd research-exchange/claude/input-audit-20261112
python3 audit_inputs.py
python3 independent_ranker_claude.py ../../shared/inputs-pack-full-20261109          # writes rank3.pkl (not committed)
python3 independent_execution_claude.py ../../shared/inputs-pack-full-20261109 ../../queen/answer-key-full-20261109
python3 stoploss_claude.py ../../shared/inputs-pack-full-20261109 ../../gpt/stoploss_w10_12_h2_4_30pct_paths.csv
```

## Questions for Queen and GPT

1. Please confirm the nine corrections independently (Massive raw vs adjusted closes for the months listed).
2. Is there a reason the pack's quality gates did not flag a −96% month for BKNG or −81% for IR?
3. Please publish a versioned pack (do not overwrite 2026-11-09) and rerun the grid to confirm zero path changes.
