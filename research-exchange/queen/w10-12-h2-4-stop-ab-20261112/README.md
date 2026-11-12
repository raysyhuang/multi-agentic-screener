# W10–12 × H2–4 × K1: research overlay of a 30% trailing stop (variants A / B)

PROVISIONAL / research only. Capital 0. Built 2026-11-12 19:00 (Asia/Taipei). Not pushed to GitHub.
The official v18 no-stop numbers (W10H2 45.24%, W12H4 43.03%, annual champion 60.5383%) are unchanged. The official
5,184-path grid does **not** include this stop.

## Inputs (read only; MANIFEST.sha256 re-checked after the run: all OK)
- `/workspace/momentum432-answer-key-full-20261109/` (paths_summary, rebalances, nav_daily) — for baseline comparison.
- `/workspace/momentum432-inputs-pack-full-20261109/` (execution_daily_closes, used for independent spot checks).
- Frozen v17-strict4 engine (`/workspace/second-sleeve-v17-targeted-ca-20261108/engine`), loaded read-only with the same
  `load_v17.py` the answer-key build used, plus the same in-memory v18 KORS/CPRI dedupe wrapper. It is used **only as a data
  oracle**: rank tables (`rank_strict`), execution price series lookup (`series_strict`), held corporate actions
  (`held_ca`, `verify_post`) and deal dates (`_dealdate`). The simulation loop and stop logic are my own (`overlay.py`).
  Reason: a replacement name bought mid-window needs the engine's own FMP holding series for a window the pack never stored;
  `series_strict(t, trigger_date, next_rebalance)` is exactly how the engine would choose that series. No Massive total-return mixing.
- ARNC/EQT double adjustment and Claude's nine rank-price corrections do not touch any W9–W12 table (Top1/Top2 unchanged), so
  the v18 pack / answer key are used as-is.

## Step 1 — baseline reproduction (gate)
`validate.py` → `baseline_reproduction_check.csv`. With the stop switched off, the loop reproduces all 108 band paths plus
W12H12K1P1: same status, identical rebalance picks and dates on every rebalance, final multiple rel. diff ≤ 5.7e-13, CAGR ≤ 9.1e-14,
MDD ≤ 1.0e-13, daily NAV vs `nav_daily` ≤ 5.2e-12 (file stores 12 significant digits). 109/109 match, so stop results are
published for every path.

## Stop rule as implemented (locked 2026-11-12)
- Peak = highest daily close of the current position since its entry close (entry day included). Not from cost; can fire
  while still in profit.
- Trigger: close ≤ 70% × peak, checked every session that is not a scheduled rebalance day and not END.
- Price series: the engine's FMP execution/holding series for that position (same series as the official NAV).
- Primary execution: same triggering close. Exit pays 25 bp, replacement entry pays 25 bp, using the engine's K=1 fee
  fixed point → equity × (1−c)/(1+c) = 0.9950125 (checked on all 1,797 stop events). Cash case: equity × (1−c).
- Replacement source: latest month-end signal date **on or before** the trigger date (if the trigger falls on a month-end,
  that same day's table is used; footnote: ranks use that day's close, which is also the trade close).
  - A: W12 Top1 of that table (annual-half rule, evaluated on that date).
  - B: Top1 of the path's own W at that table.
  - If Top1 is the name just stopped, take Top2. Only the name just stopped is excluded.
  - Replacement is ineligible → cash until the next scheduled rebalance if: table UNKNOWN (missing ranks), no Top2, the
    replacement has a deal date in (signal, trigger], no verified engine series, or engine says not tradable.
    No deeper substitution (Top3+). **This never happened (0 cash events in 1,797 stop events).**
- The replacement is a new position: peak resets to its entry close. It can itself be stopped again.
- Stop is an interrupt only: at the next scheduled rebalance the path returns to its own W/H Top1 with normal costs. If the
  scheduled pick equals the name already held, no trade (turnover 0), and the position — and its peak — continue (the
  position was not reopened). Same for consecutive scheduled picks of the same name in the no-stop engine. Example:
  W12H12P1 held AMD from 2019-01-02 through the 2020-01-02 rebalance, peak 2020-02-19, stopped 2020-03-12.
- Corporate actions on a held position follow the engine rules. Peak is re-expressed on those days so a distribution does
  not fake a drawdown: spin/special dividend → peak × P/(P + net per-share value received); stock deal → peak converted to
  acquirer units by value. Only LB_VSCO spin (2021-08-03), LB→BBWI and CVC cash deal touched held positions; no stop fired on
  any such day.
- UNKNOWN propagation: any failure in a stop variant makes the whole path UNKNOWN (none occurred).
- CAGR years = calendar days / 365.25 over the official window (first session of entry month 2015 → first session of the
  same month 2025). Daily MDD on daily NAV.
- Sensitivity (`*_next`): same rules but stop trades at the **next** session close (decision still from the trigger close,
  replacement table still the latest ME ≤ trigger date). If the next session is a scheduled rebalance / END, the pending stop
  is dropped (never happened).

## Files
| file | content |
|---|---|
| `REPORT.md` | results and 人话 headline |
| `per_path_W10-12_H2-4_K1.csv` | 108 rows: official vs reproduced no-stop vs A vs B (+ next-session sensitivity): CAGR, daily MDD, final multiple, n_stops, n_cash_stops, n_scheduled_rebalances, status, ΔCAGR/ΔMDD in pp |
| `side_annual_W12H12K1P1.csv` | annual champion side comparison (not a replacement of 60.5383%) |
| `per_path_long.csv` | long format incl. first stop date, days to first stop |
| `stop_events.csv` | every stop: trigger date/close, peak date/close, drawdown, signal table used, Top1/Top2, replacement, series, equity before/after |
| `summary_by_group.csv` | band, per-cell (12 phases) and phase-1 summaries |
| `combined_5050_phase1_books.csv` | optional 50/50 books (phase 1), separate sleeves, no cross-sleeve rebalancing |
| `nav_daily_anchors_and_annual.csv.gz` | daily NAV for W10H2K1P1, W12H4K1P1, W12H12K1P1 under every variant |
| `baseline_reproduction_check.csv` | step-1 gate |
| `spot_checks_raw.txt` | hand checks against pack daily closes |
| `overlay.py`, `validate.py`, `analyze.py`, `extra.py` | code; rerun: `python3 overlay.py && python3 validate.py && python3 analyze.py && python3 extra.py` (~15 s) |

## What was NOT checked / limits
- Look-ahead is avoided by construction (only tables dated ≤ trigger), but intraday stops, gaps through the stop and
  slippage beyond 25 bp are not modelled; daily-close check only.
- No re-verification of the engine's series choice for replacement windows beyond the engine's own ≤0.5% month-end
  identity check; 894/900 A/B replacement entry closes exist in the pack under the same series name; the 6 others are
  `fmp:BBWI+vendor_deadj` (pack stores that series under a `#hash` name).
- Prices are split-adjusted, no dividends (same as v18). Membership still PROVISIONAL (public reconstruction).
- Peak continuation across no-trade rebalances is a convention; a "reset peak at every scheduled rebalance" variant was
  not run.
- No significance testing; the stop was chosen after seeing this history, so nothing here is out-of-sample.
- Top3+ substitution, partial exits, re-entry filters were not tested. Only K=1.
- Combined books: only phase 1, buy-and-hold 50/50 sleeves, no periodic re-balancing between sleeves.
