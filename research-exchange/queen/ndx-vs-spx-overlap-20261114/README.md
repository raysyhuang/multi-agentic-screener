# NDX100 post rule vs S&P 500 momentum — overlap and performance (2026-11-14, Asia/Taipei). Research only, capital 0. ALL PROVISIONAL.

Engines (both imported read-only, nothing changed in either folder):
- NDX: /workspace/ndx100-fill-20261114-v1.1 (post rule: 6m MEAN of monthly returns, Top3 EW, month-end close -> month-end close, 25 bps/side, price only).
- S&P 500: Queen v17 strict4 + v18 KORS/CPRI dedupe via /workspace/no-nvda-grid-20261114/build/load_v17.py (signal at month-end,
  buy at next first-session close, 365-day seasoned PIT universe, strict deal/CA handling). W6H1K3P1 reproduces the answer key
  exactly (x1.7317, CAGR 5.6440%, MDD -54.42%, status PROVISIONAL_C).
- "S&P same signal" = NDX post-rule signal (mean of last 6 one-month rank-price returns, spin/special-div adjusted inside each month)
  plugged into the S&P engine (H1, K3, phase 1). Execution timing stays the S&P engine's (next first-session close), not month-end close.

## 1. Membership overlap (month-ends 2015-01..2024-12, each engine's own PIT membership)
NDX members also in S&P 500: mean 80.5% (min 75.2%, max 84.5%; 77.1% at 2015-01, 84.2% at 2024-12). In the S&P engine's
seasoned (eligible) universe: mean 77.3%. Ticker aliases used: FB=META, PCLN=BKNG, YHOO=AABA.

## 2. Pick overlap (NDX Top3 monthly picks)
| window | NDX picks in S&P500 | in S&P seasoned universe | also in S&P same-signal Top3 | also in S&P W6H1K3P1 | in S&P same-signal Top10 | months 3/3 same | months 0/3 |
|---|---|---|---|---|---|---|---|
| 2015-2024 (120 m) | 68.1% | 60.8% | 31.7% | 32.8% | 50.8% | 1 | 45 |
| 2025-01..2026-09 (22 m) | 77.3% | 50.0% | 30.3% | 28.8% | 47.0% | 2 | 11 |
Biggest non-S&P NDX contributors (simple sum of 1/3 x stock month return while not in S&P): TSLA +79 pp (16 m, pre-Dec-2020),
ZM +40, MRNA +39, CRWD +19, MELI +19 (2015-2024); APP +27, MSTR +11 (2025-26). File: non_sp500_ndx_picks_contribution.csv.

## 3. Performance (25 bps/side; QQQ/SPY price only)
| series | 2015-01..2024-12 | 2015-01..2026-09 (S&P part UNAUDITED extension) |
|---|---|---|
| NDX post rule Top3 | x11.50, 27.66%, MDD -49.7% | x69.89, 43.54%, MDD -49.7% |
| NDX post rule Top3, universe NDX∩S&P500 | x8.69, 24.14%, MDD -38.4% | x38.57, 36.46%, MDD -45.2% |
| S&P same signal Top3 (LENIENT, see caveat) | x2.45, 9.36% (engine), MDD -43.6% | x7.13, 18.20%, MDD -43.6% |
| S&P W6H1K3P1 (6m total return) | x1.73, 5.64% (engine), MDD -54.4% | x3.44, 11.10% (LENIENT), MDD -54.4% |
| S&P same signal, no seasoning (sensitivity) | x1.61, 4.87% (engine), MDD -59.4% | UNKNOWN (exec_price_missing BNY 2026-06-01) |
| QQQ price | x4.95, 17.34%, MDD -35.6% | x7.16, 18.25%, MDD -35.6% |
| SPY price | x2.85, 11.05%, MDD -34.1% | x3.71, 11.81%, MDD -34.1% |
Monthly return correlation NDX Top3 vs S&P same-signal Top3: 0.44 (2015-2024, 120 m), 0.56 (to 2026-08 signals, 141 m);
vs S&P W6H1K3P1 0.42 / 0.55; NDX Top3 vs QQQ 0.61. Months aligned by signal month-end (S&P month is 1 session later).

## Caveats
- STRICT S&P same-signal path is UNKNOWN: pick_not_tradable ANDV (rank 2 at signal 2018-09-28; MPC deal effective 2018-10-01).
  LENIENT = drop names whose deal is effective by the trade date and take the next rank (only ANDV changed a pick in 2015-2024).
- Extension past 2025-01-02 (to 2026-10-01): MEL extended in memory; corporate actions/membership after 2025 NOT audited;
  strict path fails at DD (Qnity spin, rank_missing 2025-11..2026-04); LENIENT drops DD from ranking. S&P membership file ends 2026-08-18.
- Timing mismatch: NDX month-end close vs S&P next first-session close. Signal mismatch for W6H1K3P1 (total vs mean).
- S&P engine has a 365-day seasoning filter the post rule lacks; no-seasoning sensitivity drops names lacking 6m rank prices (246 drops, many spin-offs/renames) -> approximate.
- NDX CAGR is month-count based; S&P "engine" CAGR is the engine's day-count; the CSV recomputes S&P CAGR with calendar days from engine wealth (tiny differences vs engine CAGR).
- NDX coverage gaps as in v1.1 (99.99% 2015-2024). ndx.pkl/spx.pkl in build/ are scratch dumps.
- Box incident: the first spx_run.py run wrote spx.pkl into the v17 engine folder (load() chdirs there); it was moved out within
  minutes, no engine file was changed (only that directory's mtime). Script now writes with an absolute path.

Files: membership_overlap_by_month.csv, pick_overlap_by_month.csv, pick_overlap_summary.csv, non_sp500_ndx_picks_contribution.csv,
performance_side_by_side.csv, monthly_returns_aligned.csv, monthly_correlation.csv, sp500_engine_paths.csv, sp500_rebalance_picks.csv,
sp500_lenient_drops.csv, spx_engine_stdout.txt, build/{ndx_dump.py,spx_run.py,analyze.py}. Rerun order: spx_run.py, ndx_dump.py, analyze.py.
