# No-NVDA 5,184-path grid (2026-11-14, Asia/Taipei). Research only, capital 0.

Engine: Queen v17 strict4 + v18 KORS/CPRI dedupe, loaded read-only from /workspace/second-sleeve-v17-targeted-ca-20261108/engine
(same as answer-key build/run_part.py). Only change: NVDA (single ticker NVDA throughout; no renames/CAs) removed from the
ranked list and from the missing list at every signal, so lower-ranked names move up. Everything else is unchanged:
month-end signal, first-session close, 25 bps/side, equal weight 1/K, no stop. Phase convention unchanged: phase p runs from
first session of month p 2015 to first session of month p 2025 (only P1 = 2015-01-02..2025-01-02).

## Sanity check (unchanged rerun vs answer key 2026-11-09)
5,184/5,184 status match; known paths max abs diff: final 2.8e-14, CAGR 2.8e-16, MDD 3.3e-16; UNKNOWN reason/ticker/date all match;
136,198 rebalance rows, 0 pick mismatches. Status 4341/377/466. W12H12K1P1 60.5383%, W10H2K1 median 45.2403%, W12H4K1 43.0262%.

## No-NVDA results
Best cell (12-phase median, known phases): W12H12K1 23.17% (12/12 known, min 6.37%, max 32.36%, median MDD -55.24%, worst -74.53%).
Best single path: W11H9K1P9 62.78% CAGR, MDD -42.46% (2015-09-01..2025-09-02; its cell median is only 14.94%).
Anchors: W12H12K1P1 28.15% (MDD -41.88%, 11.95x); W10H2K1 median 12.43%; W12H4K1 median 18.44%; W12H12K1 median 23.17%.
Status: PROVISIONAL_C 4343 / CONDITIONAL 366 / UNKNOWN 475 (pick_not_tradable 417, corporate_action_unresolved 58).
New UNKNOWN: 9 paths (all W4 K3, STJ corporate_action_unresolved 2017-01-05, i.e. held STJ into the Abbott deal); 0 UNKNOWN became known.
Cell medians fell in 409 of 420 comparable cells; median drop 8.5 points. SPY price-only same window: 11.02% (FMP SPY close 205.43 -> 584.64).

All numbers PROVISIONAL (membership effective dates and CA audit unchanged from v18).

## Files
paths_summary_no_nvda.csv, cells_12phase_medians_no_nvda.csv, rebalances_no_nvda.csv.gz (all picks, all paths),
rebalance_picks_best_no_nvda.csv (best cell P1, top paths, with period growth), new_unknowns_vs_base.csv, build/*.py.
