# Taiwan W×H Top3 momentum grid with start-month sensitivity — PRE-REGISTERED SPEC
Written 2026-11-12 (UTC+8) BEFORE any grid run. Research / paper only. Results provisional.
Requested by Ray via Queen ("W1-12，H1-12，top3，month sensitive 台股").

## Inputs (reused, read-only; nothing in /workspace/taiwan-winner-check-2026-10-21/ is modified)
- PIT membership (Wikipedia/FTSE reconstruction, SECONDARY source):
  TW50 = membership_pit.json; TW150 = membership_tw150_pit.json; MC100 = membership_mc100_pit.json.
  The synthetic '1900-01-01' back-fill era is NEVER used: seasoning windows must start on/after the first real
  snapshot (TW50/TW150 2003-01-20; MC100 2004-12-20). Snapshots run through 2026-09-21.
- Prices: existing FinMind extract raw/TaiwanStockPrice_*.json (raw, incl. delisted; rows with close>0 and volume>0).
- Adjustments: the loader from short_mom/grid_tw150.py copied verbatim in logic (FinMind DividendResult/Dividend,
  TWSE TWT49U fallback, FinMind CR / TWSE TWTAUU fallback, SplitPrice, gap_events.csv pre-2011 CR proxies,
  MC100 extra gap CR list). Unknown 權息 stock/cash split: price-only factor bounded lo=1 / hi=TR factor.
  lo is PRIMARY; hi run as robustness and mentioned only where it flips a vs-0050 comparison.
- No new downloads of any current-constituent list. Any genuinely missing price is logged, never silently skipped.

## Grid
W (lookback months) 1..12 × H (hold months) 1..12 × Top3 equal weight × start month M 1..12 × universe {TW50, TW150, MC100}.

## Rules (frozen)
- Trading calendar for month-ends: 2330 trading dates (as prior tests). Signal = last trading day of the signal month.
  Only complete months are used: last allowed signal month = 2026-09.
- Phase: for start month M, the anchor is the earliest month-end with calendar month M such that
  (a) s − 365 calendar days ≥ first real membership snapshot of that universe, and
  (b) the base month-end W months before s is ≥ 2003-01 (first price month).
  Signals = anchor, anchor+H months, anchor+2H, … (year wrap). Periods are non-overlapping.
- Eligibility at s: continuously a member over [s − 365d, s] (all snapshots in that interval).
- Rank key: W-month price-only return, close on/before the base month-end → close on/before s, times price-only
  event factors (stock dividends, splits/par changes, capital reductions; NOT cash dividends). Both closes must be
  within 15 calendar days of their target dates, else the name is dropped from that ranking and logged.
  Members with no price file / no valid rows are logged.
- Picks: Top 3 by rank key. If fewer than 3 rankable names, hold what exists and flag.
- Trade: buy each pick at its first open strictly after s; mark daily; sell at close of next signal date e.
  Delisted / halted names: last close carried as cash. A pick with no trade at all in (s, e] is carried as cash
  at its signal-date value (gross return 0) and flagged.
  Costs every period (full round trip, even if retained): ×0.997 at buy, ×0.994 at sell.
- Bases: price-only (fp factors) and total return (ftr factors: cash dividends reinvested at ex reference, plus
  stock dividends / CR). Same picks.
- Benchmark: 0050 TR on the identical per-period windows (first 0050 open after s → 0050 close at e), no costs, chained.
- Censoring: the period still open at the last data date (2026-10-21) is censored and only listed as current picks.
- Windows (completed periods whose hold END date is inside the window):
  5y = 2021-01-01..2025-12-31, 10y = 2016-01-01..2025-12-31, full = 2005-01-01..2025-12-31.
  If no feasible period ends in 2005 (e.g. MC100), the full window starts at the first feasible period; labeled.
  CAGR uses the actual span (first included signal date → last included end date, /365.25).
- Metrics per (universe, W, H, M, window): n periods, net price multiple, net TR multiple, TR CAGR, max daily TR
  drawdown (chained mark-to-market incl. buy cost), % periods TR net > 0050 TR, worst TR period, 0050 TR multiple,
  n flagged periods (<3 names or no-trade pick).
- OFFICIAL phase: M = 12 (December-end signals, then every H months). The other 11 months are SENSITIVITY ONLY;
  the best month is never presented as the official result.
- Month-sensitivity summary per (universe, W, H, window): the 12 TR multiples, min/median/max, # of 12 months whose
  TR multiple > 0050 TR multiple on their own periods, best and worst month. An edge is called "month-fragile" if
  fewer than 9 of 12 start months beat 0050 in that window.
- Sanity gate BEFORE publishing the grid: replicate TW150 W6 H6 M12, lo: TR 9.92x / 22.39x / 22.63x
  (n=10/20/42), 0050 ≈2.48x/5.69x/10.40x; and M12 H12 0050 TR ≈2.49x/5.83x/10.79x. Material miss (>2% relative on
  any multiple) ⇒ stop, diagnose, report; no full grid.
- Multiple testing: 12×12×12×3 = 5184 cells (×2 bounds) — any "best" cell is heavily data-mined. Descriptive only.
