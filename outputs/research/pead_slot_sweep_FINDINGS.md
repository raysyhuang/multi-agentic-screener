# PEAD concurrency + entry-limit sweep — harness + synthetic lock only (2026-08-15)

Pre-registered in STRATEGY_REVIEW_2026-08 §0.2 and §2.6. Follow-up to
`pead_FINDINGS.md`, which found mid-third concurrency-capped equity lost
(−12.7% CAGR) while per-trade was still +1.01%, and to the live observation
that 13 PEAD positions were open on 2026-08-06 against a "cap" of 5 (which is
actually a per-run pick quota). With 20-day untrailed holds (since PR #43),
earnings-season clustering can make PEAD the book's largest exposure.

**This PR delivers the HARNESS and SYNTHETIC LOCK. §0.2 and §2.6 remain open
until a real PEAD trade list is run on a checkout that has the cache.**

Script: `scripts/pead_slot_sweep.py`. Tests: `tests/test_pead_slot_sweep.py`
(15 tests, all pass in CI without data files). Same lesson as `sniper_pick_count_FINDINGS.md`:
the comparison is capital-aware compounded return vs summed per-trade P&L.

## Question

On a PEAD trade list, after capital is real (slots, not summed per-trade P&L):

1. What is peak concurrent open PEAD when uncapped (20-day holds)?
2. At open-slot caps {3, 5, 8, 10, uncapped}, what are taken/skipped/peak/return/maxDD?
   (10 is the live `pead_max_concurrent` on main; include for ranking vs shipping default.)
3. Crossed grid (§2.6): max-entries-per-week {2, 3, 5} × sector cap {None, 2} ×
   slot caps {3, 5, 8, 10, uncapped}. Weekly/sector are PRE-FILTERS on the trade
   list, then replayed through capped-equity sim at each slot cap.
4. Pre-registered kill note (§0.2): if a cap would exclude >30% of the backtest
   cohort, write that the 30-trade promotion clock restarts under that cap.

## Method

Event-driven portfolio simulator (`src/backtest/portfolio.py` `simulate_book`)
with equal-weight slots and a concurrency cap. Weekly/sector limits are
PRE-FILTERS on the trade list (not portfolio sim changes). Synthetic unit tests
lock the mechanism without requiring the 3Y OHLCV parquet or real PEAD trade
cache:

- `test_uncapped_measures_peak_concurrent`: 25 consecutive-daily 20-day trades
  → peak concurrent = 20 (the hold period).
- `test_open_slot_cap_skips_excess`: cap=5 on stacked earnings week skips trades.
- `test_weekly_entry_limit_binds`: max 3 entries/7d pre-filter drops excess.
- `test_sector_cap_binds`: max 2 per sector concurrently pre-filters by sector.
- `test_exclusion_threshold_check`: cap=3 on 100 consecutive 20-day trades
  excludes >30% (triggers the kill note).
- `test_weekly_plus_slot_cap_skips_more`: weekly pre-filter + slot cap together
  skip more than either alone (§2.6 grid mechanism).

All 15 unit tests pass in CI without data files (verified 2026-08-15).

## Results

**COHORT DATA NOT PRESENT IN THIS CHECKOUT.** The 3Y OHLCV parquet and PEAD
trade cache are gitignored and were not in this clone at sweep time. Synthetic
unit tests LOCK the mechanism; the full cohort sweep is for a checkout that has
`outputs/research/ohlcv_polygon_3y.parquet` and the PEAD trade list.

To run the full sweep when data is present:

```bash
python scripts/pead_slot_sweep.py --cohort <pead_trades.csv>
```

The script exits 0 with a message if the cohort file is missing.

## Synthetic test verdict

The mechanism is locked:

- Uncapped replay measures peak concurrent correctly (20 for 20-day holds).
- Open-slot caps skip excess entries when slots are full.
- Weekly entry limits bind as a pre-filter.
- Sector caps bind when sector data is present (>50% non-null).
- The 30% exclusion threshold check is computable (cap=3 on 100 consecutive
  20-day trades excludes >30%).

## Interpretation for live config (pending real cohort)

Once real cohort data is available, the sweep will answer:

1. **Peak concurrent uncapped**: how many PEAD positions stack in earnings season?
2. **Open-slot cap {3, 5, 8, 10}**: which balances skipped-signal cost vs book
   exposure? (10 is the current live `pead_max_concurrent` on main.)
3. **Crossed grid (§2.6)**: for each weekly limit {2, 3, 5} and sector cap
   {None, 2}, what are the taken/skipped/peak/return/maxDD at slot caps
   {3, 5, 8, 10, uncapped}? Does a weekly pre-filter flatten the peak without
   dropping >30% of the cohort (which would restart the 30-trade promotion clock)?
4. **Sector cap**: if sector data is available, does max 2 per sector keep
   PEAD from becoming a single-sector bet?

The pre-registered kill rule from §0.2: if a candidate cap would have **excluded
>30% of the backtest cohort**, the 30-trade promotion clock restarts under that
cap — do not ship it. Report taken/skipped, peak concurrent, compounded return,
maxDD. Win rate is NOT the result (same lesson as sniper_pick_count).

## Caveats

- This is a RESEARCH-ONLY PR. No production PEAD behavior, ranker, gates, or
  live caps were changed.
- Relative arms only: the comparison is compounded return at different caps on
  a fixed population, not absolute returns (which depend on universe, costs,
  and time window).
- The sniper_pick_count lesson applies: summed per-trade P&L is not a portfolio
  result. Capital is a constraint. A wider quota or no cap can HURT if it fills
  slots with marginal signals that then skip genuinely better ones later.
- The 30% exclusion threshold is a pre-registered hard stop: a cap that drops
  >30% of the cohort changes what the paper trial measures, so the promotion
  clock restarts. This is not a "30% is always bad" rule — it is a guardrail
  against shipping a cap that was not validated by the paper stream.

## Next step (when cohort data is in-tree)

Run the full sweep on the real PEAD trade list and populate this findings doc
with the actual metrics table. The synthetic tests ensure the mechanism is
correct; the real cohort will answer which cap (if any) should become live.

Until then: unit tests pass, mechanism is locked, no production changes.

---

**Research-only.** See CLAUDE.md: `outputs/` is gitignored; this markdown was
force-added via `git add -f` and requires `ALLOW_LOCAL_FILES=1` on commit.
