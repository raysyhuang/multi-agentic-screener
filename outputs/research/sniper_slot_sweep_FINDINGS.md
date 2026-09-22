# Sniper slot cap — would more slots deliver 2 picks/day?

**Date:** 2026-08-11 · **Script:** `scripts/sniper_slot_sweep.py` ·
**Cohort:** `sniper_truth_E_live_fixed_2026-07-26.csv` (749 signals at k≤2, 445 entry days)

## Question

Ray expects 2 sniper picks every trading day. On 2026-08-11 the pipeline produced
0 picks with 9 sniper candidates available, because `sniper_max_positions=3` was
full (ONTO d1, FSLR d0, FROG d0). Is the cap the constraint, and should it rise?

## Why PR #56 does not answer it

#56 varied the **daily quota k** at a fixed 3 slots and rejected widening. Its
mechanism was specific to that setup: with only 3 slots a wider quota fills them
with the day's marginal candidates and then skips better signals later
(`skipped` 1 → 136 → 279 as k goes 2 → 4 → 6, while `taken` plateaus at ~933).

Raising the **slot count** at k=2 has no such mechanism — the same top-2 are
taken, just held concurrently. Untested until now.

## Methodological catch

`simulate_book` sizes each position at `equity / max_concurrent`
(`portfolio.py:110`). **The cap is also the capital divisor.** Comparing returns
across caps in that model conflates two changes — more slots *and* smaller
positions — and the return decline it shows is almost entirely the sizing:

| slots | return | maxDD | (divided model) |
|---|---|---|---|
| 3 | +180.5% | 26.3% | |
| 5 | +92.6% | 16.6% | |
| 10 | +40.5% | 8.6% | |

Read naively this says "raising the cap costs 88pp of return". It says nothing of
the kind. It says a 5-slot account puts 20% of equity per trade instead of 33%.

So the sweep reports both models. **`fixed`** holds position size at `equity/3`
— today's live sizing — so extra slots add gross exposure, which is the change
actually under discussion.

## Result — fixed position size, k=2

| slots | taken | skipped | blocked days | peak concurrent | return | maxDD | Sharpe |
|---|---|---|---|---|---|---|---|
| **3 (live)** | 748 | 1 | 1 | 3 | +180.5% | 26.3% | 1.46 |
| 4 | 749 | 0 | 0 | 4 | +180.5% | 26.3% | 1.46 |
| 5 | 749 | 0 | 0 | 4 | +180.5% | 26.3% | 1.46 |
| 10 | 749 | 0 | 0 | 4 | +180.5% | 26.3% | 1.46 |

## Findings

1. **The cap is nearly non-binding on this cohort.** At k=2 it skipped **1
   signal out of 749**, on 1 day out of 445.
2. **Raising it is nearly free — and nearly pointless.** At fixed position size,
   4 slots and 10 slots produce byte-identical return and drawdown. Natural
   concurrency peaks at **4** even uncapped, so a cap of 5 is equivalent to no
   cap at all. It costs 0.0pp of drawdown and buys exactly one trade in 445 days.
3. **The cap is therefore not why throughput is ~1.2/day.** Signal supply is:
   the cohort yields **1.68 signals/day** at k≤2, so on many days fewer than two
   sniper setups exist at all. No slot count creates signals that were not there.

## The caveat that matters

This cohort is the one flagged in `research-sniper-backtest-universe`: the 3Y
Polygon cache is large-cap (median ATR% 2.28%) while sniper needs ATR% ≥ 5, so it
produces **~7× fewer trades than live**. Live signal density is higher, which is
exactly why the cap bound on 2026-08-11 while binding once in 445 backtest days.

So the numbers above are trustworthy for the **relative** question (does adding
slots hurt? no) and untrustworthy for the **absolute** one (how often does the
cap bind live?).

## Recommendation

Raising `sniper_max_positions` 3 → 5 is a low-risk change: on this cohort it is
equivalent to uncapping, with zero measured drawdown cost at live position size.
But do not expect it to deliver 2 picks/day — supply, not slots, is the limit
here, and the live blocking rate has never been measured.

**Measure it first.** PR #61 now persists ranked candidates with a `picked` flag
on every run. Counting days where sniper candidates existed, none were picked,
and 3 sniper positions were open gives the live blocking rate directly, from data
the pipeline already records. That is the number the decision should rest on, and
it accumulates without any change to production.
