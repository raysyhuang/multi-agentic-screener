# Selection quality: does rank ordering carry information? (2026-08-08)

**Answer: almost none. The top-2-of-N choice is close to random, and the
review's own prior does not survive its CIs.** Script: `scripts/rank_quality_audit.py`.

Tier-1 item 2.2 from `STRATEGY_REVIEW_2026-08.md`. This is the "hit rate"
question in its sharpest form: the pipeline ranks ~10 candidates daily and takes
2. If score→outcome ordering is noise, then which two get picked is a coin flip
and all the alpha is in *being in the ten*.

## Result (days that actually had a choice: >2 candidates)

| cohort | rank 1-2 | rank 3+ | edge | 95% CI | per-year |
|---|---|---|---|---|---|
| Sniper (Run E, fill-realistic) | +0.235% (n=392) | +0.006% (n=586) | **+0.229pp** | **[−0.430, +0.881]** ✗ | 2024 −0.03 / 2025 **−0.40** / 2026 +0.59 |
| MR (3Y Polygon) | −0.057% (n=1338) | +0.044% (n=26463) | **−0.101pp** | [−0.213, +0.012] ✗ | 2023 +0.34 / 2024 +0.20 / 2025 **−0.50** / 2026 −0.07 |

Spearman(rank, pnl): sniper **−0.060**, MR **+0.048** — both ≈ 0, and MR's is
the wrong sign (worse ranks did slightly better).

**The review's prior is refuted.** It reported sniper rank 1-2 at +0.47%/trade vs
rank 3+ at +0.01% and called the ordering "real". At a like-for-like comparison
(restricted to days with a genuine choice), with a bootstrap CI and a per-year
split, the edge is +0.229pp with a CI spanning zero and a **sign flip in 2025**.
The whole apparent effect rests on 2026 — the same one-good-year artifact the
per-year rule was adopted to catch, now caught in a claim the review itself made.

## The number that matters most

Comparing what the ranker achieves against random selection and perfect foresight
on the same daily candidate sets:

| cohort | actual top-2 | random-2 | perfect foresight | **value captured** |
|---|---|---|---|---|
| Sniper | +0.235% | +0.025% | +2.905% | **7.3%** |
| MR | −0.057% | −0.135% | +3.513% | **2.1%** |

There is **~3pp/day of selection value sitting on the table** and the ranker
captures 2-7% of it. That is not a tuning problem — a score with IC ≈ 0 cannot
be tuned into one with IC > 0. It is the strongest confirmation yet of the
project's recurring finding, from a new direction: **the gates are the alpha,
the ranking is not.**

## Does taking MORE picks help instead?

If selection is near-random, pick *count* should matter more than pick *quality*
— and the book runs at ~32% gross with a 10-slot cap that never binds.

| top-k/day | Sniper avg | Sniper sum | MR avg | MR sum |
|---|---|---|---|---|
| 1 | +0.489% | +217.6pp | −0.037% | −25.1pp |
| 2 (live) | +0.471% | +352.7pp | −0.057% | −78.0pp |
| 4 | +0.451% | **+481.0pp** | −0.081% | −217.8pp |
| 6 | +0.432% | **+523.9pp** | −0.088% | −347.5pp |
| 10 | +0.340% | +443.5pp | −0.095% | −600.5pp |

- **Sniper**: per-trade decays only slightly out to k=6 while total nearly
  **doubles** (+353 → +524pp). Widening sniper is a genuine candidate — but note
  2023 is negative at every k, and the per-trade decay past k=6 is real.
- **MR**: negative at every k on this unselected population and gets **worse**
  with more picks. Do not widen MR. (Consistent with MR being edgeless before the
  live gate funnel — the live +0.48%/trade comes from selection this cohort
  doesn't model.)

## What this does and does not license

- ✅ **Stop investing in ranking quality.** Cross-model score normalization,
  confluence bonuses, tie-breakers — all operate on a signal with no measurable
  ordering information.
- ✅ **Sniper pick-count widening is worth a proper test** (capped-equity sim
  with concurrency, not just summed per-trade — the sum column above ignores
  capital constraints entirely).
- ❌ **Does not license widening MR.**
- ❌ **Does not settle the LIVE question.** These are backtest cohorts; the
  sniper universe caveat applies (relative arms only). The live arm now ships:
  `export_dashboard_data.py` exports the `candidates` section the pipeline has
  been persisting all along, and `rank_quality_audit.py` reads it. It needs ~40
  resolved picked-vs-passed candidates before it reports.

## Method note

The first cut compared all rank 1-2 trades against all rank 3+ trades, including
days with ≤2 candidates where no selection occurred. Restricting to days with a
genuine choice is what makes it a test of *selection* rather than of daily
opportunity count.
