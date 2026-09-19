# H2 — MR avoid-filter after an earnings miss: REJECTED at G1 (2026-09-19, v2 after Codex review)

**Verdict: REJECTED — the sign is right, but the effect is not established.** The hypothesis was that MR trades entered shortly after a ≤ −10% EPS miss do worse than other MR trades. At live selectivity (min_score 75) they do lose money, and they underperform the rest in every year. But condition (b) requires the date-cluster CI upper bound of the difference to be below 0, and it is not in any run:

| Run | Post-miss − rest | Date-cluster CI |
|---|---|---|
| Registered run (v1) | −0.28pp (n=1,167) | [−0.64, +0.08] |
| v2, refreshed earnings, registered "known" rule | −0.28pp (n=1,185) | [−0.65, +0.09] |
| v2, look-ahead-free "known next day" rule | −0.29pp (n=1,085) | **[−0.61, +0.02]** |

**Codex's review found the registered rule leaks information.** The rule treats a report as known on its own date. FMP gives a date but no release time, so an after-close report is not yet known at that day's close, which is when the MR decision is made.

With the leak removed on the review's pre-refresh data, H2 **passed** narrowly: CI [−0.63, −0.007]. After the stale earnings cache was refreshed, the same leak-free rule gives an upper bound of **+0.02**, and H2 fails again.

A verdict that flips on refreshing a data cache is not an established effect in either direction. H2 stays REJECTED under its pre-registered rule, which names "right sign, CI crosses zero" as a failure in advance.

## Pre-registration

- The criteria are in the docstring of `scripts/h2_mr_post_miss.py`. It was committed in `d44fc9c` and pushed at 2026-09-19T05:14:02Z, before any H2 computation.
- The "known next day" rule is a post-review amendment recorded in the same docstring. It is not a registered criterion.
- A second review pass found that the first implementation of that rule picked the latest report *before* checking whether it was known yet, so an unavailable same-day report could hide an earlier known miss (CERE 2024-05-08). Availability is now applied first. This moved one trade (n 1,084 → 1,085) and left the CI upper bound at +0.02.

## Results, v2 (`h2_mr_post_miss_v2_{registered,next_day}.json`)

**Condition by condition:**

| Condition | Registered rule | Next-day rule |
|---|---|---|
| (a) n ≥ 30 | 1,185 ✓ | 1,085 ✓ |
| (b) diff CI upper bound < 0 | +0.09 ✗ | +0.02 ✗ |
| (c) flagged mean < 0 | −0.29% ✓ | −0.31% ✓ |
| (d) negative in ≥ 2 years (n ≥ 10) | 4/4 ✓ | 3/3 ✓ |

**Descriptive cells, next-day rule** (variants; none can rescue a fail):

| Cell | Diff | CI |
|---|---|---|
| 5-session window | −0.33pp | [−0.74, +0.08] |
| **20-session window** | −0.27pp | **[−0.49, −0.05]** |
| miss ≤ −5% | −0.25pp | [−0.56, +0.06] |
| mirror: post-beat | +0.15pp | [−0.11, +0.41] |
| min_score 50 | −0.005pp | [−0.25, +0.23] |

## Reading

- **The direction is consistent everywhere.** Every window, every threshold, both "known" rules and every year point the same way, and post-beat trades are mirror-positive. That pattern is what a slow reaction to bad news would produce, but an uncertain association does not establish the mechanism.
- **It is still not established.** At about −0.28pp per trade the effect is small relative to its noise.
- **The 20-session window is significant under both rules, but it is post-hoc.** It was chosen from five descriptive cuts after the results were visible. It is logged as a lead to test on MR trades from after 2026-09-18, which this study never saw. At roughly 400 flagged trades a year, a decisive forward read needs about a year.
- **The backtested MR population shows no edge at this universe's breadth.** The unflagged backtest trades average −0.01% per trade. That is a statement about the broad backtest, not about the live book, whose n=49 cannot settle it either way.

## Caveats

- These are backtested MR trades from the unified exit engine at live parameters, not the live book; the live book has only n=49.
- The universe and earnings caveats are the same as H1 (`h1_pead_wide_FINDINGS.md` § 6).

## Reproduce

```bash
python scripts/h2_mr_post_miss.py --known registered --json-out outputs/research/h2_mr_post_miss_v2_registered.json
python scripts/h2_mr_post_miss.py --known next_day   --json-out outputs/research/h2_mr_post_miss_v2_next_day.json
```
