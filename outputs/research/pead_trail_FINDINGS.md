# PEAD live-vs-backtest config mismatch (2026-08-04)

**Result: the global trail was costing PEAD ~2.1pp per trade — its entire edge.
Fixed by making trail width per-model.**

Script: `scripts/pead_trail_decompose.py`.

## The mismatch

`scripts/pead_backtest.py:98` — the study that justified PEAD — calls
`simulate_trade(...)` with **no** `trail_activate_pct` / `trail_distance_pct`.
Both default to `0.0`, so it ran with **no trailing stop**, a 20-day hold and a
3×ATR stop. The live tracker (`src/output/performance.py`) applied the **global**
0.5%/0.3% trail, because nothing special-cased the model. Same root as the
`Unknown signal_model 'pead'` health warning: PEAD inherited MR/sniper-shaped
defaults everywhere.

Symptom visible in live data before any backtest: **median hold of 1 day on a
20-day drift strategy**, with winners capturing 20% of their peak.

## Decomposition — live E1-gated population (n=306, 504 tickers, 3Y)

| config | WR | avg/trade | 95% CI | med hold | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| A designed (3×ATR, no trail, 7.5bp) | 58.8% | +2.263% | [+1.21,+3.31] | 28d | +6.04 | +2.44 | +2.30 | +0.93 |
| **B LIVE EXACT (tiered stop, trail 0.5/0.3, 10bp)** | 53.9% | **+0.099%** | [−0.35,+0.55] | 2d | −0.74 | +0.09 | +0.21 | +0.22 |
| C live minus trail | 55.9% | +2.101% | [+1.08,+3.13] | 28d | +5.18 | +2.13 | +1.95 | +1.38 |
| D live minus tier-bug | 54.9% | +0.076% | [−0.40,+0.56] | 2.5d | −0.23 | +0.24 | +0.05 | −0.01 |
| **E FIX (3×ATR, no trail, 10bp)** | 58.8% | **+2.212%** | [+1.16,+3.26] | 28d | +5.98 | +2.39 | +2.25 | +0.88 |

**The trail is the entire problem.** Removing it alone: +0.099 → +2.101.
Removing the tiered-stop rescaling alone: +0.099 → +0.076 — no help at all.

Raw EPS≥10% population (n=1574) shows the same shape: +1.811% designed vs
−0.002% live-exact. Cost (7.5 → 10bp) is worth −0.05pp; irrelevant next to the trail.

The E1 gate earns its keep: **+2.212% gated vs +1.760% raw**, consistent with the
project's recurring finding that the gates are where the alpha lives.

## Per-year split (the rule from `exit_layer_FINDINGS.md`)

Unlike the rejected MR stop change (3 of 4 years negative), the fix is positive
in **all four years**. But it is decaying monotonically: +5.98 → +2.39 → +2.25 →
+0.88. **Honest forward expectation is ~+0.9%/trade, not the +2.2% headline.**

A wider trail is NOT the answer — at 8%/5% the 2026 cohort is already negative
(−0.22%). The fix is *no* trail for PEAD.

## The fix

`Settings.trail_for_model(signal_model)` returns `(0.0, 0.0)` for `pead` and the
global trail for everything else. Unknown models fall back to the global value so
a new strategy is never silently un-trailed.

**MR and sniper are unchanged, deliberately.** Sniper without a trail is
−1.429%/trade at 20% WR (`exit_layer_FINDINGS.md`) — the trail is that strategy's
entire risk control. One global trail cannot serve a 3-day reversion strategy and
a 20-day drift strategy; that is the actual lesson.

## Known-remaining, not fixed here

`performance.py` computes `tier_atr = abs(entry - stop) / 0.75`, hardcoding
mean-reversion's 0.75×ATR convention. With PEAD's 3×ATR stop that yields
`tier_atr = 4×ATR`, rescaling the intended 3×ATR stop to **2.0× (score<70),
3.4× (70-84), 5.0× (≥85)**. 1073 of 1574 events land in the 2.0× bucket — a stop
33% tighter than designed. Measured cost is small (config D above), so it is left
for a separate correctness fix rather than bundled into a P&L change.

## Caveats

- No-trail carries 22% stop-outs at 3×ATR and 63-68% expiry: check the
  concurrency-capped equity curve and drawdown before sizing.
- 20-day holds tie up capital far longer than the 1-day reality PEAD has had;
  concurrency limits will bind differently.
- PEAD remains a quarantined paper stream. This changes what the paper stream
  measures, not the book.
