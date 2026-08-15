# Paper Sleeve Acceptance Criteria (pre-registered)

## Status

**PRE-REGISTERED — committed 2026-08-15, before the measurement lane produced its first scheduled results.**

The afternoon mark-to-market lane (Hermes job `94022cc9cad0`, `35 21 * * 1-5` UTC) first runs **Monday 2026-08-17 21:35Z**. Everything below was written while the paper book held **four closed trades total**, which is a smoke test and not evidence. The commit timestamp is the point: a bar written after the numbers exist is not a bar.

**Any change to this document after 2026-08-17 must be a new commit with a stated reason, and must not be made while a decision is pending.** See [Amendment rule](#amendment-rule).

## Goal

Define, in advance and numerically, what would make the paper sleeves worth promoting, worth continuing, or worth stopping — so that "is it working?" and "what counts as enough data?" cannot be renegotiated in the presence of results.

This is not a strategy-tuning protocol. It sets acceptance thresholds only.

## What is being measured

| Stream | Status |
|---|---|
| `sniper \| mas_official` | Live book. Measured. |
| `mr \| mas_official` | Live book. Measured. |
| `pead_paper` | Quarantined paper. Measured, never in the book. |
| `pead_neglected` | Quarantined paper. Measured, never in the book. |

**Streams are measured separately and never blended.** Blending `mas_official` with the manual sleeve is what produced the false "MR is a coin flip" conclusion; the same error is available here and is explicitly out of bounds.

**Source of truth:** `outcome` rows in the mirror DB, stamped by `run_afternoon_check` via the afternoon lane, and the per-stream `alpha_summary` in the exported bundle. Not screenshots, not the Discord brief, not a hand-tallied list.

## Definitions

These exist so that `n` and "a valid week" are fixed before anyone wants them to be something else.

**Closed trade.** An `outcome` row with `still_open = false` and non-null `pnl_pct`, whose `entry_date <= to_date`. Positions whose entry is in the future (the `to_date < entry_date` guard) are not trades and are not counted.

**n.** Count of closed trades **per stream**. Never pooled across streams to reach a threshold.

**Valid measurement day.** A scheduled afternoon run that produced a bundle with a non-null `dashboard_sha256` and exited zero.

**Valid week.** A Mon–Fri week with **≥ 4 of 5 valid measurement days**. Trades from an invalid week still count toward `n` — they happened — but an invalid week does not count toward any elapsed-time criterion, and:

> **Two consecutive invalid weeks halt the measurement.** Fix the lane, then resume. A sample drawn from a feed that is silently skipping days is not a random sample of market conditions, and the 2026-08 earnings-feed incident is precedent: a coverage gate that could not detect its own broken feed.

## The bar

### Tier 0 — Display only (already in effect)

`_alpha_summary` (`scripts/export_dashboard_data.py:66`) emits per-stream stats at **n ≥ 3**. This threshold exists so the dashboard has something to render. **It is not a read.** No decision of any kind may cite a stream below Tier 1.

### Tier 1 — First permissible read: **n ≥ 30 closed trades per stream**

Inherited from `min_stat_trades = 30` (`src/backtest/validation_card.py:308`) and the `min_regime_trades = 30` veto floor from PR #40. The repo already refuses to let a cohort veto below 30; a paper sleeve should not get a weaker standard than a regime cohort.

Below n = 30 the only permitted statements are descriptive: "n closed trades so far, mean X%, CI crosses zero."

### Tier 2 — Promotion candidate

**All** of the following, on the same stream, simultaneously:

1. **n ≥ 30** closed trades.
2. **Bootstrap 95% CI of mean alpha vs SPY strictly above zero** — i.e. `alpha_summary.ci_low > 0`. A positive mean whose CI crosses zero is a lean, not an edge; that sentence is already the docstring of the function that computes it.
3. **≥ 2 distinct market regimes represented**, each with ≥ 10 closed trades. A sleeve that has only ever traded one regime has not been tested.
4. **No execution-config drift** during the measurement window (see [Invalidating conditions](#invalidating-conditions)).

Tier 2 makes a sleeve *eligible for a promotion discussion*, not promoted. Promotion remains Ray's decision and still goes through the validation card.

### Tier 3 — Threshold/parameter claims: **n ≥ 100**

Inherited from `min_threshold_trades = 100` (`validation_card.py:336`). Any claim of the form "threshold X is better than Y" needs 100 closed trades on that stream. Below that, parameter differences are noise — the repeated finding of this project is that no tunable parameter improves MR (score IC ≈ 0).

## Stop condition — stated as a number, not a judgment

A stream **stops** (paper trading halted, sleeve retired or rebuilt) when **either** fires:

**S1 — Statistically established negative.** At **n ≥ 30**, the bootstrap 95% CI of mean alpha is **entirely below zero** (`ci_high < 0`). This is the symmetric mirror of Tier 2's promotion test. It cannot be argued away by "small sample" — 30 is the same floor promotion must clear.

**S2 — Drawdown breach.** Paper-sleeve equity drawdown (concurrency-capped, as computed by the unified exit engine, not a sum of per-trade returns) reaches the number in the table below. This one fires regardless of `n`, because a large enough loss is decision-relevant before it is statistically significant.

> **Neither stop condition is discretionary.** If S1 or S2 fires, the sleeve stops and the restart requires a written reason. "It's about to turn around" is not a reason.

## Thresholds Ray must set

Everything above is derived from thresholds already in the codebase. These three are genuinely new and are **Ray's to set** — I've proposed defaults, they are not decisions I should make:

| Parameter | Proposed | Rationale |
|---|---|---|
| **S2 drawdown breach** | **20%** | Sniper's Run E backtest showed 24% DD; a paper sleeve exceeding 20% is behaving worse than its own already-unflattering backtest. |
| **Max measurement window** | **6 months** | If a stream cannot reach n = 30 in 6 valid months, its signal supply is too thin to matter regardless of edge. Supply, not the cap, is why picks are ~1.2/day (#68). |
| **PEAD promotion size** | **small third sleeve, sizing TBD** | Already the standing plan; the number has never been written down. |

## Invalidating conditions

The measurement itself is void — reset `n`, do not merge the windows — if any of these happen mid-window:

- **Execution-config drift.** Trail activate/distance, slippage, stop/target multiples, or hold period change for a measured model. This is not hypothetical: PEAD ran a global 0.5/0.3 trail its justifying backtest never had, and that alone cost the entire edge (+2.21% → +0.10%/trade). A config change starts a new window.
- **Mirror falls behind `main`.** The measurement is only valid for the code it ran. The mirror must be fast-forwarded before each PAPER run; a window spanning a stale checkout is void for the stale portion.
- **Stream blending.** Any read that pools streams to reach a threshold voids that read.
- **Provenance gap.** Artifacts without a `get_last_ohlcv_provenance()` stamp are not admissible, per the repo's provenance rule.

## Not quotable

> **Until a stream reaches n ≥ 30, its numbers are not quotable outside a descriptive sentence that includes `n` and states that the CI crosses zero.**

Not in the Discord brief as a headline, not in a PR body as justification, not to argue for a config change, and not as "early results look good." As of this writing the entire paper book has **four closed trades across two streams** — ANET +5.39%, TQQQ −0.12%, IOT −0.15% (`sniper|mas_official`, n=3) and BRK-B −3.77% (`pead_neglected`, n=1, below even the display floor).

This project has retired an 82% sniper win rate and a 69.5% MR win rate that were both artifacts. The cost of quoting early numbers is not embarrassment; it is that they get built on.

## Amendment rule

1. Amendments are commits to this file with a stated reason in the commit body.
2. **No amendment while a decision is pending on the stream it affects.** If a sleeve is at n = 28 and someone wants to lower Tier 1 to 25, the answer is no.
3. Lowering a threshold requires a reason that is not "the current data would pass the lower one."
4. Raising a threshold is always allowed.
