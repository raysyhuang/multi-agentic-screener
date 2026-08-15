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

**Exact field paths** — named now so no test can be "clarified" later into whichever field happens to pass:

| Quantity | Path |
|---|---|
| Mean alpha CI, lower bound | `alpha_summary[<stream>]["spy"]["ci_lo"]` |
| Mean alpha CI, upper bound | `alpha_summary[<stream>]["spy"]["ci_hi"]` |
| Sample size | `alpha_summary[<stream>]["spy"]["n"]` |
| Per-trade alpha | `trades[<stream>][i]["alpha_spy"]` |
| Regime for a trade | **join** `trades[<stream>][i]["signal_date"]` → `run_history[date == signal_date]["regime"]` |

**The benchmark is SPY, fixed in advance.** `BENCHMARKS = {"spy": "SPY", "qqq": "QQQ"}` (`export_dashboard_data.py:43`), so two benchmarks exist and both are exported. Every test in this document reads `["spy"]`. Choosing the benchmark after seeing which one passes is not permitted.

**Do not use the `significant` field for any test here.** It is two-sided — `bool(lo > 0 or hi < 0)` (`:89`) — while Tier 2 and S1 are both one-sided. Read `ci_lo` and `ci_hi` directly.

## Definitions

These exist so that `n` and "a valid week" are fixed before anyone wants them to be something else.

**Closed trade.** An `outcome` row with `still_open = false` and non-null `pnl_pct`, whose `entry_date <= to_date`. Positions whose entry is in the future (the `to_date < entry_date` guard) are not trades and are not counted.

**n.** Count of **countable** closed trades **per stream**. Never pooled across streams to reach a threshold.

**Valid measurement day.** A scheduled afternoon run that produced a bundle with a non-null `dashboard_sha256` and exited zero.

**Countable trade.** A closed trade counts toward `n` only if its **`exit_date` fell on a valid measurement day**. The trade happened either way, but its `pnl_pct` was stamped by a run that may have missed days, and a mark produced by a lane that skipped bars is not a mark to build a threshold on. Uncountable trades remain visible in `trades[]` and are excluded from `n`, Tier tests, and S1.

**Valid week.** A Mon–Fri week with **≥ 4 of 5 valid measurement days**. An invalid week does not count toward any elapsed-time criterion, and:

> **Two consecutive invalid weeks halt the measurement.** Fix the lane, then resume. A sample drawn from a feed that is silently skipping days is not a random sample of market conditions, and the 2026-08 earnings-feed incident is precedent: a coverage gate that could not detect its own broken feed.

## The bar

### Tier 0 — Display only (already in effect)

`_alpha_summary` (`scripts/export_dashboard_data.py:66`) emits per-stream stats at **n ≥ 3**. This threshold exists so the dashboard has something to render. **It is not a read.** No decision of any kind may cite a stream below Tier 1.

### Evaluation points — fixed in advance

The lane runs daily and `n` grows daily, but the CI is a fixed 95%. **Under repeated evaluation a null stream will eventually clear zero by chance** — a 95% interval checked sixty times is not a 95% interval.

> **The bar may only be evaluated at n = 30, n = 50, and n = 100. Nowhere else.**

Between those points the numbers may be *described* (see [Not quotable](#not-quotable)) but no Tier or Stop determination may be made. If a stream passes through an evaluation point without anyone looking, the next permitted look is the next point — not the day someone noticed.

### Tier 1 — First permissible read: **n ≥ 30 closed trades per stream**

Inherited from `min_stat_trades = 30` (`src/backtest/validation_card.py:308`) and the `min_regime_trades = 30` veto floor from PR #40. The repo already refuses to let a cohort veto below 30; a paper sleeve should not get a weaker standard than a regime cohort.

Below n = 30 the only permitted statements are descriptive: "n closed trades so far, mean X%, CI crosses zero."

### Tier 2 — Promotion candidate

**All** of the following, on the same stream, simultaneously:

1. **n ≥ 30** closed trades — `alpha_summary[<stream>]["spy"]["n"] >= 30`.
2. **Bootstrap 95% CI of mean alpha vs SPY strictly above zero** — `alpha_summary[<stream>]["spy"]["ci_lo"] > 0`. A positive mean whose CI crosses zero is a lean, not an edge; that sentence is already the docstring of the function that computes it.
3. **≥ 2 distinct market regimes represented, each with ≥ 10 closed trades**, where a trade's regime is obtained by joining `trades[<stream>][i]["signal_date"]` to `run_history[date == signal_date]["regime"]`, using the repo's `bull` / `bear` / `choppy` keys. Trades whose `signal_date` has no matching `run_history` row are **excluded from the regime count** (they still count toward `n`). A sleeve that has only ever traded one regime has not been tested.
4. **No execution-config drift** during the measurement window (see [Invalidating conditions](#invalidating-conditions)).

Tier 2 makes a sleeve *eligible for a promotion discussion*, not promoted. Promotion remains Ray's decision and still goes through the validation card.

### Tier 3 — Threshold/parameter claims: **n ≥ 100**

Inherited from `min_threshold_trades = 100` (`validation_card.py:336`). Any claim of the form "threshold X is better than Y" needs 100 closed trades on that stream. Below that, parameter differences are noise — the repeated finding of this project is that no tunable parameter improves MR (score IC ≈ 0).

## Stop condition — stated as a number, not a judgment

A stream **stops** (paper trading halted, sleeve retired or rebuilt) when **either** fires:

**S1 — Statistically established negative.** At **n ≥ 30**, the bootstrap 95% CI of mean alpha is **entirely below zero** (`ci_high < 0`). This is the symmetric mirror of Tier 2's promotion test. It cannot be argued away by "small sample" — 30 is the same floor promotion must clear.

**S2 — Drawdown breach.** Paper-sleeve equity drawdown (concurrency-capped, as computed by the unified exit engine, not a sum of per-trade returns) reaches the number in the table below. This one fires regardless of `n`, because a large enough loss is decision-relevant before it is statistically significant.

> **Neither stop condition is discretionary.** If S1 or S2 fires, the sleeve stops and the restart requires a written reason. "It's about to turn around" is not a reason.

## Known limitation of the CI — it is optimistic

`_alpha_summary` bootstraps by resampling **trades** iid (`rng.choices(a, k=n)`, `export_dashboard_data.py:81`). Concurrent positions share the same day's market move. Alpha-vs-SPY strips the index factor but **not** sector co-movement or same-day clustering, so the effective sample is smaller than `n` and **the interval is narrower than it should be**.

Consequence: the CI is biased toward *firing* — both Tier 2 promotion and S1 stop trigger more readily than a correct interval would justify. This is stated here rather than silently inherited.

Two mitigations, in order of preference:
1. **Proper fix** — resample entry-*days* rather than trades. Requires a code change to `_alpha_summary` and is the right answer if anyone has the time.
2. **Zero-code mitigation, in force until then** — Tier 2 must clear at a **pre-named evaluation point** (n = 30 / 50 / 100), never "at some point when it happened to look good." The fixed evaluation points above are what keep an optimistic interval from being harvested.

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

## Pre-registered prediction

Recorded **before** any scheduled result, so that it cannot be reinterpreted afterward:

> **The most likely outcome under this bar is that `pead_neglected`, and possibly `mr`, reach the maximum measurement window without ever reaching n = 30. They die of signal supply, not of negative alpha.**

If that happens, the correct report is **"we never got a read"** — not "it didn't work." Those are different findings with different consequences: the first says the sleeve is untestable at current supply and the question is whether supply can be raised; the second says the edge is absent. Conflating them retires a strategy for the wrong reason.

Supporting context: picks run ~1.2/day and #68 established that the sniper slot **cap is not the binding constraint — supply is**. `pead_neglected` currently has one closed trade.

## Amendment rule

1. Amendments are commits to this file with a stated reason in the commit body.
2. **No amendment while a decision is pending on the stream it affects.** If a sleeve is at n = 28 and someone wants to lower Tier 1 to 25, the answer is no.
3. Lowering a threshold requires a reason that is not "the current data would pass the lower one."
4. Raising a threshold is always allowed.
