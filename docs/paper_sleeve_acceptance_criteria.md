# Paper Sleeve Acceptance Criteria (pre-registered)

## Status

**PRE-REGISTERED — committed 2026-08-16, before the measurement lane produced its first scheduled results.**

Verify rather than trust this line: `git log --format='%h %ad %s' --date=iso -- docs/paper_sleeve_acceptance_criteria.md`. The earliest commit touching this file must precede **2026-08-17 21:35Z**, the afternoon lane's first scheduled run. If it does not, this document is not a pre-registration and should not be treated as one.

The afternoon mark-to-market lane (Hermes job `94022cc9cad0`, `35 21 * * 1-5` UTC) first runs **Monday 2026-08-17 21:35Z**. Everything below was written while the paper book held **four closed trades total**, which is a smoke test and not evidence. The commit timestamp is the point: a bar written after the numbers exist is not a bar.

**Any change to this document after 2026-08-17 must be a new commit with a stated reason, and must not be made while a decision is pending.** See [Amendment rule](#amendment-rule).

## Scope

This document governs **thresholds and decision rules for paper-sleeve results**. It is the single acceptance bar; there is no other.

It does **not** authorize promotion to a live executor. Clearing Tier 2 makes a sleeve eligible for a promotion *discussion*, which still goes through the validation card and remains Ray's decision.

`docs/paper_mirror_acceptance.md` (PR #89) covers **launcher operational acceptance** — did the afternoon lane run, did it stamp fills, did it avoid touching alpha-bearing code. That is a different question and a legitimate one. Any threshold or citation rule in that document is **superseded by this one**; two acceptance bars is how a result gets graded against whichever bar has no failure mode.

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

> **A row with `pnl_pct: null` is not a trade and never enters any count.** This is stated explicitly because the retracted **85.7% win rate** was computed over exactly such rows — positions that had never resolved, in a book where `_evaluate_position` had never executed. A rate computed over unresolved rows is not a low-confidence estimate; it is not a measurement at all.

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

### Time dispersion — required alongside `n`, at every tier

`n` alone is not a sample. Thirty trades entered inside one week are not thirty independent observations of anything; they are one week, sampled thirty times. This is the same clustering that makes the bootstrap CI optimistic (see [Known limitation](#known-limitation-of-the-ci--it-is-optimistic)), and it is the failure `n` by itself cannot detect.

> **Every tier threshold additionally requires that the qualifying trades span at least `max(15, n/2)` distinct trading days**, measured on `entry_date`.

| Tier | n | Required distinct entry days |
|---|---|---|
| Tier 1 | 30 | 15 |
| — | 50 | 25 |
| Tier 3 | 100 | 50 |

**It scales with `n` deliberately.** A flat 15-day floor binds at Tier 1 and is nearly free at Tier 3 — 100 trades across 15 days is 6.7 entries per day, which is exactly the clustering the constraint exists to catch, and it would pass. `max(15, n/2)` caps the average at two entries per active day at every tier.

This is a **dispersion constraint on the existing threshold, not a second count threshold.** There is exactly one sample floor — n = 30, anchored to `min_stat_trades`. The dispersion idea is adopted from the launcher acceptance draft (PR #89), which got it right where the count-only version of this document got it wrong.

**On the two date fields — this is a deliberate choice, not an inherited inconsistency.** Countability is measured on `exit_date`; dispersion on `entry_date`. They answer different questions:

- **Countability asks whether the mark is trustworthy.** A `pnl_pct` is stamped at exit, so a trade counts only if *its exit* landed on a valid measurement day.
- **Dispersion asks whether the bets are independent.** Correlated exposure is created when positions are *opened* into the same conditions, so clustering is measured on entry.

Using one field for both would break one of the two tests.

### Tier 1 — First permissible read: **n ≥ 30 closed trades per stream**

Inherited from `min_stat_trades = 30` (`src/backtest/validation_card.py:308`) and the `min_regime_trades = 30` veto floor from PR #40. The repo already refuses to let a cohort veto below 30; a paper sleeve should not get a weaker standard than a regime cohort.

Below n = 30 the only permitted statements are descriptive: "n closed trades so far, mean X%, CI crosses zero."

### Tier 2 — Promotion candidate

**All** of the following, on the same stream, simultaneously:

1. **n ≥ 30** closed trades — `alpha_summary[<stream>]["spy"]["n"] >= 30`.
2. **Bootstrap 95% CI of mean alpha vs SPY strictly above zero** — `alpha_summary[<stream>]["spy"]["ci_lo"] > 0`. A positive mean whose CI crosses zero is a lean, not an edge; that sentence is already the docstring of the function that computes it.
3. **≥ 2 distinct market regimes represented, each with ≥ 10 closed trades**, where a trade's regime is obtained by joining `trades[<stream>][i]["signal_date"]` to `run_history[date == signal_date]["regime"]`, using the repo's `bull` / `bear` / `choppy` keys. Trades whose `signal_date` has no matching `run_history` row are **excluded from the regime count** (they still count toward `n`). A sleeve that has only ever traded one regime has not been tested.
4. **No execution-config drift** during the measurement window (see [Invalidating conditions](#invalidating-conditions)).
5. **Beats the pinned live-book comparator on mean alpha vs SPY** — the sleeve's `alpha_summary[<stream>]["spy"]["mean"]` exceeds the comparator's, read from the pinned artifact (see [Comparators](#comparators--the-live-books-not-backtest-bands)). Necessary, never sufficient — a sleeve that clears condition 2 but is worse than what is already running has demonstrated an edge and not a reason to deploy it.

   **The metric is named because "expectancy" was ambiguous and the two candidates differ.** For the pinned `sniper|mas_official` book: mean alpha vs SPY is **+0.6408%**, raw `avg pnl_pct` is **+0.7490%**. Conditions 2 and 5 now read the same quantity from the same field, so one bar cannot be cleared on one metric and judged on another.

> **Conditions 2 and 5 apply different evidentiary standards, deliberately. Condition 5 is the softer one.**
>
> Condition 2 is an **evidence** test: the sleeve's own CI must exclude zero. Condition 5 is a **deployment** test: is this better than what is already running? A point-estimate comparison is legitimate for that question — but it is not evidence, and the pinned comparator is not an established edge. Every live stream in the pinned artifact carries `significant: false`; `sniper|mas_official`'s own alpha CI is **[−0.5856, +1.8895]**, 2.48 points wide and spanning zero.
>
> Two consequences, both binding:
>
> - **Condition 5 may never be upgraded into evidence.** Beating the comparator establishes nothing about whether the sleeve has an edge. It is a relative statement about two things that may both be noise.
> - **Condition 2 may never be downgraded into a comparison.** "Its CI is tighter than the live book's" is not condition 2. `ci_lo > 0` is condition 2, and nothing else satisfies it.
>
> This asymmetry was invisible until 2026-08-16. Making it visible is the fix; removing it is not, because the two conditions are answering different questions and a single standard would break one of them.

> **Condition 5 is evaluable only while an admissible pinned comparator exists.** If none exists when a stream reaches an evaluation point, condition 5 is recorded as **NOT EVALUABLE** — *not passed, not failed*. Tier 2 is **deferred, not denied**, and **the absence of a comparator is itself reported as the finding.**
>
> This clause exists so that a condition can never become permanently unsatisfiable-by-inevaluability. A criterion with only a failure mode and no path to satisfaction is the mirror image of the defect this document rejected in the launcher acceptance draft — one that no result can fail, versus one no result can pass. Both are ways of guaranteeing a no-decision, and neither is a bar.
>
> It also keeps the reporting honest in the case that matters: **"no admissible comparator existed"** and **"nothing cleared the bar"** are very different claims about the strategies, and only one of them is about the strategies at all.

> **Resumption — a comparator arriving is not an evaluation point.** When a comparator is pinned *after* a stream has already been recorded NOT EVALUABLE, the deferred determination resumes **at the next scheduled evaluation point** (n = 30 / 50 / 100) — **never on the day the comparator lands**.
>
> Otherwise comparator-arrival becomes an unscheduled look, timed by an event that has nothing to do with the stream. An extra look at a fixed 95% interval is precisely the multiple-comparisons problem the fixed evaluation points exist to prevent, and the pinning date is not a property of the evidence. The concrete failure: a stream deferred at n = 30 and now sitting at n = 41 would get a fresh look the moment someone finished pinning, rather than waiting for n = 50.
>
> A stream that is already past n = 100 when the comparator lands is evaluated **once**, on the qualifying trades as of that point, and gets no further scheduled look. One look, not one per pin.

Tier 2 makes a sleeve *eligible for a promotion discussion*, not promoted. Promotion remains Ray's decision and still goes through the validation card.

> **Consistency note.** Condition 5 was implicit until 2026-08-16: the Comparators section asserted that beating the live book is "necessary," while the Tier 2 list named only four conditions and the blocker table said pinning blocks Tier 2. Victor (Claude Code, VPS) caught the contradiction. It is resolved here by making the comparator an **explicit** Tier 2 condition rather than by weakening the blocker table — the stricter branch, which the amendment rule permits unconditionally ("raising a threshold is always allowed"). The blocker table is therefore accurate as written: **comparator pinning does gate Tier 2.**

### Tier 3 — Threshold/parameter claims: **n ≥ 100**

Inherited from `min_threshold_trades = 100` (`validation_card.py:336`). Any claim of the form "threshold X is better than Y" needs 100 closed trades on that stream. Below that, parameter differences are noise — the repeated finding of this project is that no tunable parameter improves MR (score IC ≈ 0).

## Comparators — the live books, not backtest bands

"Beats zero" is a weak question. The useful one is **"does this reproduce the live book?"** Compare against **published live performance**, never against truth-matrix or backtest bands — the backtest universe is known-wrong for sniper (large-cap cache, median ATR% 2.28 against a floor of 5), so a paper sleeve beating a backtest band tells you nothing.

### The sign conflict is RESOLVED. The −0.97% figure is unsupported.

Independently recomputed from the raw production export by Victor and reproduced exactly by Hawk from the same artifact:

| Stream | n | Mean `pnl_pct` | Mean alpha vs SPY | Alpha CI | `significant` |
|---|---|---|---|---|---|
| `sniper \| mas_official` | 60 | **+0.7490%** (53.3% WR) | **+0.6408%** | [−0.5856, +1.8895] | **false** |
| `mr \| mas_official` | 35 | +0.4039% | — | [−0.1881, +1.3334] | **false** |
| manual sleeve | 63 | +0.1847% | — | [−0.4158, +0.7623] | **false** |
| `pead_paper` | 10 | −0.1942% | — | crosses zero | false |
| `pead_neglected` | 10 | −0.6139% | — | crosses zero | false |

**The previously recorded −0.97% is not supported by this artifact and is withdrawn.** The older +0.74%/trade figure reproduces. **IBKR is `unlocatable`, not retracted** — it is a separate broker and its absence from the MAS export is the wrong source, not disconfirming evidence.

### The comparator must be pinned to a saved artifact, not a URL

`data.json` is **overwritten every run on a rolling `window_days: 90`**. A URL is not a pin: next Monday it silently drops everything before ~2026-05-18 and the comparator moves with no amendment. **Pin a saved copy with its hash.**

Reference artifact for the figures above: `generated_at 2026-08-14T20:59:45Z`, `sha256` beginning `f4de7a2e7bf566b0`. A pinned comparator record must carry: stream, metric, value, n, date range, `generated_at`, and full sha256.

### ⚠️ Which artifact — the field path does not resolve uniquely

**The mirror bundle and the production Pages export use identical stream keys.** `alpha_summary["sniper|mas_official"]["spy"]["ci_lo"]` resolves in **both** and returns **different books** — the mirror had n=3 for that key on 2026-08-15 while production had n=60.

> **Every reference in this document to `alpha_summary[...]` means the MIRROR bundle** — the paper sleeve under measurement, written by the afternoon lane into `<date>/afternoon/`. **The production Pages export is the COMPARATOR only, never the measured object.**

A reader who takes condition 2 from the production bundle evaluates the live book **against itself**, satisfying conditions 2 and 5 with identical rows. Naming the path is not enough; the artifact must be named too.

### Condition 5 compares alpha, and is the weaker of the two tests

**Which expectancy** was ambiguous and is now fixed: **condition 5 compares mean alpha vs SPY**, not raw `pnl_pct`. For the reference artifact that is **+0.6408%**, not +0.7490%. Condition 2 and condition 5 must read the same quantity or the bar mixes metrics.

> **Stated plainly: conditions 2 and 5 apply different evidentiary standards, and 5 is the weaker.** Condition 2 requires the paper sleeve's alpha CI to **exclude zero**. Condition 5 requires only beating a live book whose own CI is [−0.5856, +1.8895] with `significant: false` — a point-estimate comparison against a book that has itself established nothing.
>
> This is deliberate and is not a licence to relax condition 2. Condition 5 asks "is this better than what we already run," which is a **deployment** question and legitimately a point comparison. Condition 2 asks "is there an edge at all," which is an **evidentiary** question. Do not upgrade 5 into evidence or downgrade 2 into a comparison.

### Reproduction and beating are different tests. Do not weld them together.

**Both live books are negative.** Any rule phrased as "reproduces *or* beats the live book" therefore passes a sleeve running −0.9%/trade, which reproduces MAS-GH faithfully while losing money. With a negative comparator the two tests point in opposite directions:

| Test | Question | Where it lives |
|---|---|---|
| **Reproduction** | Does the paper book behave like the live book? | **Mirror fidelity.** Belongs to `docs/paper_mirror_acceptance.md` (launcher acceptance), not here. |
| **Beating** | Is this sleeve worth trading? | **Sleeve value.** This document. |

The rules for this document:

> **Beating the live book is NECESSARY and NEVER SUFFICIENT.** A sleeve must clear the live comparator *and* clear Tier 2 on its own terms (`ci_lo > 0` on alpha vs SPY). Clearing the comparator alone establishes only that it is less bad than what is already running.
>
> **Reproducing a negative live book is a STOP signal, not a pass.** A paper sleeve faithfully tracking a losing book is evidence that the mirror works and the sleeve does not — it confirms fidelity and disconfirms value in the same measurement.
>
> **This rule is currently hypothetical, and stays in force anyway.** As pinned, the live book is **positive** (`sniper|mas_official` +0.6408% alpha), so there is no negative book to reproduce today. The rule exists because the comparator is re-pinned over time and a book can turn negative between evaluation points — at which moment the temptation to read fidelity as success is strongest. Do not delete it for being inactive.

The decisional table below is unchanged and remains authoritative: `ci_lo > 0` on alpha vs SPY is the only quantity that decides anything. This section constrains interpretation; it does not add a decision rule.

### ⚠️ BLOCKING PREREQUISITE — comparator pinning is unassigned

The pinning requirement above sits **on the critical path to any Tier 2 read**, and as of this writing **no one owns it**. If it stays unassigned, the first stream to reach n = 30 arrives at a bar it cannot be evaluated against, and the likely response under time pressure is to quote a remembered number — which is the failure this document exists to prevent.

| Field | Value |
|---|---|
| **Owner** | **UNASSIGNED — Ray to name** |
| **Due** | Before the first stream reaches n = 30 |
| **Blocks** | Tier 2 for every stream |

**Definition of done:** the comparator is re-derived from the reconciliation artifact — not from memory, not from this document, not from a dashboard rendering — and recorded here as: metric, value, row count, date range, and the source artifact's identifier. Both live books, and the sign conflict resolved with evidence rather than by preference.

Until that exists, streams may reach n = 30 and be *described*, but **no Tier 2 determination may be made.**

**Explicitly not comparators:** the retracted 82% sniper win rate, the retracted 69.5% MR win rate, the retracted 85.7% paper WR, and `trade_pnl_pct` from `sniper_component_ic.py` (frozen V3 params, zero slippage — not expectancy, and the file says so).

## Stop condition — stated as a number, not a judgment

A stream **stops** (paper trading halted, sleeve retired or rebuilt) when **either** fires:

**S1 — Statistically established negative.** At a fixed evaluation point with **n ≥ 30**, the bootstrap 95% CI of mean alpha vs SPY is **entirely below zero** — `alpha_summary[<stream>]["spy"]["ci_hi"] < 0`. This is the symmetric mirror of Tier 2's promotion test. It cannot be argued away by "small sample" — 30 is the same floor promotion must clear.

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

## Decisional vs descriptive metrics

Both get reported. Only one decides anything.

| Metric | Role |
|---|---|
| **Mean alpha vs SPY + bootstrap CI** | **DECISIONAL.** Tier 2 and S1 read this and nothing else. |
| Win rate (% `pnl_pct > 0`) | Descriptive only. **Never decisional.** |
| Average `pnl_pct` per trade | Descriptive; the expectancy comparator against the live books. |
| Sharpe (≥30 trades, distinct entry dates) | Descriptive. |
| Max drawdown, concurrency-capped equity | Descriptive — **except** as stop condition S2, which is decisional. |

**Win rate is deliberately excluded from every decision rule in this document.** The trail/stop sweep purchased a **90% win rate at approximately zero profit**: a tighter trail buys WR and sells expectancy. A bar whose headline metric is win rate would reward exactly the change that destroys the edge. Report it; never decide on it.

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
4. **Raising a *promotion* threshold is always allowed.** Tier 1, Tier 2, Tier 3, the dispersion floor: raising these makes promotion harder, so no justification is needed.
5. **Raising a *stop* threshold is a loosening, and takes rule 3's test.** S1's sample floor and S2's drawdown breach are the thresholds a sleeve must cross to be **stopped**. Raising one keeps a losing sleeve alive longer, which is the same act as lowering a promotion bar wearing the opposite sign. So it requires a reason that is **not** "the current drawdown would breach the lower one."

   > Rules 3 and 4 were written as if every threshold pointed the same way. They do not. "Raise = conservative" holds for a bar you must clear to *proceed*, and inverts for a bar you must cross to *halt* — moving S2 from 20% to 40% mid-drawdown would be permitted by rule 4 as written, and it is exactly the abuse rule 3 exists to stop.

6. **Direction is judged by effect, not by arithmetic.** If an amendment makes it easier for a sleeve to keep running or to be promoted, it is a loosening and takes rule 3's test, whatever happens to the number.
