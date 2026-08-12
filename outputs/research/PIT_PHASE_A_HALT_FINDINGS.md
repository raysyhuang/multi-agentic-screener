# Phase A — HALT accepted, root cause of the exchange failure, and a proposed versioned amendment

Status: **Phase A dataset acceptance HALTED. Phase B NOT authorised. Research consumption BLOCKED.**
Branch `feat/pit-universe-phase-a` @ `ce30895`, unmerged, no PR. Nothing below is accepted evidence.

Author: Claude Code. Reviewer of record: Neo. This document requests a ruling; it does not assert one.

---

## 0. Withdrawn: my proposed threshold resolution

I reported two months breaching the §3d/§3b 0.5% monthly exchange-drift threshold and proposed
three ways to resolve it — pooling, raising the threshold, or narrowing it to membership flips.
**All three are withdrawn.** I proposed them after seeing which cases breached, which is post-hoc
redefinition of an acceptance criterion. It is the same error I have flagged in strategy work
repeatedly this month (see the `time_stop` and MR-stop rejections), committed in the audit layer,
where it does more damage than it would in a backtest: a strategy result that is wrong gets caught
by the next gate, but a corrupted acceptance rule disables the gates themselves.

Neo's objection stands without qualification. The arithmetic observation underneath it is still
true and still needs a ruling (§4 below), but it must be ruled on for reasons independent of which
cases breached.

---

## 1. Root cause of both exchange disagreements: neither is a labelling error

The audit flagged two disagreements between the forward-held monthly label and the date-specific
per-ticker record:

| ticker | audit date | held | actual | type |
|---|---|---|---|---|
| LNG | 2024-02-22 | AMEX (XASE) | NYSE (XNYS) | CS both |
| JCPB | 2026-04-17 | NASDAQ (BATS) | NYSE (ARCX) | ETF both |

I initially reported LNG as a false exclusion caused by a bad monthly label. **That diagnosis was
wrong.** Probing the per-ticker endpoint across February 2024:

```
LNG  2024-01-31 XASE   2024-02-01 XASE   2024-02-05 XNYS   2024-02-08 XNYS
     2024-02-12 XNYS   2024-02-15 XNYS   2024-02-22 XNYS   2024-03-01 XNYS
JCPB 2026-04-01 BATS   2026-04-17 ARCX   2026-05-01 ARCX
```

And across the monthly snapshots, Polygon's record is internally consistent — XASE for the seven
snapshots 2023-08 … 2024-02, XNYS from 2024-03 onward.

**LNG genuinely transferred from NYSE American to NYSE between 2024-02-01 and 2024-02-05.** §3a takes
the snapshot on the first session of the month, so the held label was taken on 2024-02-01 — a date on
which LNG really was on AMEX. The held label was correct as of its own as-of date. JCPB is the same
shape: a real venue move inside the month.

So there is no vendor inconsistency, no stale snapshot, and no labelling defect. The frozen rule
performed exactly to specification.

## 2. Why that does not make it benign

A correct rule produced a wrong dataset. LNG was excluded from `eligible_pre_mcap` for roughly 17
sessions of February 2024 while it was, in fact, NYSE-listed common stock meeting every constraint.

The defect is structural, not statistical: **exchange and security type are event-driven attributes.**
A venue transfer or a reclassification happens on a specific date. A monthly snapshot cannot represent
a mid-month event, so forward-held monthly cadence carries a guaranteed error window of up to one month
on every such event, for every affected ticker. The audit did not discover a rare accident; it sampled
a mechanism that fires on every transition.

This is what makes "relax the threshold" the wrong response, independent of Neo's methodological
objection: the rate is not noise to be tolerated at some level, it is a systematic error whose size is
set by the cadence.

## 3. Proposed amendment §3a-v2 — transition-resolved classification

**Rule.** Forward-held monthly remains the base. Additionally, for any ticker whose security type, or
whose exchange in a way that crosses the eligible set, differs between two consecutive monthly
snapshots, the exact transition date is resolved by bounded binary search on the per-ticker endpoint
over the sessions of the intervening month, and the label is applied with **day** resolution.

**Cost**, measured from the 37 monthly snapshots already frozen in the vintage — no new calls required
to produce this estimate:

```
month-over-month exchange changes            206
  of which cross the eligible set             21   <- resolution required
month-over-month type changes                218   <- resolution required
binary search, ceil(log2(21 sessions)) = 5 calls each
total additional Phase A calls            ~1,195
```

**Properties.** The amendment makes the rule strictly more accurate, never more permissive. It
eliminates the staleness class on both axes rather than tolerating a measured amount of it. After it,
the §3b drift audit becomes a genuine zero-tolerance check on both axes — any disagreement is then a
real defect, because no legitimate mechanism produces one.

**Independence from the observed failures.** The justification is that these attributes are
event-driven and monthly cadence cannot represent them. That statement is true, and the ~1,195-call
cost is the same, if LNG and JCPB had never been drawn in the sample. The amendment is not fitted to
the cases that breached: it does not reference AMEX, NYSE, February 2024, or any observed ticker, and
it would have been the correct rule had the audit returned zero disagreements. I am nonetheless
flagging that I found it by investigating a breach, so the ordering is on the record for Neo to weigh.

**If adopted**, the sequence is: version the contract → re-run acquisition for the ~239 transitions →
re-run the audit under the new rule with a bumped `SAMPLER_VERSION` → present a fresh result. The
existing audit result is void under the new rule; it is not carried forward.

## 4. Separate ruling requested: the monthly threshold is not resolvable at this sample size

Independent of everything above, and true before any result was seen: §3d/§3b specify a **0.5%
per-month** exchange-drift threshold, evaluated against ~134 labelled samples per month. The smallest
non-zero rate expressible at n=134 is 1/134 = 0.75%. **No sample can land between 0% and 0.5%**, so as
written the rule is zero-tolerance while not being specified as zero-tolerance.

This is a defect in a document I wrote and it needs a ruling regardless of the §3a-v2 decision. I am
explicitly **not** proposing a resolution here, because any number I suggest now is contaminated by
having seen the data. Options exist in both directions — a larger per-month sample makes 0.5%
resolvable; declaring the axis zero-tolerance matches the type axis; pooling changes the unit of
analysis — and the choice should be Neo's, made against the ruling's purpose rather than against this
vintage's outcome.

## 5. Repairs required before re-submission — accepted in full

All four blocking findings are accepted. None are disputed.

| # | finding | status | fix |
|---|---|---|---|
| 1 | No persisted request ledger; no hard Phase A ceiling | confirmed | append-only JSONL ledger per request (endpoint, params digest, status, attempt, bytes); ceiling enforced in `_get`, aborts the run on breach |
| 2 | 5xx aborts rather than bounded retry + durable failure record | confirmed at `_get` — network errors and 429 retry, 5xx goes straight to `raise_for_status()` | bounded retry on 5xx; on exhaustion write a durable failure record and continue, so one bad shard cannot void a run |
| 3 | Pooled unknown rates hide a bad month | confirmed — `type_unknown_pct` is computed over the pooled `pre_classification` denominator across all 751 sessions | per-month and trailing-12-month gates; a synthetic 100%-unknown month must halt |
| 4 | Missing live-universe divergence check | confirmed absent | compare PIT membership against live `Candidate`/universe funnel on overlapping dates |
| 5 | Raw + manifest under gitignored `outputs/`; no Release, no hash verification, no clean replay | confirmed | GitHub Release archive, manifest force-added to the repo, `verify` subcommand recomputing SHA-256 against the committed manifest, documented clean-VPS replay |

One item I verified rather than assumed: the live universe filter is
`src/signals/filter.py:227`, `if exchange not in ("NYSE", "NASDAQ")`, which matches Phase A's
`ALLOWED_EXCHANGES` exactly. There is no live-vs-PIT divergence on the exchange constraint itself.

## 6. Provisional figures — explicitly not evidence

Recorded only so a later run can be diffed against them. Every number below is produced by the
noncompliant branch and carries no weight:

```
distinct eligible tickers (pre-mcap)   5,355
type_unknown / exchange_unknown        0.32% / 0.00%   (pooled — the defective statistic, item 3)
audit: labelled 4,957 · type_disagree 0 · exchange_disagree 2
ATR% median 4.71 · share >= 5.0: 46.25%   (last 60 sessions, diagnostic only)
Phase B projection                     64,260 calls vs 75,000 ceiling
```

The `resolvable_unknown: 2,278` count is a sampling artifact, not a population rate: the audit
allocates a third of each month's sample to the `unknown` bucket by design, while the population rate
is 0.32%. It measures that the bulk list endpoint is less complete than the per-ticker endpoint, which
is worth reporting but is not drift.

## 7. Process note

Three defects in this workstream were mine and all three shared one shape — a comparison that treated
an **absent** value as a **differing** value, or a rate computed over the wrong denominator:

1. the first audit run counted 2,278 absent labels as disagreements and halted all 37 months;
2. `type_unknown_pct` pools across the whole vintage, which is item 3 above;
3. the exchange threshold is finer than the sample can resolve, which is §4.

The audit caught the first because two independent counters moved identically every month. Neo caught
the second and third. The pattern to watch on the next submission is not "unknown vs known" specifically
but **any statistic whose denominator I chose without stating what it makes undetectable.**
