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
| 4 | Missing live-universe divergence check | **now implemented — see §5a** | compares PIT membership against the frozen live dashboard export, attributing each divergence three ways |
| 5 | Raw + manifest under gitignored `outputs/`; no Release, no hash verification, no clean replay | confirmed | GitHub Release archive, manifest force-added to the repo, `verify` subcommand recomputing SHA-256 against the committed manifest, documented clean-VPS replay |

### 5a. The divergence check found two real PIT-vs-live mismatches

Run against 489 live candidate-days inside the vintage (`raw/live/dashboard.json.gz`,
hashed into the manifest so the result replays from frozen bytes).

**(i) PIT applies a constraint live does not have — 39 candidate-days, 8 of them PICKED.**
`src/signals/filter.py` gates exchange, ETF/fund flags, price, volume and market cap. It
never requires common stock. PIT's `type == "CS"` therefore removes every ADR the book
actually trades:

```
ARM BP CX EQNR ERIC GGAL GSK KSPI NOK NVS PBR PKX PSO RIO
SQM TS TSM UMC VALE VIST VOD YPF ZTO          (23 distinct, all ADRC)
```

PBR and NOK were picked at rank 1 and rank 5. **A backtest on this universe would be
structurally blind to 23 liquid names the live book trades**, which is a selection
divergence in the direction that makes backtests silently unrepresentative. Whether the
constraint or live is right is a contract question — but they must agree, and today they
do not. Requesting a ruling.

**(ii) The volume constraint is evaluated on a different quantity than live — 7 candidate-days,
1 PICKED (FTS, rank 2, 2026-08-03).** All seven pass PIT's own type and exchange checks and
fail only `MIN_SHARE_VOLUME`, on that session's actual volume:

```
TFII 466,235 · HCC 417,301 · NSIT 413,740 · GSAT 472,176
LGND 291,126 · FTS 422,890 · HRI 306,483        (floor: 500,000)
```

PIT uses the session's own share volume from grouped bars; live uses the FMP screener's
volume field, which is not the same quantity for names hovering near the floor. This is a
definitional mismatch, not a data error, and it also needs a ruling.

**(iii) PIT correctly catches a known live defect — 12 candidate-days, 0 picked.**
ETF/FUND names (TQQQ, QQQ, QQQM, SMH, VONG, IUSG, PDBC, FTGC, UFO, PDI) that live's dead
ETF gate admitted until #63. Here PIT is right and live was wrong.

**A defect in my own first implementation:** it bucketed ADRs together with ETFs under
"explained by live gates", reporting a PIT over-restriction as a live defect and inverting
the conclusion. Attribution is now three-way and pinned by a test. This is the third time
in this workstream that collapsing two distinct categories into one produced a confident
wrong answer.

One item I verified rather than assumed: the live universe filter is
`src/signals/filter.py:227`, `if exchange not in ("NYSE", "NASDAQ")`, which matches Phase A's
`ALLOWED_EXCHANGES` exactly. There is no live-vs-PIT divergence on the exchange constraint itself.

## 5b. Two more gate defects, both mine, both found by re-reading the frozen text

Neo's exact-SHA review of #80 returned MODIFY on two controls. Reading §A.5 line by line to
implement them exposed a third I had also missed. All three are accepted.

**(i) The call ceiling could be walked through by retries.** The budget was checked once per
logical request, then the retry loop issued up to six more. Neo's offline reproduction —
11,999/12,000 in, repeated 503, 12,005 out — is exact. The check now runs before every
outbound attempt. A regression test starts at `ceiling - 1` against a permanently-503 client
and asserts on requests actually SENT, not on the ledger count, since the contract governs
outbound calls. Verified by reintroducing the defect and watching the test fail.

**(ii) The trailing-12-month gate was not the frozen rule.** §A.5 requires, for **both**
metrics, `monthly rate > 2x the trailing-12-month MEDIAN for that metric`. I implemented a
pooled trailing *type* rate compared against a fixed 1%, and omitted exchange entirely — wrong
statistic, wrong comparison, wrong scope. Now literal, with the window strictly prior to the
month under test so a bad month cannot raise its own baseline.

**Three months breach it that my version could not see:** 2025-05 (0.685% vs 2x0.223%),
2025-09 (0.542% vs 2x0.262%), 2026-06 (0.772% vs 2x0.374%). All are well under the 1% absolute
gate, so only the relative rule catches them. The gate I got wrong was concealing three real
breaches.

*Flagged, not silently softened:* where the trailing median is 0 — the normal state for
`exchange_unknown` — any non-zero month exceeds 2x0 and halts. That may be stricter than
intended. Implemented as written; softening it would be a second unilateral reinterpretation.
Requesting a ruling.

**(iii) The §A.5 live-count divergence gate was absent entirely.** Neo did not flag this; the
contract does. It requires PIT daily eligible count vs contemporaneous live count, median over
the overlap, to stay within 15%. I had built a per-candidate membership check instead — useful,
and the source of the ADR and volume findings, but a different question and not the frozen gate.

**It fails, and it is the hardest failure in the vintage:**

```
overlap dates                 60
PIT  eligible  min/med/max    1,818 / 2,013 / 2,660
LIVE universe  min/med/max    1,441 / 2,653 / 2,953
median SIGNED divergence      -23.5%      (threshold: 15% absolute)
PIT smaller than live on      59 of 60 dates
```

PIT is systematically ~23.5% smaller than the live universe, not noisily different — 59 of 60
dates point the same way. The ADR exclusion (23 tickers) and the volume-basis mismatch account
for part of it and are already before you for ruling; the residual is unexplained and I am not
going to speculate about it in the same document that reports it.

Worth separate note: the **live** universe count itself swings 1,441-2,953, a 2.05x range,
against PIT's 1.46x. Whatever explains that instability is a live-pipeline question, not a PIT
one, but it makes "the contemporaneous live count" a noisy reference to be gated against.

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


---

# 8. CORRECTION — the -23.5% divergence measured a LIVE defect, not a PIT one

Added 2026-08-13, after #80 merged. **This retracts the interpretation in §5b(iii).** The
measurement is unchanged; the conclusion drawn from it was wrong.

## What I reported

"PIT is systematically ~23.5% smaller than the live universe, 59 of 60 dates, and a ~10-point
residual survives both candidate causes — there is a third cause not yet identified."

## What is actually true

The third cause is that **the live reference was itself defective over almost the entire
comparison window.** `#63` — "Actually exclude ETFs from the universe" — merged
**2026-08-11T11:25Z**, two days before this analysis. The dashboard `run_history` spans ~65 days,
so for 59 of the 60 overlapping dates, live's `universe_size` counted ETFs, ETNs, FUNDs and REITs
that the gate was supposed to remove and did not.

PIT's own funnel puts that at roughly **626 ETF/day** passing price and volume, before REITs.

Confirmation, from the frozen live snapshot — ETF/FUND names appearing as *ranked live candidates*:

```
total                       14
date range                  2026-06-08 .. 2026-08-10
before #63 merged           14
on/after #63 merged          0        <- last one is TQQQ on 2026-08-10
```

And the divergence, split at the merge date:

```
                                        n     median signed divergence
BEFORE (live counted ETFs)             59              -23.6%
AFTER  (ETF gate actually live)         1              +35.3%

2026-08-10   PIT 2,018   live 2,606    -22.6%
2026-08-11   PIT 1,949   live 1,441    +35.3%   <- gate active; live falls 45%
```

The sign inverts the day the gate starts working. PIT did not change; the reference did.

## Consequences

1. **§A.5's live-count divergence gate cannot presently be evaluated.** Its reference changed
   definition mid-window. The pre-2026-08-11 dates are contaminated by a known, now-fixed defect,
   and exactly one clean date exists. A gate whose baseline is a moving definition certifies
   nothing, in either direction.
2. **Acceptance now has a waiting period.** The gate needs roughly 30+ trading days of post-#63
   live observations before its median is meaningful — approximately late September 2026. That is
   a scheduling constraint on Phase A acceptance, not a code change, and it should be stated in the
   contract rather than discovered later.
3. **The post-fix direction is the expected one and is not alarming.** PIT being ~35% larger is
   consistent with PIT lacking two constraints live applies: market cap >= $300M
   (`fmp_client.py:275`) and the REIT/suffix exclusions. Phase B adds market cap, which will move
   PIT toward live rather than away.
4. **The ADR and volume-basis findings survive intact.** Both were measured against PIT's *own*
   labels and against individual live candidates, never against the contaminated count, so neither
   depends on the retracted interpretation. They still need rulings.

## The error, and the rule it implies

I gated PIT against the live universe without checking whether the live universe was healthy over
the window I was gating against. The defect was **already recorded in my own project memory** —
TQQQ reaching the official candidate pool at 97.5 is the reason #63 exists — and I still did not
connect it when I chose live as a reference.

This is the same family as "never let the artifact under audit define the audit population", one
step outward:

> **A reference is not a baseline until it has been audited over the comparison window.**
> Especially a reference you know has recently been repaired: the fix date partitions the data,
> and comparing across that boundary measures the repair, not the subject.

That makes four instances in this workstream of the same underlying failure — reporting a confident
conclusion from a comparison whose two sides were not the same kind of thing.
