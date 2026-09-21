# MAS strategy reset — 2026-09-21

> **This work ships as two pull requests.** After four review rounds and
> thirteen defects, the defects had stopped being spread evenly: the book
> retirement and the plumbing were stable from round 1 onward, while every
> defect in rounds 2–4 was in the measurement layer (the alpha CI, the cluster
> rules, the window), two of them introduced while fixing the previous round.
> A change set that keeps producing defects in one region is too entangled to
> land in one piece, so this PR carries the retirement and the plumbing, and
> the measurement layer follows separately on top of it. The review record
> below covers both halves; sections describing `_alpha_summary`, entry-date
> clusters, `decision_eligible` or the measurement window belong to the
> second PR.

## Decision

Keep the pipeline and its audit/governance machinery. Retire the current alpha
book to shadow measurement. Do not build a 31-strategy vote, and do not add an
allocator until at least two sleeves independently establish positive alpha.

This is a measurement reset, not a capital de-risking event: `trading_mode` is
PAPER and `main` has no broker executor. The official book is intentionally
empty. Sniper records as `sniper_shadow`; mean reversion records as `mr_shadow`;
PEAD remains quarantined paper. A synthetic end-to-end test exercises the
dormant official route on every integration run so empty supply cannot let that
path rot.

MR moved because the full population is edgeless and the rank score's measured
information content is approximately zero. The latest short losing streak was
not used as evidence.

## What changes now

- Alpha confidence intervals resample whole entry-date clusters, not iid trades.
- Production validation cards use registered model-family variant-count floors:
  MR 162, sniper 19, PEAD 12. Unknown models receive a nontrivial floor of 2.
- The retired MR and PEAD expectation bands are null. They remain visible as
  labels but cannot draw a chart band or trigger a drift comparison.
- Every admitted PEAD paper entry can create a paired `pead_60d_shadow` row.
  It uses the same T+1 fill and costs, then holds exactly 60 observed sessions
  with no stop, target, trail, partial exit, alert, cooldown or slot use.
- The constant 10 bp slippage haircut remains explicitly a lower-bound check.
  It is not promoted into a realistic stress test; that requires a full exit
  engine replay at stressed costs.

## Research queue

Only two active evidence questions are allowed at once.

1. **PEAD 60-session paired forward observation** — ACTIVE. This measures the
   September lead without changing or duplicating the primary PEAD selection.
   The 20- and 60-session rows are paired observations, not independent trades.
2. **Sector-neutral residual reversal** — BLOCKED-DATA at G0. The repository has
   the wide price panel but no admissible historical sector classification.
   Replaying today's sector labels backward would introduce look-ahead. Start
   only after a dated sector-membership source exists; do not substitute current
   profiles. Before running, compute the minimum detectable 3/5-session excess
   return on the available history. If the detectable effect is economically
   too large, close the experiment as underpowered without a sweep.

No second strategy is added merely to fill the research quota.

## Checkpoints

### By 2026-10-21 (30 days)

- All measurement fixes above remain green in CI.
- The synthetic official-pick smoke test has run against a real migrated DB.
- PEAD-60 pairing is visible in the dashboard and never changes primary PEAD
  counts, slots, alerts or official results.
- The clustered-CI amendment is recorded before any n=30 decision.

### By 2026-11-20 (60 days)

- Report the PEAD-60 supply count and pairing integrity; performance remains
  descriptive unless the pre-registered paper criteria permit a read.
- Either obtain admissible sector history and issue one residual-reversal G1
  verdict, or report BLOCKED-DATA/UNDERPOWERED. Do not replace it post-hoc with
  the best result from a new idea sweep.
- No allocator work begins unless two separate sleeves have established edge.

### On 2026-12-20 (90 days)

This checkpoint is not expected to contain a PEAD promotion verdict. At the
historical event rate, the stream is likely to have only about 15–20 resolved
candidates. Judge the checkpoint on:

- measurement fixes shipped and still correct;
- at most two pre-registered G1 verdicts or honest blocked/underpowered results;
- descriptive PEAD and PEAD-60 counts, explicitly labeled below threshold;
- whether any sleeve has independently earned the right to continue.

If nothing establishes an edge, keep the pipeline as a research instrument and
the official book empty. Do not interpret absence of evidence as a reason to
increase strategy count.

## Review round 1 (Codex, 2026-09-21) — verdict NO-GO, four defects fixed

All four findings were reproduced against the working tree before any change.

**1. The cluster bootstrap degenerates on a single entry date.** The resample
draws whole clusters, so a stream whose trades all entered on one day has one
thing to draw: every resample reproduces the same set and the interval
collapses to zero width. Three same-day losses would export `ci_hi < 0` — S1's
"statistically established negative" read off a number containing no variation.
The `n >= 30` and comparator conditions make an S1 stop at n=3 impossible
today, so the immediate stop risk was overstated; the exported statistic was
misleading regardless. `_alpha_summary` now withholds the whole summary below
three distinct entry dates, the same contract as `n < 3`. Recorded in the
acceptance criteria as a tightening under amendment rule 4.

**2. Integration tests could write to production. (The hard blocker.)** pytest
reads the developer `.env`, whose `DATABASE_URL` is a remote managed Postgres,
and the integration fixtures had no DSN guard. `pytest -m integration` on a
laptop would have run the real morning pipeline against it — overwriting
today's `DailyRun` and inserting official positions in synthetic tickers
("AAAA"). `tests/integration/db_guard.py` now requires a loopback PostgreSQL
DSN and aborts the session otherwise. CI already uses
`postgresql://postgres:postgres@localhost:5432/mas_ci`, so it is unaffected.
There is deliberately no environment opt-out: an escape hatch is set once and
inherited forever, so a remote run has to be a reviewed edit.

**3. MR-shadow alerts were asymmetric.** `format_outcome_alert` labels
`mr_shadow` closures, but no morning section rendered its entries, so positions
would appear to close that the reader never saw open. MR now has an entry
section mirroring sniper's, and a test asserts that every source labeled in
outcome alerts has an entry-section parameter — so the next retirement cannot
reinstate the asymmetry by wiring only one half. `pead_60d_shadow` is exempt by
construction: it is suppressed from outcome alerts too.

**4. Pipeline and dashboard settings could disagree.** `MEAN_REVERSION_IN_BOOK`
reached the pipeline job but not `publish-dashboard`, which builds
`BOOK_STREAMS` from the same setting. Setting the repository variable would
have traded MR officially while the published dashboard reported an empty book.
The variable is removed; book composition is a reviewed change to
`src/config.py`.

**Also fixed (Claude's own finding, ranked P2 by review):** the dashboard
counted each PEAD idea twice under "Open positions", because the 60-session
paired observation — which explicitly represents no capital — was counted as a
position. It is excluded from the count and shown separately in the tile
subtitle, while remaining visible in the position list.

Everything else in the change set was verified as correct: PEAD pairing,
fixed-horizon expiry, slot and cooldown isolation, variant floors, null
baselines and empty-book composition.

## Review round 2 (Codex, 2026-09-21) — two further defects fixed

**5. Three clusters was enough to compute an interval and not enough to decide
on one.** The round-1 fix removed the zero-width case but left the underlying
problem: `n >= 30` with no cluster requirement means 30 trades booked on 3
entry dates clear Tier 2 on 3 market observations. With k clusters all of one
sign every resample reproduces that sign, so under a null where each cluster is
positive with probability ½ an "entirely above zero" interval arrives by sign
alone with probability 2⁻ᵏ — **12.5% at k=3**, five times the nominal 2.5%, and
below the claimed tail only from k=6.

Tier 2 gains condition 1b and S1 gains its symmetric form: **≥ 10 distinct
entry dates**, exported as `decision_eligible`. Ten is the sign bound with
margin. It binds on concentration, not on supply — 30 trades over 10+ dates is
the ordinary shape of every stream here, so this does not make promotion
harder to reach, only harder to fake.

Also taken from the same finding: withholding the summary entirely below 3
clusters hid the descriptive mean and the reason. The descriptive fields now
export with `ci_lo`/`ci_hi` as `null` and `ci_unavailable` giving the reason;
`max_cluster_share` is exported so concentration is visible rather than
inferred. `null` is fail-closed in both consumers — JavaScript compares false,
Python raises.

**6. The paired 60-session row still inflated "Picks today" and the funnel.**
Round 1 excluded it from the open-position count but not from `today_picks`,
which feeds the hero line, the tile and the funnel's final stage — the same
double count the open-position fix had just rejected, in three other places.
Fixed in the exporter rather than the page, so one rule covers all three call
sites and they cannot drift apart. The row stays visible in the position list
and keeps its own stream.

Confirmed correct in round 2 and unchanged: the integration DB guard (covers
both integration modules including the ledger fixture's Alembic subprocesses,
runs before any connection, no opt-out, does not touch the unit run); the
MR-shadow wiring end to end; and the removal of `MEAN_REVERSION_IN_BOOK`.
`PEAD_ENABLED` and `PEAD_60D_SHADOW_ENABLED` were checked for the same
split-brain shape and do not have it — the dashboard export is driven by
persisted stream rows and consults neither.

## Review round 3 (Codex, 2026-09-21) — three defects fixed, one item left open

> **Superseded in part by round 4**: the "≥ 10 distinct entry dates" figure quoted in the round-2 record above is **obsolete** — it was removed in round 3. The live rule is `max(15, n/2)`, rounded UP (round 4). Do not quote the 10 from the historical record.

**7. The round-2 cluster rule contradicted a rule this document already had.**
The time-dispersion section already required `max(15, n/2)` distinct entry days
at *every tier threshold* — 15 at the first n=30 read. The 10-date condition
added in round 2 was weaker and redundant for Tier 2, and worse, it gave **S1 a
laxer standard than promotion**, on the side that retires a sleeve. The new
number is removed. Tier 2 condition 1b now points at the existing rule, and S1
is brought under it explicitly: the rule said "every tier threshold" and a stop
is not a tier, so S1 was never covered — that was the real gap, and closing it
needs no new number.

`decision_eligible` is recomputed from that same rule via
`min_decision_clusters(n)`, so the export and the document can no longer
disagree. It is descriptive: nothing enforces it, the document decides, and the
field says so. The dashboard now also withholds the green "excludes zero" badge
and the significance colour until the dispersion rule is met — an interval can
exclude zero long before enough distinct days carry it.

**Left open deliberately: there is still no concentration rule.** Sixteen
trades on one day plus fourteen singletons is fifteen dates, so it passes the
count while one day carries half the estimate. `effective_clusters` (Kish's
`1/Σw²`, worth ~3 even days in that example) and `max_cluster_share` are now
exported as diagnostics, but **no threshold on either is registered**, because
inventing one after seeing a reviewer's counterexample is how a bar gets fitted
to an argument rather than to a decision. Codex's suggestion for a first
operational read was ~20 distinct dates and ≥15 effective clusters. This has to
be settled before the first Tier-2 read, while no stream has results.

**8. Paired rows still inflated two counts outside the page.** `compute_drift`
counted them in `total_resolved`, which gates whether *any* drift alert is sent
(≥10 closed trades): the paired stream has a null baseline so it can never
raise its own alert, but it could push the total over the line and release
another stream's. The exporter's console summary likewise logged one PEAD
position plus its pair as two. Both fixed.

The underlying cause was that the same rule lived in two files. Stream
classification now has one home, `src/streams.py`, imported by the pipeline,
the drift monitor and the exporter. Three counts, three call sites, one rule
remembered in two of them was the shape of finding 6 as well.

**9. The document carried both the old and the new contract** for a summary
below three clusters ("returns `None`" four lines above the correct null-CI
description). Corrected.

Confirmed intact in round 3: the integration DB guard, the MR-shadow wiring,
the repository-variable removal, the `today_picks` filter (hero, tile, funnel,
pick list and mirror summary all covered), and the paired row's exclusion from
portfolio capital, the selection ledger, the scorecard, PEAD slots, cooldowns
and every Telegram count.

## Review round 4 (Codex, 2026-09-21) — full-branch pass, four more defects

**10. The field the document calls decisional did not hold the decisional
cohort.** The window opened 2026-09-19 and every earlier entry is OUT — but
that rule existed only in prose. The exporter selected a rolling 90 days with
no window filter, so pre-window trades were inflating `n`, the entry-date
dispersion, the Kish diagnostics and the CI in the exact field Tier 2 and S1
read. `MEASUREMENT_WINDOW_START` is now in `src/streams.py`, the exporter drops
pre-window trades for measured streams, and `pre_window_excluded` plus
`measurement_window_start` are stamped in the bundle — "excluded 4" and "had
none" are different facts. Comparator streams keep their full history.

**11. Deferring the concentration threshold did not fail closed.** Round 3
exported the diagnostics and left the threshold to Ray, which would have been
fine if nothing could be decided meanwhile. It could: the documented
counterexample (16 trades on one day + 14 singletons) produced
`entry_date_clusters=15`, `effective_clusters=3.33`, `max_cluster_share=53%`
and **`decision_eligible=true`**, with a green badge if the returns were
positive. Shipping that is the defect, not the missing number.

`decision_eligible` is now false for **every** stream while
`CONCENTRATION_THRESHOLD is None`, with `decision_blocked_reason` naming why.
The threshold is still Ray's to register — the round-4 candidate is ≥20
distinct dates and ≥15 effective clusters — but nothing can read as eligible
until it exists. The tile also shows the effective cluster count whenever it
falls meaningfully below the raw date count.

**12. `n // 2` was laxer than the documented `n / 2`.** Entry days are
integers, so "at least n/2 days" at n=31 is 16, not 15. Floor division was
quietly below the document at every odd n, on the rule the interval depends on.
Now rounds up, with odd-n tests — the previous tests only covered even n.

**13. A quarantined PEAD row could suppress an official pick.** Official
cooldown history was "every source not in `SHADOW_SOURCES`", and `pead_paper`
/ `pead_neglected` are not in that set — so a paper PEAD pick could knock out
an eligible official one, the reverse of the quarantine. Defined from what the
book **is** (`BOOK_SOURCES`) instead of from a complement. Dormant while the
book is empty; it would have mattered the moment the official route was
re-enabled, which is exactly the route the synthetic smoke test exists to keep
alive.

Also: `SNIPER_CAP_SOURCES` and the remaining `pead_60d_shadow` literals in
`src/output/performance.py` and `src/main.py` now come from `src/streams.py`
(round 3's "single definition" claim was not yet true), `total_resolved` counts
only rows that were actually measured, and the obsolete 10-date figure in the
round-2 record above is marked superseded so a search cannot resurrect it.

Confirmed intact: S1's wording, the `src.main` re-exports (same objects, no
circular import), the console-summary filter, the fixed-horizon PEAD walker,
and the paired row's exclusion from capital, slots, scorecards, validation
cards, portfolio simulation, health alerts and outcome alerts.
