# MAS strategy reset — 2026-09-21

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
