# Forward-return decay on LIVE picks — sniper vs MR (2026-08-13)

**Method.** Every live pick in the published dashboard `data.json` (90d window),
measured buy-and-hold from the ACTUAL live entry (T+1 open) out to 5/7/21/42/63
trading bars, vs SPY over the identical window. Polygon strict, no fallback,
provenance stamped (288/288 tickers on the pooled run). Sleeves kept SEPARATE per
CLAUDE.md. Scripts: `scripts/sniper_forward_returns.py`, scratch
`mr_and_selection.py`. Data: `sniper_live_picks_forward_returns_2026-08-13.csv`.

## Headline: the two models want OPPOSITE exits

Median forward return (medians, not means — DELL +107% / ARM +108% / ZETA +58%
distort every mean in this dataset):

| horizon | sniper (n=64) | MR official (n=35) | MR sleeve (n=66) |
|---|---|---|---|
| realized as traded (~1d) | **+0.31%** | +0.24% | +0.09% |
| 5 bars (~7 cal days) | −1.41% | **+1.00%** | **+0.80%** |
| 7 bars | −2.41% | +1.31% | +0.55% |
| 21 bars (~1 month) | −0.70% | **+6.06%** | **+3.05%** |
| 42 bars (~2 months) | −6.95% | (n=3) | −1.22% |
| ~63 bars (~3 months) | −19.90% (n=12) | — | — |

Paired per-pick delta of holding 5 bars instead of taking the live exit:

| stream | mean | 95% CI (bootstrap) | median | hit rate | split-half |
|---|---|---|---|---|---|
| MR official | +1.11pp | [−0.77, +3.14] | +1.00pp | 58% | +0.78 / +1.42 |
| MR sleeve | +0.63pp | [−0.58, +1.84] | +0.98pp | 58% | +0.72 / +0.55 |
| sniper | −0.42pp | [−2.80, +2.08] | −0.77pp | 48% | +1.69 / **−2.54** |

## What survives stress testing — and what does not

- **SURVIVES (weakly): MR gains from holding ~7 calendar days.** Two INDEPENDENT
  sleeves, same direction, same ~+1pp median, 58% hit rate both, and positive in
  **4 of 4 half-samples**. Alpha vs SPY stays positive (+1.15% / +0.69% @5b), so
  it is not pure beta. CI still crosses zero — a hypothesis, not a shippable change.
- **DOES NOT SURVIVE: anything at 21 bars.** MR sleeve halves −1.45 / +4.55.
  Sign-unstable.
- **DOES NOT SURVIVE: "sniper's exit is optimal."** Halves +1.69 / −2.54 at 5b.
  The honest claim is only that holding sniper longer is **not** better — not that
  the exit is tuned right.
- **HOLDS: sniper decays hard at 2-3 months** (−6.95% / −19.90% median, SPY
  flat-to-up, 45%/83% of picks drawing >20%). But that cohort is ONE market
  episode (May 15 – Jun 8 entries), so treat magnitude, not precision.

## Why prior MR exit rejections could not have seen this

[[research-trail-and-stop-sweep]] rejected MR stop/trail changes twice — on the
**backtest signal population**, which that same note says has **no edge at live
selectivity** ("live +0.38-0.51%/trade is the ranking + correlation filter + gate
funnel, which the backtest does not model"). A sweep over an edgeless population
is structurally blind to how the exit interacts with the LIVE-SELECTED subset.
This analysis is the first measurement of the exit on the population that is
actually traded. It does not overturn those rejections; it asks a question they
could not reach.

## Selection quality — the instrument is broken

`candidates` (526 rows, score + rank + picked) looked like it enabled a selection
audit. It does not, as exported:

1. **Export `rank` is an ordering the pipeline never used.**
   `scripts/export_dashboard_data.py:223` re-sorts `ranked + correlation_dropped`
   by score and re-numbers from 1. `Candidate.strategy_rank` (`src/main.py:613`),
   the real selection order, is never exported. A correlation-dropped name can
   take export rank 1 with `picked: False`, pushing the actually-selected pick to
   rank 2+. **`scripts/rank_quality_audit.py:62-63,120` audits this fiction.**
2. **Export `picked` conflates four different causes.** It is derived from
   `Signal` + `Outcome.skip_reason` (`export_dashboard_data.py:210-217`), not
   `Candidate.selected`. `picked=False` mixes: below-quota, capacity-censored
   sniper (`src/main.py:628-643`), correlation-dropped, and fragility-blocked
   (`src/main.py:1770-1780`). The DB HAS `rejection_stage` / `rejection_reason`;
   the exporter drops them.

Consequence: my first-pass reads ("rank-1 candidates skipped", "not-picked beats
picked") were both artifacts of (1). Day-controlled, the rank effect collapses
(sniper +1.52pp @5b t=0.55; @21b median −1.27pp, 37% of days positive).

**What IS measurable and holds:** by-day score IC vs forward alpha —
sniper **+0.002 (t=+0.01, n=12 days)**, MR **+0.076 (t=+0.88, n=27 days)**.
Sniper's score orders nothing. Every live sniper score is 84.5–110.5 against a
`sniper_min_score=70` gate, so that gate never binds either.

## Two latent bugs found (neither is currently firing)

1. **`scripts/backfill_phantom_exits.py:133-140` is model-blind** — hardcodes
   `TRAIL_*_MR = 0.5/0.3` for everything non-sniper, so running it over a PEAD
   range **re-stamps PEAD closed rows with a trail PEAD does not have**, undoing
   PR #43 in the data. `scripts/cohort_replay.py:158-160,217-218` has the same
   drift while advertising "mirrors live".
2. **`build_exit_config_snapshot` (`src/main.py:117-133`) stamps the GLOBAL
   trail** into every signal's provenance, so post-#43 PEAD signals claim a
   0.5/0.3 trail they never ran under — defeating the reconciliation the snapshot
   exists for. Exactly the parameter-mismatch class CLAUDE.md warns about.

## PEAD watch item RESOLVED (positive)

"PEAD holds should go 1d → ~20d" — they did. The closed PEAD rows still show
median hold 1d / 6-7 of 10 `trail_stop`, but those are **pre-fix**: PR #43 landed
2026-08-05 and `check_open_positions` only touches `still_open=True`, so closed
rows are frozen. The 10 OPEN PEAD positions all carry `hold_days=20` with
`days_held` 5-12. Correct cohort filter is `entry_date > 2026-08-05`.

⚠️ Separately: `pead_neglected` at 5 bars is +4.30% mean / 90% win / t=+4.12 —
but **alpha vs SPY is +0.22%**. The apparent strength is market beta. Judge the
paper trial on alpha, not raw return.
