# Research registry — every hypothesis, its verdict, its variant count

**Read this before proposing any strategy idea.** It is the grounding pack: what has been tested, how it died (or survived), and how many variants were tried along the way. If your idea is in the [DO-NOT-RETEST](#do-not-retest) list, it needs a **new pre-registered mechanism or feature**, not a new parameter.

Backfilled 2026-09-19 from the tracked `outputs/research/*_FINDINGS.md`, `STRATEGY_REVIEW_2026-08.md`, two branch-only findings, and in-code disable comments. Maintained by hand; `tests/test_research_registry.py` fails if a tracked findings file is not referenced here.

## How to use it

1. **Before a run**, add (or update) the hypothesis row with the mechanism and the pass/fail criteria for the gate you are about to attempt. Criteria written after the numbers exist are not criteria.
2. **Every threshold, parameter, cohort cut or horizon you try increments `variants`.** The count is what makes a deflated Sharpe honest. If you do not know how many were tried, write `unrecorded (≥N)` with the lower bound the source supports — never invent a number.
3. **After the run**, record the verdict and link the findings. A rejected idea is a result; record it with the same care.

### Funnel gates

| Gate | What | Pass line (pre-registered per idea; these are the defaults) |
|---|---|---|
| **G0** | Mechanism statement: *who is on the other side of this trade, and why do they lose to us?* | Cannot be stated → not tested |
| **G1** | Stage-0 base rate (~1 hour, data already owned): event/forward **excess return vs a same-day matched base rate**, entry-**date-cluster** bootstrap, per-year split | excess ≥ **+0.5% / 20d** AND cluster-CI lower bound **> 0** AND positive in **≥ 2 of 3** years |
| **G2** | Full-scale replay through the unified exit engine (`src/research/signal_backtest.py::simulate_trade`): `gap_through=True`, 10 bp/side, per-model trail (`Settings.trail_for_model`), **at live selectivity** | net per-trade > 0 with cluster-CI lower bound > 0; concurrency-capped equity positive |
| **G3** | Validation card `generate_validation_card(..., variants_tested=<this registry's real count for the family>)` + the 8-check gate | all 8 checks pass; DSR computed on the real trial count |
| **G4** | Quarantined paper variant (`signal_source=...`) under [`paper_sleeve_acceptance_criteria.md`](paper_sleeve_acceptance_criteria.md) | that document's Tier 1/2 at n = 30 / 50 / 100 |

Standing method rules, each learned the hard way (citations in the rows below): print the **per-year split beside any CI**; resample **entry dates, not trades**; a market-regime label for an open entry uses the **last completed session**; an unconditional signal is not a strategy edge until it is **conditioned on the live-selected population**; small probes are confirmed at **full scale**; backtest cohorts on the S&P-500 cache are **survivorship-biased and large-cap only**.

## Registry

Families: `exit` · `entry-filter` · `event-drift` · `intraday` · `regime` · `selection` · `sizing` · `universe` · `model` · `infra`.
Verdicts: `SHIPPED` · `REJECTED` · `BLOCKED-DATA` · `WATCH` · `OPEN`.

<!-- registry:start -->
| ID | Family | Mechanism / claim tested | Data | Gate reached | Verdict | Variants | Date | Source |
|---|---|---|---|---|---|---|---|---|
| R-2026-07-sniper-truth-matrix | model | The frozen sniper baseline (~82% WR) is real | S&P-500 3Y cache | G2 | REJECTED — fill-realism artifact; honest Run E = 54.3% WR / +0.54% | 5 (Runs A–E) | 2026-07-19 | `sniper_truth_matrix_FINDINGS.md`, `HANDOFF_gap_through_diagnosis.md` |
| R-2026-07-breakout | model | Daily breakout model has edge | S&P-500 2Y, 58K trades | G2 | REJECTED — Sharpe −0.00, PF 1.00; model disabled | unrecorded (≥1) | 2026-02 | `src/main.py:1248` comment |
| R-2026-07-catalyst | model | Earnings-calendar catalyst model | FMP calendar | G0 | BLOCKED-DATA — sparse `days_to_earnings`, cannot be backtested; disabled | 0 | 2026-02 | `src/main.py:1332` comment |
| R-2026-07-gap-continuation | event-drift | Gap-up ≥3% on volume continues | S&P-500 daily + minute bars | G2 | REJECTED — +13 bp/3d excess dies under stops + cost; best config is best-of-~20 | unrecorded (≥20) | 2026-07-19 | `gap_continuation_FINDINGS.md` |
| R-2026-07-intraday-mr | intraday | Intraday reversion to session VWAP | Polygon minute bars, N=2,420 | G1 | REJECTED — effect is momentum, wrong sign | 1 | 2026-07-19 | `intraday_vwap_FINDINGS.md` |
| R-2026-07-intraday-vwap-mom | intraday | Long ≥+1% above VWAP into the close | Polygon minute bars | G2 | REJECTED — +0.022%/trade at 5 bp, cost-fragile, not sub-period stable; probe was an overlap artifact | unrecorded (≥3 cost levels + control) | 2026-07-26 | `intraday_vwap_FINDINGS.md`, `api_utilization_stage0_FINDINGS.md` §#3 |
| R-2026-07-pead | event-drift | Post-earnings drift on big EPS beats (underreaction) | FMP earnings PIT + S&P-500 3Y | G4 | SHIPPED to paper (`pead_paper`); decaying; window opened 2026-09-19 | unrecorded (≥5 surprise buckets) | 2026-07-19 | `pead_FINDINGS.md` |
| R-2026-07-pead-e1 | event-drift | Only high-quality beats: revenue beat ≥2% + day-1 reaction in [+2,+12]% | same | G4 | SHIPPED (paper gate) — +2.21% gated vs +1.76% raw | unrecorded (≥3) | 2026-07-26 | `pead_trail_FINDINGS.md`, `scripts/pead_e1_test.py` docstring |
| R-2026-07-short-volume | entry-filter | FINRA short-volume ratio predicts forward return | Polygon short volume, 276,362 obs | G1 | REJECTED — ≤38 bp/20d, noise; 8-ticker smoke (+99 bp) reversed at full scale | 1 | 2026-07-19 | `short_volume_FINDINGS.md` |
| R-2026-07-days-to-cover | entry-filter | Short-interest days-to-cover squeeze factor | Polygon short interest, 95,043 obs | G1 | REJECTED — null | 1 | 2026-07-19 | `short_volume_FINDINGS.md` addendum |
| R-2026-07-short-gate-sniper | entry-filter | Avoid high-short names in sniper | 1,335 Run-E trades | G2 | REJECTED — reverses when conditioned: high short vol is breakout fuel; every CI crosses 0 | unrecorded (≥2) | 2026-07-26 | `api_utilization_stage0_FINDINGS.md` §#2 |
| R-2026-07-hy-oas-regime | regime | HY credit spread as a per-trade regime filter | FRED `BAMLH0A0HYM2` | G2 | REJECTED as alpha — right sign, every CI crosses 0; opposite signs for sniper vs MR. Wired as CONTEXT only (`regime_hy_oas_enabled=False`) | unrecorded (≥2) | 2026-07-26 | `api_utilization_stage0_FINDINGS.md` §#1 + follow-up |
| R-2026-07-guidance-pead | event-drift | Guidance raises sharpen PEAD | FMP `analyst_estimates` | G0 | BLOCKED-DATA — plan-gated on the $29 Starter tier | 0 | 2026-07-26 | `api_utilization_stage0_FINDINGS.md` §#4, follow-up 2 |
| R-2026-07-rev-accel-pead | event-drift | Accelerating revenue growth sharpens PEAD (guidance proxy) | FMP revenue actuals | G2 | REJECTED — INVERTS: decelerating growers drift more | unrecorded (≥2 thresholds) | 2026-07-26 | `api_utilization_stage0_FINDINGS.md` follow-up 2 |
| R-2026-07-pead-neglected | event-drift | Neglected beat: >10% beat + decelerating YoY revenue growth | same, N=669 | G4 | SHIPPED to paper (`pead_neglected`) — only arc candidate to clear the full card | 12 (declared search size on the card) | 2026-07-26 | `api_utilization_stage0_FINDINGS.md` follow-up 3, `scripts/pead_neglected_beat_valcard.py` |
| R-2026-07-mr-dtc-filter | entry-filter | Drop MR names with days-to-cover ≥ 3 | 27,822 MR trades | G2 (selectivity gate) | REJECTED — helps at min_score 50/60, noise at the live 70/75 | 4 (score floors) | 2026-07-26 | `api_utilization_stage0_FINDINGS.md` follow-up 4 |
| R-2026-07-insider | entry-filter | Insider cluster buying predicts drift | FMP insider, filingDate-keyed, 1,453 obs | G1 | REJECTED — 25-ticker smoke (+497 bp) fully reversed at full scale; fetch dropped | unrecorded (≥2 incl. cluster-buy) | 2026-07-27 | `api_utilization_stage0_FINDINGS.md` follow-up 5 |
| R-2026-07-options-flow | entry-filter | Put/call flow | Polygon options | G0 | BLOCKED-DATA — infeasible over REST (millions of calls); broken client method removed | 0 | 2026-07-27 | `api_utilization_stage0_FINDINGS.md` follow-up 6 |
| R-2026-08-trail-guard | exit | Enforce the trail on its arming bar (intraday ordering) | live streams + Polygon 1-min | G2 | REJECTED — mechanism confirmed, P&L refuted: buys 93% WR at lower profit | 1 | 2026-08-04 | `exit_layer_FINDINGS.md` H1 |
| R-2026-08-mr-stop-width | exit | Widen MR's 0.75×ATR stop | 3Y Polygon, 504 tickers | G2 | REJECTED twice (2026-07, 2026-08) — moves MR from slightly negative to zero at live selectivity; low-selectivity "significance" is one year | 5 (rows reported) | 2026-08-04 | `exit_layer_FINDINGS.md` H2 |
| R-2026-08-pead-trail | exit | PEAD live config matches its justifying backtest | E1-gated n=306 | G2 | SHIPPED fix — global 0.5/0.3 trail cost the whole edge (+2.21% → +0.10%); trail now per-model | 5 (configs A–E) | 2026-08-04 | `pead_trail_FINDINGS.md` |
| R-2026-08-pead-bear-gate | regime | Bear-block PEAD | SPY market regime per event | G2 | REJECTED — bear is PEAD's best regime (+6.09% E1, n=47; raw n=274 meets the pre-registered no-gate rule) | 1 | 2026-08-08 | `STRATEGY_REVIEW_2026-08.md` Phase C E2 |
| R-2026-08-letter-bias | universe | Round-robin universe cap loses signal EV vs a dollar-vol cap | 42,962 MR trigger-days | G1 | REJECTED — killed by its own control; the cost is the cap, not the mechanism | 1 | 2026-08-08 | `STRATEGY_REVIEW_2026-08.md` Phase C E3 |
| R-2026-08-sniper-timestop | exit | Relax `sniper_time_stop_days=1` | Run E + 9 live stops | G2 | REJECTED — real fired stops saved money; also established live sniper stops are 2.5×ATR | unrecorded (≥2) | 2026-08-08 | `STRATEGY_REVIEW_2026-08.md` Phase C E1, `scripts/sniper_timestop_study.py` |
| R-2026-08-rank-quality | selection | The ranker's top-2-of-N ordering carries information | Run E + 3Y MR | G1 | REJECTED — Spearman ≈ 0; ranker captures 2–7% of available selection value | 1 | 2026-08-08 | `rank_quality_FINDINGS.md` |
| R-2026-08-sniper-pick-count | selection | Take more sniper picks per day | Run E, capped equity | G2 | REJECTED — k=2 best on every axis at the live 3-slot cap; summed-P&L gain ignored capital | unrecorded (≥6 k-values × 3 caps) | 2026-08-08 | `sniper_pick_count_FINDINGS.md` |
| R-2026-08-sniper-slot-cap | sizing | Raise `sniper_max_positions` to deliver 2 picks/day | Run E cohort, 749 signals | G2 | REJECTED — supply, not slots, is the constraint | unrecorded (≥7 caps) | 2026-08-11 | branch `research/sniper-slot-sweep` (PR #68): `sniper_slot_sweep_FINDINGS.md` |
| R-2026-08-choppy-multiplier | regime | Ranker's choppy ×0.6 shuts sniper out of its best regime | Run E + 3Y MR, SPY regime | G1 | WATCH — unselected sniper beats unselected MR in choppy (CI excludes 0) but the selected-vs-selected question is unmeasured; shadow-track | 1 | 2026-08-08 | `choppy_sniper_FINDINGS.md` |
| R-2026-08-forward-decay | exit | Live picks: would holding longer beat the live exit? | dashboard bundle, Polygon strict | G1 | REJECTED for sniper (negative at every horizon); MR "+1pp at 5 bars" recorded as hypothesis → see R-2026-09-mr-hold | unrecorded (≥5 horizons) | 2026-08-13 | `FORWARD_DECAY_FINDINGS.md` |
| R-2026-08-quality-veto | entry-filter | Pre-ranking vetoes: extended tape, dilution, data sanity | live candidates | G0 (shadow-only, unmeasured) | WATCH — built shadow-only; no outcome study yet | 3 vetoes, 0 outcome variants | 2026-08-15 | branch `cursor/quality-veto-layer-329f` (PR #91): `quality_veto_FINDINGS.md` |
| R-2026-08-pead-slot-sweep | sizing | PEAD concurrency / entry-limit sweep | needs a real PEAD trade list | G0 | OPEN — harness + synthetic lock only | 0 | 2026-08-15 | branch `cursor/pead-slot-sweep-315e` (PR #90) |
| R-2026-08-atr-floor-mismatch | infra | Sniper backtests sample the production universe | code audit | — | WATCH — confirmed defect (backtest default 3.5 vs live 5.0); fix parked | 0 | 2026-08-16 | `ATR_FLOOR_MISMATCH_FINDINGS.md` |
| R-2026-08-pit-universe | universe | Point-in-time universe dataset (Phase A) | Polygon reference + grouped daily | — | BLOCKED-DATA — dataset acceptance HALTED; research consumption blocked | 0 | 2026-08-12 | `PIT_PHASE_A_HALT_FINDINGS.md`, `PIT_UNIVERSE_CONTRACT.md` |
| R-2026-09-sniper-gap-vol-live | entry-filter | Trailing overnight gap-vol flags the day-1 gap-through losers ex ante | 32 live trades + Run-E backtest (n=1,357) | G1 | REJECTED at its pre-registered read — no cutoff of 32 removes ≥5/7 losers while dropping ≤3/25 winners | 8 reported (all 32 cutoffs enumerated) | 2026-09-18 | `sniper_gap_risk_live_FINDINGS.md` (#118) |
| R-2026-09-mr-hold | exit | Holding MR ~5–7 bars beats the live exit | 49 live picks + 27,822 full-scale | G2 | REJECTED — fails out-of-sample (n=16, −0.98pp); `max_hold` rarely binds (trail exits 79% at median 1 bar) | 3 (holds 3/5/7) | 2026-09-18 | `mr_hold_and_bear_FINDINGS.md` (#119) |
| R-2026-09-mr-bear-gate | regime | Bear-block MR | full-scale, lagged SPY regime | G2 | REJECTED (NO ACTION under the pre-registered rule) — bear +0.498% but date-cluster CI [−0.66, +1.68]; 2026 is the bad tape, not bear as a class | 1 | 2026-09-18 | `mr_hold_and_bear_FINDINGS.md` (#119) |
| R-2026-09-sniper-retire | model | Sniper belongs in the official book | 90d live + Run E + forward decay | — | SHIPPED decision — retired to a shadow stream (`sniper_in_book=False`); **discretionary, S1 stop did NOT fire (ci_hi +1.51)** | 0 | 2026-09-18 | PR #117, `src/config.py` `sniper_in_book` comment |
| R-2026-08-loss-streak-halt | sizing | Halt after a loss streak | live streams | G1 | REJECTED — noise triggers at WR ~54%; post-streak entries were the best trades | unrecorded (≥1) | 2026-08-08 | `STRATEGY_REVIEW_2026-08.md` Tier 3 |
| R-2026-08-regime-sizing | sizing | Regime-scaled exposure (0.5 / 0.75) | live streams | G1 | REJECTED — dead knob; costs 7.5% of profit, no DD improvement | unrecorded (≥1) | 2026-08-08 | `STRATEGY_REVIEW_2026-08.md` Tier 3 |
| R-2026-08-choppy-rr | exit | Choppy R:R degradation (stop ×0.8 / target ×0.75) | live streams | G1 | REJECTED — moot, 14/16 choppy exits are trail exits | unrecorded (≥1) | 2026-08-08 | `STRATEGY_REVIEW_2026-08.md` Tier 3 |
| R-2026-08-regime-hysteresis | regime | Regime hysteresis to stop whipsaw | 90d run history | G1 | REJECTED — one whipsaw in 90d; solves a non-problem | unrecorded (≥1) | 2026-08-08 | `STRATEGY_REVIEW_2026-08.md` Tier 3 |
| R-2026-08-mr-targets | exit | MR targets as a lever | live streams | G1 | REJECTED — decorative (3/33 hit) | unrecorded (≥1) | 2026-08-08 | `STRATEGY_REVIEW_2026-08.md` Tier 3 |
| R-2026-08-mr-intraday-timing | intraday | MR later-intraday entry / overnight split | live streams | G1 | REJECTED — later-intraday ≈ 0; a MOO-exit variant was noted as an untested cheap kill-fast | unrecorded (≥1) | 2026-08-08 | `STRATEGY_REVIEW_2026-08.md` Tier 3 |
<!-- registry:end -->

### Operational records (not hypotheses)

- **PR #120** (2026-09-19): the paper-sleeve measurement window opened — start = first `entry_date` on or after 2026-09-19. Evidence: [`paper_sleeve_window_evidence_2026-09-19.md`](paper_sleeve_window_evidence_2026-09-19.md). Every paper `n` is 0 as of that date.
- `outputs/research/COMPARATOR_PINNED_2026-08-16.md` / `COMPARATOR_PINNING_METHOD.md`: the frozen Tier-2 comparator (+0.6408% mean alpha vs SPY).
- Dormant config flags with no recorded outcome study in this repo: `weekly_trend_gate_enabled`, `shock_killswitch_enabled`, `confirm_entry_enabled`, `partial_tp_enabled` (all `False`, `src/config.py`). They are not evidence for or against anything; an idea that needs one starts at G0.

### Known from session memory, no in-repo citation on `main`

These were run and their verdicts are recorded in agent memory, but the findings file or script is **not** on `origin/main`, so they are listed here rather than in the table. Treat as "very probably dead"; re-deriving one requires first recovering or re-running the source.

| Claim | Remembered verdict | What is citable in-repo |
|---|---|---|
| Trail-width sweep for sniper/MR (2026-07) | REJECTED — best candidate failed the card (DSR 0.00 over 19 variants) | only indirect: `exit_layer_FINDINGS.md` ("trail rejected (2026-07)"), `scripts/sniper_gap_risk.py` docstring |
| Sniper gap-vol filter on the backtest cohort (2026-07) | REJECTED — habitual gappers gap both ways | `scripts/sniper_gap_risk.py`; re-confirmed by R-2026-09-sniper-gap-vol-live |
| MR composite-score component IC (2026-07) | score IC ≈ 0 | indirect: `exit_layer_FINDINGS.md:91`, `rank_quality_FINDINGS.md` |
| E2 analyst upgrade/revision clusters (2026-07) | REJECTED at Stage 0 — clusters drift DOWN ~2pp (analysts herd at tops) | none on `main` (script lives on branch `research/trail-sweep`; cache `data/cache/grades/` is local-only) |
| PEAD sub-period decay (+2.38% → +1.05%) | decaying but positive | `scripts/pead_e1_test.py` docstring |
| Sniper hold-aware earnings blackout (structural fix) | SHIPPED | `src/main.py` sniper earnings-blackout comment |

## DO-NOT-RETEST

A retest is legitimate **only with a new pre-registered mechanism, feature or population** — never a new parameter on the same idea. "It might work with a different threshold" is how the variant count grows without anyone noticing.

| Dead idea | Why it is dead | Retest condition | Citation |
|---|---|---|---|
| Price/technical pattern models on liquid large-caps (breakout, gap-continuation, intraday VWAP momentum or reversion) | efficiently priced; edge ≤ cost | a non-price mechanism, or a genuinely different universe measured on a survivorship-free set | R-2026-07-breakout, -gap-continuation, -intraday-* |
| MR parameter tuning: stop width, hold length, targets, trail | raw MR is edgeless at live selectivity; every lever moves it from slightly negative to zero | new evidence on the **live-selected** population via an engine-faithful replay (needs `stop_loss`/`target_1` from the DB) | R-2026-08-mr-stop-width, R-2026-09-mr-hold, R-2026-08-mr-targets |
| MR / PEAD bear gates | PEAD: bear is its best regime. MR: point estimate positive, cluster CI spans zero | a new pre-registered regime definition with ≥ 100 independent entry **dates**, not trades | R-2026-08-pead-bear-gate, R-2026-09-mr-bear-gate |
| Short-volume / short-interest / days-to-cover as a gate | real unconditionally, reverses or vanishes once conditioned on the selected population | a different strategy population that has not been conditioned on yet | R-2026-07-short-*, R-2026-07-mr-dtc-filter |
| Insider cluster buys | small probe reversed at full scale | new data (e.g. role/size-filtered) with a stated mechanism, full scale first | R-2026-07-insider |
| Sniper: trailing gap-vol filter, time_stop relaxation, more picks/day, more slots | each failed its own pre-registered read | a different ex-ante feature for the day-1 gap-through losers, pre-registered | R-2026-09-sniper-gap-vol-live, R-2026-08-sniper-timestop, -pick-count, -slot-cap |
| Ranker score as a selection signal | IC ≈ 0; "a score with IC ≈ 0 cannot be tuned into one with IC > 0" | a new feature with measured IC, not a reweighting | R-2026-08-rank-quality |
| Loss-streak halts, regime-scaled sizing, regime hysteresis, choppy R:R degradation | measured, keep dead | none proposed | `STRATEGY_REVIEW_2026-08.md` Tier 3 |
| Same-bar trail enforcement | win rate is purchasable and worthless | none | R-2026-08-trail-guard |
| HY-OAS as a bear tilt | opposite signs for sniper vs MR | per-model use with a pre-registered rule | R-2026-07-hy-oas-regime |
| Analyst upgrade clusters as a long signal | inverted (memory-only record) | recover the source first | memory table above |

Blocked, not dead (would need data we do not have): guidance/forward estimates (FMP tier), options flow (flat files or months of forward collection), catalyst model (historical earnings-date coverage), a point-in-time universe (Phase A halted), buyback announcements (no source wired).

## OPEN candidates (2026-09-19 plan) — criteria registered before any run

All five share the template that the one survivor (PEAD) has: **an event, a behavioural reason for slow price adjustment, a multi-week hold.** Default G1 pass line applies to each (excess ≥ +0.5%/20d vs a same-day, same-liquidity-bucket base rate; date-cluster CI lower bound > 0; ≥ 2 of 3 years positive). `variants: 0` for all as of registration.

| ID | Hypothesis | Mechanism (who loses, why) | Data | Pre-registered controls |
|---|---|---|---|---|
| **H1** | PEAD holds, or is stronger, **outside the S&P 500** | Thinly-covered names digest earnings more slowly; the existing backtest only ever sampled 503 current S&P members (survivorship-biased, large-cap) while live PEAD fires on ≤ 1,000 names down to $300M cap | FMP earnings calendar swept by date window + a wide Polygon price set incl. delisted names; event-time liquidity filter (price ≥ $5, 20d avg dollar volume ≥ $2M, using bars strictly before the event) | report by **liquidity tercile** and **S&P-in vs S&P-out** separately; E1 and neglected-beat thresholds applied **unchanged**; report events/year (how fast a paper sleeve reaches n=30); first reproduce the published S&P numbers (+1.8%/trade raw, E1 n≈306) with the new harness — if it does not reproduce, fix the harness before testing anything |
| **H4** | A second consecutive ≥ 10% beat drifts more than a first | analysts anchor; repeated surprise is still under-extrapolated | same calendar | compare against first-beat events on the same dates; no threshold tuning |
| **H2** | MR avoid-filter: skip oversold names within N days of an EPS miss ≤ −10% | bad news is also under-reacted to → falling knives keep falling | calendar + `scripts/gen_mr_trades.py` | **conditioned on live selectivity (`min_score=75`)** — an unconditional result does not count; N registered before the run |
| **H3** | Drift after dividend initiations / large increases; drift after split ex-dates | management confidence signal absorbed slowly | Polygon `/v3/reference/dividends` (`declaration_date`, PIT-safe), `/v3/reference/splits` (execution date only) | splits: **only post-ex-date drift is testable** — announcement drift is not, and the findings must say so |
| **H5** | "Good company, mispriced after a real drawdown": growth ∧ valuation support ∧ drawdown | over-reaction to drawdowns in quality names | FMP `key-metrics` / `ratios` quarterly history with a conservative PIT lag (period end + 75 days) | **matched control = equal drawdown without the quality/valuation conditions** (otherwise this measures MR-style rebound); **must** run on the survivorship-free set — a dip-buying study on current index members is biased upward by construction |

## Known gaps

- **Nothing in production counts trials.** `src/output/performance.py:698` passes `variants_tested=1`, and `deflated_sharpe_ratio` returns 0.0 when `num_trials <= 1`, so the DSR never bites on the live cards. Research scripts must pass this registry's count by hand (see `scripts/pead_neglected_beat_valcard.py` for the pattern).
- PR #114's overfit diagnostic requires "honest complete-trial disclosure" as an input; this registry is that input.
- Several 2026-07 verdicts survive only in agent memory (table above). If one becomes load-bearing, recover or re-run its source first.
