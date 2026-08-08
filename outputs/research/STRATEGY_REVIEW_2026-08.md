# Strategy & Pipeline Review — 2026-08-08

Five parallel Fable 5 reviewers (universe/data, portfolio/risk, ranker/selection,
validation/measurement, out-of-the-box alpha) + synthesis. 39 findings; every
claim below carries a file:line or computed-number citation from the agents'
verified work. Framing rule inherited from the exit-layer study: **win rate is
purchasable and nearly worthless** — everything here targets expectancy per
trade, its stability, and value-per-emitted-pick.

Caveat that applies to every live number: the entire live history is 2026 —
per-year splits are impossible; per-month/per-regime splits are used instead.

---

## TIER 0 — Wrong today, will bite soon (fix-design done, PR-ready)

### 0.1 The slippage_sensitivity check is the next multi-day outage
`performance.py:664` applies slippage as a constant shift, so the check reduces
to `0.10/|window mean| < 0.5` — i.e. **it fails any stream whose 90d mean is
inside ±0.20pp and passes a stream losing −1.0%/trade** (sign-symmetric). Monte
Carlo at sniper's honest expectancy (+0.27, sd 4.92): **85% chance of ≥1 blocked
day per 6 months, E≈21 blocked days**. Also violates CLAUDE.md's own rule
(slippage changes exit paths; never post-hoc constant). Fix: gate on "expectancy
stays >0 at 1.5× cost from a cached re-run", or an absolute floor — never
ratio-to-own-mean.

### 0.2 PEAD has no concurrency cap — and holds just went from 1d to 20d
`pead_max_positions=5` caps picks/day (`main.py:1334,1350,1416`); the only
open-position control is same-ticker dedup. **13 PEAD positions open now vs the
"cap" of 5.** At ~2.4 signals/day in earnings season with 20-day holds, the
quarantined paper stream structurally becomes the book's largest exposure.
Needs sniper-style slot logic (`main.py:1387-1397`) before promotion is even
discussable — and if a cap would have excluded >30% of the paper-trial cohort,
the 30-trade promotion clock restarts (the uncapped record doesn't represent a
capped sleeve).

### 0.3 Two regimes gate the same run
`main.py:693` computes `allowed_models` from the Step-1 **preliminary** regime
(no breadth); Step 3b recomputes the regime WITH breadth (`main.py:805-817`) but
never re-gates — ranking uses the stale allow-list (`main.py:1276`) while the
governor uses the breadth-aware regime (`main.py:1859`). On a preliminary-bear /
final-choppy day, sniper is blocked by a regime the system itself overwrote.

### 0.4 Gate-blocked picks are deleted before persistence
`main.py:1548-1575` filters blocked picks out before the Signal persist at
`:1749` — both July outages left **zero record of foregone P&L**, so the gate's
false-positive cost is permanently unmeasurable. Near-free fix: persist with
`skip_reason="validation_blocked"` (every stats consumer already filters skips —
`performance.py:631`, `telegram.py:556`). Shadow-booking turns the next outage
into a measured cost.

### 0.5 The only drift monitor watches a dead table with the retired fantasy baseline
`drift_check.py:55` queries `EnginePickOutcome` (engines scaled to zero 2026-03)
against `win_rate=0.716 / Sharpe 2.47` (the retired 24,670-trade fantasy) and
needs ≥10 resolved rows from that dead table → silent nightly no-op, while
actual PEAD decay (live alpha −1.26/−1.78 vs claimed +1.8/+2.42) goes unwatched.
Repoint at live streams; natural home for the CUSUM design (§2.1).

---

## TIER 1 — Pre-registered experiments (top 3, run this session)

### E1. Sniper time_stop: parity forensic, then conditional sweep
**The biggest single per-trade lever found.** Truth-matrix B-vs-C join: the 171
time_stop-fired trades averaged −3.81% fired vs −1.22% if held (+2.59pp per
marginal trade; non-bear subset +2.70pp) → ~+0.31pp/trade portfolio-level on
sniper's +0.54 base (~60% uplift). BUT the held-side tail reaches −22.1%, and
the live time_stop cohort (10/67, avg −5.30, MFE +0.30) never went green — the
stop has protective value; naive deletion is wrong. Candidate: conditional
trigger `close < entry − 1.0×ATR` instead of `close <= entry` (engine line 236),
and/or day sweep {1,2,3,5}.
**Confound to clear FIRST (forensic):** live fires 0/67 hard stops vs
backtest-expected 5.3% (P≈0.026). Hypothesis: live stops are silently WIDENED to
1.7-2.5×ATR by the score-tier rescaling (`performance.py` tier path, sniper
`tier_atr=2×ATR`), while the backtest ran 1.5×. Replay the 10 live time_stop
trades bar-by-bar through `walk_exit` under both stop conventions; whichever
reproduces the recorded exits is the live truth.
**Kill criteria:** no variant beats C-arm Sharpe 0.75 with drawdown ≤ C-arm's on
the non-bear cohort, with sub-period signs agreeing; or the forensic shows the
tracker diverges from the engine (then fix parity first, conclusions later).

### E2. PEAD per-trade regime stamp — the promotion decision hinges on it
Live PEAD in bear tape: paper −0.77%/trade WR 22% (n=9), neglected −0.76 (n=9).
The 3Y backtest tape was 78-84% bull **every year** — the +1.8-2.2%/trade PEAD
estimate is a bull-tape number; bear is effectively unsampled. (Also settled by
the agent: regime mix is stable across backtest thirds, so the +2.11→+0.93 decay
is real alpha decay, NOT regime composition.) Rerun the PEAD backtest stamping
each trade with the run-date regime; report bear-cohort expectancy + Wilson CI.
**Decision rule:** bear cohort ≥ +0.5%/trade at n≥100 → live bear loss is
small-n noise, no gate needed. Bear ≤ 0 → PEAD gets a bear block like sniper,
and the promotion criteria change accordingly.

### E3. Universe letter-bias: does the missed-letter cohort carry signal EV?
Selection probability spans **6.7× by first letter** (C-names 15%/day vs Z-names
~100% at the live 38% cap ratio; `universe_selection.py:28-52`), with ~8%/day
churn and 26% of names intermittent (62% invisible on a given day). Churn does
NOT truncate features (history refetched fresh — `main.py:762`); the cost is
**missed trigger days**. Quantify: MR signal-day density × (1 − selection rate)
per letter cohort on the 3Y parquet → forfeited signal EV.
**Kill criterion:** missed-letter cohort contributes <5% of total signal EV →
letter bias is cosmetic, close the finding (fix stays cheap regardless: rank by
20d-avg dollar volume + hysteresis band, drop the round-robin).

---

## TIER 2 — High-value backlog (designs ready, next sessions)

- **2.1 Replace daily gate level-votes with CUSUM vs frozen baselines.** At
  n=33/68 a daily WR vote cannot do its job (SPRT: ~51 trades to separate 53%
  from 40%). CUSUM (k=0.5σ, h=4σ) run on actual live streams reproduced the
  human verdicts — MR/sniper healthy (no alarms), pead_neglected 8 alarm-days —
  with zero false alarms. Alarm ⇒ halve size/paper-mode + Telegram, never
  hard-block. Home: the dead drift monitor (§0.5).
- **2.2 Selection-quality audit on LIVE candidates — data already exists.**
  `main.py:1724-1734` has persisted every ranked top-10 `Candidate` row all
  along (~64 runs in Neon). Add a `candidates` section to
  `scripts/export_dashboard_data.py` (runs in CI with DB access) and the
  rank-1-2-vs-3-10 expectancy test runs on the next export. Backtest prior
  (agent-computed): sniper rank ordering IS real (rank1-2 +0.47%/trade vs rank3+
  +0.01, n=749/586); MR ordering carries nothing (IC −0.023). Hit rate lives here.
- **2.3 Choppy sniper ×0.6 shuts sniper out of its best regime.** Live: 0 sniper
  picks in 10 choppy runs. Run E: choppy +0.451%/trade (n=285) > bull +0.215
  (n=187). The multiplier acts as a hard switch (sniper 77×1.3 vs MR 85×0.9 in
  bull; reversed in choppy) — the bull-side MR shutout is empirically RIGHT
  (MR bull −0.38%, n=41 sleeve), the choppy-side sniper shutout contradicts the
  engine's own data. Test: replay choppy-day sniper candidates vs the MR picks
  they'd displace (MR choppy +0.97% is a high bar — may survive).
- **2.4 Bear emits picks from an edgeless stream.** Bear admits only MR
  (`regime.py:255`); live bear MR ≈ +0.02-0.20%/trade (n=32 pooled). 28% of runs
  are bear. "Emit nothing in bear" is a live alternative; test = pooled CI.
- **2.5 Correlation filter is a coin flip and its decisions are invisible.**
  88.7% of 20-obs flags unconfirmed at 120 obs; 58.9% below 0.50; window-shift
  keeps 37.6%. Plus a real bug: `main.py:1365-1368` passes the same post-filter
  count as both ranked and post-correlation — the envelope cannot record drops.
  Fix logging first (1 line), then test at 60 obs; only matters if drops change
  ranks 1-2.
- **2.6 PEAD earnings-season concentration cap** — the pead_FINDINGS caveat that
  was never actioned: mid-third capped equity −12.7% CAGR while per-trade was
  +1.01% (clustering). Sweep max-entries/week {2,3,5} × sector cap in the
  existing capped-equity sim. Pairs with §0.2.
- **2.7 Kill unconsumed paid fetches**: news sentiment (fetched ALL candidates
  every run, sole consumer disabled, est. 20-60s/run), analyst_view (~20% of FMP
  budget, zero readers), per-ticker earnings-surprise, MCP client (zero
  callers). KEEP ratio_profile (MR reads it). Before killing sentiment outright,
  one retro-IC on PEAD events (tone should moderate drift) — kill at |IC|<0.03.
- **2.8 Minimal honest risk layer for the IBKR path** (and ONLY this): 10%
  notional cap, book-wide max-10 at the order layer, sniper slots (live), PEAD
  slots (§0.2), ONE units-correct kill-switch at −8% weight-adjusted DD,
  recoverable not absorbing. Everything else in capital_guardian stays dead —
  as-is it's ~10× miscalibrated (raw-point curve: would have permanently halted
  the book 2026-07-07 and forfeited the +47.9pp that followed = 89% of profit).
  `build_trade_plan` is a booby trap (emits ~2× the sizing the reported Sharpe
  assumes) — delete or align to 10% before any consumer exists.
- **2.9 Honest-gate cleanup**: checks 1-2 validate their own call-site literals
  → re-point at persisted rows; `confidence_calibration` = WR>0.45 → replace
  with rank-IC of persisted confidence vs pnl (report-only until n≥50);
  `variants_tested` hardcoded 1 so the DSR penalty can never fire; `is_robust`
  computed, never consulted; check 8 fires only on corrupt data → relabel
  `pick_data_sanity`. Any check that still can't fail → DELETE (a check that
  can't fail manufactures false confidence).
- **2.10 Scoreboard: demote WR.** Weekly per-stream triplet already computable
  from the exporter (`export_dashboard_data.py:67-90`): expectancy net-of-SPY ±
  bootstrap CI, beat_pct, largest-single-loss share. Trail-tightening can't game
  it. Add per-month split; enforce split-beside-CI via a `mean_ci_with_split()`
  helper in `metrics.py` with a REQUIRED dates arg + exporter emits the split.
- **2.11 Smaller wired-wrong items**: dedup uses RAW score pre-multiplier
  (`ranker.py:280` — sniper can evict MR then rank out; count collisions first);
  cooldown is 5 CALENDAR days (`ranker.py:363` — suppresses MR's best re-entry
  cohort; sweep {0,2,5} trading days); the $2M liquidity floor is dead code
  (real floor = undocumented $10M single-day discard, uncounted in any funnel);
  breadth feeds regime from the churning letter-biased sample (`main.py:805`).
- **2.12 MR is a choppy-tape strategy** — live choppy +0.97 (n=16) vs bear +0.02
  (n=17); regime-tilted SIZE (not the rejected parameter tuning) is the one MR
  lever left. Thin backtest spread (2.4bp) — may die; test before believing.

## TIER 3 — Measured, keep dead / closed

- Loss-streak halts: noise triggers at WR~54% (3-streaks near-certain per 100
  trades; the post-streak entries were the BEST trades: +1.65%/trade after
  2026-07-07). Keep dead permanently below streak ~9.
- Regime-scaled sizing (0.5/0.75 exposure): empirically a dead knob — choppy was
  the best regime; counterfactual costs 7.5% of profit, no DD improvement.
- Choppy R:R degradation (stop ×0.8/target ×0.75): moot — 14/16 choppy exits
  are trail exits; the shrunken target never binds.
- Regime hysteresis: one A-B-A whipsaw in 90d; solves a non-problem (the §0.3
  consistency fix dominates).
- MR targets: decorative (3/33 hit) — like sniper's stop. Median MFE capture
  0.22-0.26 across ALL streams → dashboard health metric, not a lever.
- Overnight/intraday: MR edge = day-0 reversion + overnight gaps; later-intraday
  ≈ 0 (−0.028). MOO-exit variant worth a cheap kill-fast test someday (~+0.03-0.1pp).
- 2 picks/day: saturated for sniper (33/34 days) but no evidence pick #2 is
  worse; book runs ~32% gross, 10-cap never binds — capacity exists if §2.2
  shows ranks 3+ carry alpha (sniper yes per backtest, MR no).

---

## Bottom line

Three experiments, three verdicts, all against the reviewers' predictions —
which is the point of running them:
- **E2**: PEAD needs NO bear gate — bear is its best regime (+6.09%/trade
  E1-gated, positive every year). The live bear losses are noise.
- **E3**: the alphabetical-universe scandal is cosmetic — a same-size rank cap
  captures identical EV.
- **E1**: the time_stop "fix" (the review's biggest claimed lever) is REJECTED —
  live's fired stops saved money (−48.5 vs −61.9 held); the durable output is
  the parity fact that live sniper stops are 2.5×ATR, not 1.5×.

What actually survives as the action queue: the Tier-0 wiring fixes (slippage
check, PEAD concurrency cap, dual-regime gating, shadow-booking blocked picks,
drift-monitor repoint) — none of which is a strategy change, all of which are
the same "correct component behind wrong wiring" class every real bug this
month has belonged to. The strategy layer itself came through the review
better than expected: the edges are thin but real, the gates do their jobs,
and the two biggest "obvious wins" proposed by fresh eyes died under
pre-registered tests within hours.

Scripts: `scripts/sniper_timestop_study.py`, `scripts/pead_regime_stamp.py`,
`scripts/universe_letter_bias.py`.

## Experiment results (Phase C, this session)

### E2 — PEAD regime stamp: HYPOTHESIS REFUTED. Bear is PEAD's BEST regime. NO bear gate.
SPY-based market regime per event date (matches what live gates on), post-fix
live config (3×ATR, no trail, 10bp), scripts/`e2_pead_regime.py`:

| regime | RAW n | RAW avg | E1-gated n | E1 avg | E1 WR | E1 CI |
|---|---|---|---|---|---|---|
| bull | 889 | +1.07% | 188 | +1.72% | 56.4% | [+0.36,+3.09] |
| choppy | 390 | +1.36% | 69 | +1.17% | 53.6% | [−0.68,+3.01] |
| **bear** | **274** | **+4.91%** | **47** | **+6.09%** | **76.6%** | **[+3.34,+8.84]** |

Bear positive **every year** (2023 +6.59 / 2024 +1.82 / 2025 +5.35 / 2026 +5.12
raw). Pre-registered decision rule (bear ≥ +0.5% @ n≥100 → no gate) is met at
n=274. **Verdict: the live bear losses (n=9) are small-n noise; do NOT
bear-block PEAD — bear is where the underreaction anomaly pays most.** The
reviewer's hypothesis was reasonable and is now dead; promotion criteria do NOT
gain a bear condition.

### E3 — universe letter bias: KILLED BY ITS OWN CONTROL. Cosmetic.
`e3_letter_bias.py`: 42,962 MR trigger-days on the 3Y parquet, live cap ratio
38%. Round-robin forfeits 40.9% of total signal EV — but a straight dollar-vol
rank cap of the SAME size captures a statistically identical EV share (RR 59.1%
vs rank 58.8% of total; RR even nets slightly more EV on fewer signals). The
loss is the **cap itself**, not the letter mechanism, and with ~53 ranked
candidates for 2 picks/day, missed signal-days are surplus. **Verdict: letter
bias is cosmetic at current book size. The hygiene fix (dollar-vol rank +
hysteresis) remains cheap and reasonable but is NOT a P&L lever. The finding
that survives: any 38% cap forfeits ~40% of raw signal EV — only relevant if
the book ever needs more capacity (§Tier-3, 32% gross).**

### E1 — sniper time_stop: parity CONFIRMED (live stops are 2.5×ATR), sweep REJECTED-for-now
`e1_timestop.py`. **Forensic first: solved.** Replaying the 9 live time_stop
trades under three stop conventions: the **2.5×ATR tier stop reproduces 9/9**
recorded exits (1.5× designed: only 6/9 — HBM/IREN/KLAC would have hard-stopped
earlier). **Live sniper runs 2.5×ATR stops** (bull ×1.3 → confidence 99 → top
tier ×(1.5/0.75)), while every truth-matrix arm ran 1.5×. Fourth
live-vs-backtest parameter mismatch of the week; the 0/67-hard-stops anomaly is
explained. All future sniper backtests must use 2.5× to mirror live (or live
gets the same tier-skip treatment as PEAD — separate decision, needs the
sniper-universe rebuild first).

**Sweep (472 non-bear cohort trades, control: 416/472 per-trade match, aggregate
+0.368 vs CSV +0.357):**

| variant | 1.5× stop avg | 2.5× (live-true) avg |
|---|---|---|
| ts=1 plain (live today) | +0.368% | **+0.325%** (worst cell) |
| ts=3 plain | +0.420% | +0.402% |
| ts=5 plain | +0.437% | +0.441% |
| no time stop | +0.423% | +0.425% |
| ts=1 cond −1.0×ATR | +0.394% | +0.394% |

Direction is consistent (relaxing the 1-day stop gains +0.05-0.12pp/trade on the
cohort) BUT: (a) the reviewer's +0.31pp projection is **refuted** (~3-4×
overstated — the B−C CSV join inflated the marginal effect); (b) the 2025 split
flips sign (ts=1 +0.18 vs no-ts +0.11); (c) decisive counter-evidence: replaying
the 9 REAL live time_stop trades at live convention with no time stop → **−61.9
summed vs −48.5 fired — holding was WORSE on 7 of 9**. The time stop earned its
keep on the actual trades it fired on.
**Verdict: REJECTED for shipping.** Cohort-mean gain is small, sub-period
unstable, and contradicted by the live marginal cohort. Keep `time_stop=1`.
Revisit only after the sniper-universe rebuild produces a trustworthy absolute
cohort. The durable outputs are the parity fact (2.5× live stops) and the
refutation of the +0.31pp claim.
