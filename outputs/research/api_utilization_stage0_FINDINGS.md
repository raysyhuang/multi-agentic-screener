# API-Utilization Alpha Sweep — Stage-0/1 Findings (2026-07-26)

**Directive:** "Polygon $199 / FMP $29 / FRED have a lot of information — utilize the
potential for stronger, more reliable alpha." Approach approved by Ray: *both,
sequenced* — audit every under-used field, rank by expected impact, user picks a
shortlist, then Stage-0/base-rate vet each before any live wiring. User picked all
four candidates below.

**Bottom line: all four candidates FAIL honest vetting. No new shippable alpha.**
Rigorously killed in one session before any production code — the same outcome the
project keeps reaching: raw signals are thin, real uncorrelated alpha is rare, and
the existing gates/selection are the alpha. Value = 4 dead ends closed cheaply +
two reusable byproducts.

## Audit: what's paid-for but unused
- **Polygon $199 (entitled, idle in live pipeline):** `get_short_volume`,
  `get_short_interest` (script-only), `get_intraday_aggs` minute bars
  (script/backtest-only), `get_options_flow` + `get_snapshot` (dead).
- **FMP $29:** `get_institutional_holders` (13F) + its scorer both dead;
  `get_key_metrics` dead; analyst grades / price-target / guidance not implemented.
- **FRED (free):** only `VIXCLS` + `T10Y2Y` used; `DGS10/DGS2/FEDFUNDS/ICSA` defined
  but dormant; HY credit spread (`BAMLH0A0HYM2`) not even in the series list.

## Candidate verdicts

### #2 Short-interest / short-volume → sniper gate — REJECTED
- **Stage-0 (unconditional, look-ahead-safe, huge N) — signal is REAL:**
  - Short-volume ratio (276,981 obs): monotonic. Low <35% = +9bp vs base at 20d;
    high ≥65% = **−34bp**. High short participation → bearish drift.
  - Days-to-cover (98,081 obs): monotonic worse with crowding — ≥8 DTC = **−325bp**
    at 20d. Crowded shorts drift DOWN (informed shorts, not squeeze fuel).
- **Stage-1 (conditioned on 1,335 sniper Run-E trades) — REVERSES:** sniper's BEST
  cohort is sv[50,60) (+0.63%/trade); dropping high-short names removes the best
  trades (dropped +0.51% vs kept +0.15%). Every avoid-filter's bootstrap 95% CI on
  the expectancy delta crosses zero; directionally it HURTS.
- **Why:** sniper is a squeeze/breakout momentum setup — elevated short interest on
  a breakout is the fuel, not a warning. The unconditional "shorted names drift
  down" does not apply to sniper's already-selected candidates. Textbook
  selection-interaction; exactly the CLAUDE.md "probe reverses at full scale" trap.
- Script: `scripts/sniper_short_credit_filter.py`. Probes:
  `scripts/short_volume_probe.py`, `scripts/days_to_cover_probe.py`.

### #1 HY credit-spread (BAMLH0A0HYM2) → regime — RIGHT SIGN, NOT SIGNIFICANT
- Sniper does better in calm/tightening credit: HY tightening +0.36% vs widening
  +0.16%; HY below 50d-MA (calm) +0.32% vs above (stress) +0.18%. Correct sign,
  consistent with the risk-off thesis.
- BUT every per-trade avoid-filter's 95% CI crosses zero (e.g. drop-HY-widening
  delta +0.092%, CI [−0.376, +0.561]). Within noise at this sample.
- **Verdict:** not a shippable per-trade alpha. Adding HY to the regime classifier
  is nearly free and directionally supported, but expect regime *color*, not
  measurable edge. Optional; do not label as alpha.

### #3 Intraday sleeve (minute bars) — REJECTED
- VWAP-momentum (long ≥+1% above session VWAP, ride to close, through the unified
  exit engine, costed): **+0.022%/trade at 5bp/side**, WR 47%. Gross edge ~12bp is
  entirely eaten by cost (0bp +0.122% → 10bp −0.078%). Cost-fragile.
- Not sub-period stable (early −0.167%, mid +0.001%, late +0.343% at N=36).
- Reversion direction is wrong-signed: the below-VWAP control loses −0.53% (names
  1% below VWAP keep falling into the close) — intraday MR has negative edge, so
  that branch is dead too without a separate run.
- Script: `scripts/intraday_vwap_backtest.py` (probe: `scripts/intraday_mr_probe.py`).

### #4 Guidance / forward-revision → PEAD sharpener — BLOCKED (data)
- FMP `analyst_estimates` is **plan_gated on the $29 Starter tier** (0 rows for
  AAPL/MSFT/NVDA; endpoint status = `plan_gated`). Forward guidance/estimate
  revisions are unavailable without a tier upgrade. The earnings endpoint gives
  current-quarter revenue (already used by PEAD E1) but not forward guidance.
- **Verdict:** untestable at the current tier. Revisit only if FMP is upgraded.

## Reusable byproducts (follow-ups, not built)
1. **Short signal is real unconditionally** (heavily-shorted names drift down,
   monotonic, huge N) — wrong home on sniper, but a candidate NEGATIVE filter for
   MEAN-REVERSION (which buys oversold longs, some heavily-shorted falling knives).
   Would generalize the static MR blacklist. Needs an MR trade set to condition on;
   not built (out of the approved sniper scope).
2. **HY-OAS regime color** — free FRED add if we ever want risk-off context in the
   dashboard/regime label, with no alpha claim attached.

## Follow-up (2026-07-26, same session): short signal on MR + HY-OAS wired

Ray: "1 test short signal, 2 use hy oas." Both done.

### Short signal conditioned on MEAN-REVERSION (27,822 live-faithful MR trades)
- **Days-to-cover IS a real (thin) MR avoid-filter.** Monotonic: dtc[0,2) +0.20% →
  [2,3) +0.04% → [3,5) −0.01% → ≥5 −0.07%. Dropping dtc≥3 lifts MR expectancy
  +0.039% → **+0.097%**, bootstrap 95% CI **[+0.005, +0.110] → clears zero**.
  Crowded-short oversold names are falling knives — the thesis holds where sniper's
  breakout-fuel logic didn't. CAVEATS: edge is thin (MR raw is ~edgeless), multiple
  thresholds tested (dtc≥5 alone does not clear), needs a full validation-card gate
  (deflated Sharpe, live-selectivity) before shipping to live MR. NOT auto-shipped.
- Short-VOLUME on MR = noise (CI crosses 0). It's short-INTEREST crowding (DTC),
  not daily short volume, that bites MR.
- Script: `scripts/gen_mr_trades.py` → `outputs/research/mr_trades_polygon.csv`,
  then `scripts/sniper_short_credit_filter.py --trades <that>`.

### HY-OAS has OPPOSITE signs across the book — so it's wired as CONTEXT, tilt OFF
- Sniper does better in calm credit (not significant). **MR does better in credit
  STRESS: HY above 50d-MA +0.211% vs calm −0.086%; dropping stress trades HURTS MR
  (CI [−0.172, −0.080], significant).** Mean-reversion thrives on volatility;
  momentum likes calm.
- Implication: a naive "HY-stress → bear tilt" would downsize MR exactly when MR is
  strongest. So HY-OAS (FRED BAMLH0A0HYM2, free) is now **computed + surfaced every
  run** (level, stress flag, 20d change) in the regime assessment, `regime_context`
  (→ dashboard/DB), and the pipeline log — used as decision context. The optional
  bear-score tilt is implemented but **default OFF** (`regime_hy_oas_enabled=False`),
  documented with this opposite-sign finding. Fail-open (no key/data → no effect).
- Files: `src/data/fred_client.py` (series + `get_hy_oas` + snapshot fields),
  `src/features/regime.py` (params + surfaced signal + gated tilt), `src/config.py`,
  `src/main.py` (both regime call sites + log), tests in `test_regime.py`.

## Follow-up 2 (2026-07-26): guidance-raise test — BLOCKED literal, PROXY rejected

Ray: "run the E1/PEAD guidance-raise test next" + "check FMP, I paid $29/mo."

### FMP tier check — $29 Starter is active and correct (no billing/key issue)
- Receipt confirms **Starter Access, $29, Jul 21–Aug 21 2026.** FMP's own error
  string names the account "Starter plan." So the paid plan is live and working.
- Entitlement map on the pipeline key: earnings actuals+revenue ✅ (164 quarters),
  ratios ✅, key-metrics ✅, insider/screener/news ✅; **analyst_estimates (forward
  guidance) ❌ plan_gated (0 rows); 13F institutional ❌ 402 Restricted.** Forward
  estimates/guidance + 13F are simply not part of Starter — they need a higher tier
  (Premium). Nothing to fix. (Probing the 13F endpoint trips the client's fatal-402
  self-disable for that process only — a probe artifact, not the pipeline state.)
- The literal guidance-raise (forward-estimate up-revision) is therefore
  **untestable on Starter** — confirmed a 3rd way (MCP `financial-estimates` = ACCESS
  DENIED; pipeline `analyst_estimates` = plan_gated; 13F = 402).

### Achievable proxy — revenue-growth ACCELERATION — REJECTED (inverts)
Revenue actuals ARE on Starter, so tested the guidance-spirit quality dimension
point-in-time: does a beat with ACCELERATING YoY revenue growth (yoy_now > yoy_prev)
drift more? (Orthogonal to the revenue-SURPRISE gate already shipped.)
`scripts/pead_revaccel_test.py`, Polygon 3Y, unified engine, 7.5bp/side:

| cohort (>10% beat) | N | avg% | Sharpe |
|---|---|---|---|
| all beats | 1574 | +1.81 | 1.47 |
| + rev accel (>0) | 901 | **+1.31** | 1.06 |
| − rev decel (≤0) | 669 | **+2.42** | 2.02 |

- **INVERTS**: decelerating-growth beats drift MORE; accelerating growers drift LESS.
  Robust at >2% too (accel +1.02 vs decel +1.64). Textbook PEAD — underreaction lives
  in the NEGLECTED/surprising beat, not the already-loved momentum grower ("priced
  for perfection" gets sold). Same lesson as E2: obvious good news is contrarian for
  drift.
- **A guidance-raise / accelerating-fundamentals filter would HURT PEAD.** Rejected.
- Byproduct: the DECELERATING-beat cohort is consistently the BETTER PEAD subset
  (+2.42%/Sharpe 2.02 at >10%) — a "neglected beat" candidate, OPPOSITE of guidance-
  raise. One split / multiple-testing-adjacent → would need a validation-card gate
  before any use. Not shipped.

## Discipline notes
- Stage-0 base-rate FIRST, then condition on the actual strategy's trades — the
  conditioning is what caught the short-signal reversal. Unconditional IC ≠ strategy
  edge.
- Nothing here touches the live pipeline. No config/signal change shipped.
