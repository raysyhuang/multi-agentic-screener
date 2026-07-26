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

## Discipline notes
- Stage-0 base-rate FIRST, then condition on the actual strategy's trades — the
  conditioning is what caught the short-signal reversal. Unconditional IC ≠ strategy
  edge.
- Nothing here touches the live pipeline. No config/signal change shipped.
