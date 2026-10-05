# PIT universe — rulings needed to unblock, with recommendations (2026-10-05)

**Status:** dataset still HALTED; Phase B not authorised. This memo asks for rulings, one decision per line, so the work can resume. It does not change the contract.

**Who rules.** Neo is the reviewer of record for the contract (`PIT_UNIVERSE_CONTRACT.md` §11). Ray can rule directly or forward this memo as-is.

**Contamination disclosure.** I have seen the August vintage's outcomes. Every recommendation below is therefore justified by a principle that would hold whatever the data said, and each one is marked **clean** (independent of the observed breaches) or **seen** (I know how it would have affected the halted vintage; weigh it accordingly). Where I cannot give a clean reason, I give no number.

## What changed since the August halt

1. **#130 (merged today) changed the live universe definition again.** Before it, the live `universe_size` included ~519 rows FMP flags `isActivelyTrading=False` (VMW, PXD, DFS, TWTR…), names that never trade. PIT only admits names that traded on D. So the Aug 12 → Oct 2 live counts, the only window clean of #63, are contaminated by the same category error as the window before #63. **The live-count gate (§A.5 last row) remains DEFERRED**, and its clean window starts with the first post-#130 run (2026-10-06).
2. **No other universe-definition change merged in between.** `#84` touched `aggregator.py`, but only for earnings-calendar health.
3. **A research consumer is now waiting.** The momentum + quality core (pre-registered alongside this memo) cannot run on survivor data. The same defect undermined the PEAD sleeve's evidence and the Reclaim replay.

## Rulings

| # | Question | Recommendation | Basis | Contam. |
|---|---|---|---|---|
| **R1** | Adopt **§3a-v2** (resolve each month-over-month type change, and each exchange change that crosses the eligible set, to its exact day by binary search on the per-ticker endpoint)? | **Adopt.** | Exchange and security type are event-driven, and a monthly snapshot cannot represent a mid-month event. The rule is strictly more accurate, never more permissive. Cost was ~1,195 calls on the August vintage. | clean (found while investigating a breach, as the halt memo disclosed; the justification doesn't reference it) |
| **R2** | The **0.5% monthly exchange-drift threshold** cannot be expressed at n≈134/month (the smallest non-zero rate is 0.75%). | **If R1 is adopted: zero tolerance on both axes.** If R1 is rejected: no recommendation; that's Neo's call. | After transition resolution, no legitimate mechanism produces a disagreement, so any disagreement is a defect. This is the same logic the contract already applies to the type axis. | clean conditional on R1 |
| **R3** | **Security type:** PIT requires `type == "CS"`, but live has no common-stock constraint. That excludes 23 liquid ADRs the book trades (PBR was picked at rank 1). | **Eligible = Polygon `CS` ∪ `ADRC`.** Keep excluding ETF/ETN/fund/preferred/warrant/unit/right. | Contract §2: membership is the *live* eligibility constraints as of D. Live (FMP `isEtf/isFund=false`) admits ADR common, so PIT must too. | seen (I know it adds 23 names), but it is the contract's own rule |
| **R4** | **Volume basis:** PIT uses D's share volume (frozen §11); live uses the FMP screener field, which differs near the floor (7 candidate-days, 1 picked). | **Keep §11 (D share volume).** Record the near-floor disagreement as a known reference difference, not a PIT defect. | §11 already ruled on semantics. The screener field is a snapshot whose timing PIT can't reproduce historically, and changing PIT to chase it would make PIT depend on FMP's timing. | clean |
| **R5** | **§A.5-v2 clean-window precondition:** (a) minimum clean post-change observations; (b) what counts as a universe-definition change. | (b) **Any merge touching `src/signals/filter.py`, the screener query in `src/data/fmp_client.py`, or `src/data/universe_selection.py`**, detected from merge history automatically. (a) **No number from me.** | (b) is structural. (a) is the median's stability requirement, and any number I give now is contaminated by having seen the −23.5%. | (b) clean, (a) withheld |
| **R6** | **Ratio gate at zero:** "monthly unknown rate > 2× trailing-12-month median" halts on any non-zero month whenever the median is 0 (the normal state for `exchange_unknown`). | **When the trailing median is 0, the absolute gate (1% / 5%) governs; the 2× rule applies only when the median is > 0.** | A ratio to zero is undefined, not maximally strict. The three real ratio breaches (2025-05, 2025-09, 2026-06) had non-zero medians and stay caught. | seen (I know which months it affects) |
| **R7** | **Range:** §11 froze 3 years (≈36 monthly observations). | **Extend to the longest range Polygon's grouped-daily and reference endpoints support, at least 7 years, if the plan allows.** Probe depth first. | Power, decided before any strategy result: at a ~2%/month tracking error, 36 months gives a 95% CI half-width of ~0.65%/month (~8%/yr), which can't detect a 3–4%/yr premium. 84+ months roughly halves that. | clean |
| **R8** | **Where the vintage lives.** The repo is **PUBLIC**, so a GitHub Release asset here would republish Polygon data. | **A release asset on a new private repo** (e.g. `raysyhuang/mas-data`); the VPS fetches it through the existing authenticated API path. The manifest (hashes only) stays force-added in this repo. | It sidesteps the Polygon redistribution question, the VPS replay path already exists, and nobody needs public reproducibility. | clean |

## Sequence once ruled

1. Version the contract with the rulings (v3) and pin them in tests.
2. Repairs already accepted in the halt memo (§5: ledger, ceiling-per-attempt, 5xx retry, per-month gates, three-way divergence attribution) are merged machinery. Re-verify them.
3. Fresh vintage through 2026-10, over the R7 range, with §3a-v2 transition resolution and the R3 type set. Re-run the §3b audit with a bumped `SAMPLER_VERSION`; the August audit is void.
4. Phase B: market-cap estimate (§3c), the §3d threshold audit, and delisting handling. Names stay through their final trading day (frozen). For **delisting returns**, use the PEAD method (#125): report 0 / −50 / −100% sensitivity bounds, never a single imputed number.
5. The live-count gate stays DEFERRED until the R5(a) count of post-2026-10-06 runs exists, then it's evaluated automatically.
6. Only after sign-off: the momentum + quality core runs (`docs/prereg_momentum_quality_core.md`).

## What I am NOT asking for

No change to the §3b type-axis zero tolerance, to the 1% / 5% absolute gates, or to the membership rule. No threshold here was chosen by looking at which months passed.
