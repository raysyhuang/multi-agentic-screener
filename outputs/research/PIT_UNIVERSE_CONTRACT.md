# PIT universe cache — dataset contract (DRAFT, for Neo to freeze)

**Status:** draft. Not implemented. Freeze this before any code is written.
**Deliverable:** a reproducible dataset plus validation evidence. **No performance numbers.**

## 1. Why

The current 3Y cache is ~1,000 large-caps with median ATR% **2.28%**, while sniper requires **ATR% ≥ 5**. Roughly half the tickers can never signal and only 3.8% of bars qualify, producing ~7× fewer trades than live. Every absolute sniper claim — "weak", "improved", "the cap matters" — is bounded until this is replaced.

## 2. Membership rule (the criterion that must not drift)

> For each as-of date, construct the universe from the **live eligibility constraints available at that date**: price, liquidity, market cap, exchange, common-stock classification, and ETF/fund/leveraged exclusion.
>
> **Do not condition membership on ATR, strategy eligibility, future volatility, future returns, or later membership.** Compute and report the ATR distribution only as a post-construction diagnostic.

The failure this guards against: a universe *selected* to be ATR-rich would replace survivorship bias with selection bias — conditioning the sample on a characteristic correlated with sniper's own trigger, and handing the strategy a population pre-filtered toward its firing condition. Coverage of the eligibility region is a **number we report**, never a target we optimise toward.

Live constraints to mirror (`fmp_client.get_stock_screener`): price > $5, volume > 500K, market cap > $300M, NYSE/NASDAQ, common stock, `isEtf`/`isFund` excluded.

## 3. Data sources — verified, not assumed

Probed against the live Polygon API on 2026-08-12:

| Capability | Endpoint | Verified |
|---|---|---|
| As-of membership | `/v3/reference/tickers?date=YYYY-MM-DD` | ✅ returns date-scoped results |
| Delisted names visible | same, `active=false` | ✅ returns AABA, AAC, AACQ |
| **As-of market cap** | `/v3/reference/tickers/{t}?date=` | ✅ AAPL 2024-06-03 $2.975T / 15.334B sh vs 2026-08-01 $4.537T / 14.687B sh — genuinely point-in-time, not a current snapshot |
| Survivorship-free daily bars | `/v2/aggs/grouped/locale/us/market/stocks/{date}` | ✅ everything that traded that day |

**Design consequence:** grouped daily bars are the membership *spine* — if a ticker traded on date D it is in the candidate set for D, which is survivorship-free by construction. Reference data supplies classification and as-of market cap. This is more robust than trusting `active=false` semantics, which behaved ambiguously in probing (a 2024 as-of query returned names delisted in 2019).

**Live bug found while probing:** `PolygonClient.get_all_tickers` hardcodes `active="true"` (`polygon_client.py:256`). The Polygon universe fallback path is therefore survivorship-filtered today. Low impact (FMP is primary) but it is a real defect and should be fixed separately, not inside this ticket.

## 4. Decisions requiring Neo's ruling

1. **Date range and frequency.** Proposal: 3Y daily as-of, matching the existing cache's span so results are comparable. Cost is bounded by grouped-daily (1 call/day ≈ 750 calls) plus reference pagination.
2. **Corporate actions.** Polygon prices are split-adjusted. Proposal: keep split adjustment, record `weighted_shares_outstanding` per as-of date so market cap is reconstructable, and **exclude** dividend adjustment (the live pipeline does not use total-return prices).
3. **Symbol aliasing.** Polygon uses dot form (`BRK.B`), the normalizer uses dash (`BRK-B`). Proposal: store Polygon-native and record the alias map; never silently drop.
4. **Minimum history.** ATR(14) plus indicator warm-up needs ~200 bars. Proposal: a ticker is eligible on date D only if ≥200 prior bars exist **as of D** — never using bars after D.
5. **Delisting policy.** Proposal: a ticker remains in the universe up to and including its final trading day; no forward-fill past it.

## 5. Required diagnostic report (compare, never target)

- PIT universe count vs contemporaneous live eligible-universe count
- ATR% distribution: PIT cache vs live observed universe — quantiles, not just median
- market-cap / dollar-volume / sector distribution
- classification exclusions and unknowns (counts by reason)
- missing-history, alias-collision and delisting counts

## 6. Manifest

Every build emits an immutable manifest: source endpoints, retrieval timestamps, date range, constraint values, code SHA, config hash, per-file SHA-256, row counts. Two builds from the same SHA and range must produce identical hashes.

## 7. Adversarial PIT fixtures (acceptance gate)

Small, hand-built fixtures that **fail** if the loader leaks the future:

1. A ticker eligible on D+1 but not D must be absent from D.
2. A ticker delisted at D+5 must be **present** at D — survivorship.
3. Market cap crossing $300M at D+3 must not qualify the name at D.
4. Bars after D must be unreachable when constructing D.
5. An alias introduced later must not retro-resolve at D.

## 8. Acceptance criteria

- No lookahead or survivorship leak, demonstrated against the fixtures above
- Correct common-stock classification at construction, with unknowns counted
- ATR distribution **reported** and shown to cover the live eligibility region — as a diagnostic, not a construction target
- Identical manifest reproduced from the clean VPS path
- **Neo and Hawk jointly sign off before any strategy research consumes it**

## 9. Explicit non-goals

No strategy change. No parameter tuning. No performance claim. No backtest result in the delivery. The output is data plus evidence that the data is sound.
