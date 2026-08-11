# PIT universe cache — dataset contract (DRAFT v2, for Neo to freeze)

**Status:** draft. Not implemented. Freeze this before any code is written.
**Deliverable:** a reproducible dataset plus validation evidence. **No performance numbers.**

## 0. Timezone discipline (added after getting it wrong here)

**Every date in this contract and in the built dataset is US/Eastern market date, stated explicitly.**

Draft v1 dated the Polygon probe `2026-08-12`. The probe ran at approximately **2026-08-11 16:45 UTC / 12:45 ET**; `2026-08-12` was the author's local Shanghai date. A future-dated verification claim in a data contract is bad enough — but the underlying error is the one this dataset exists to avoid: an as-of date is meaningless without a timezone, and an off-by-one boundary silently shifts membership by a day.

Therefore:

- as-of dates are **US/Eastern market dates**;
- the boundary is the ET session close, not a UTC midnight;
- the manifest records the timezone explicitly, never a naive date;
- a fixture asserts a UTC-evening timestamp maps to the correct ET market date.

## 1. Why

The current 3Y cache is ~1,000 large-caps with median ATR% **2.28%**, while sniper requires **ATR% ≥ 5**. Roughly half the tickers can never signal and only 3.8% of bars qualify, producing ~7× fewer trades than live. Every absolute sniper claim — "weak", "improved", "the cap matters" — is bounded until this is replaced.

## 2. Membership rule

> For each as-of ET market date, construct the universe from the **live eligibility constraints as evaluated at that date**. **Do not condition membership on ATR, strategy eligibility, future volatility, future returns, or later membership.**

Guards against replacing survivorship bias with selection bias: a universe selected to be ATR-rich conditions the sample on a characteristic correlated with sniper's own trigger.

**Grouped daily bars are evidence, not the definition.** Having traded on date D is a *necessary* condition — it establishes the ticker existed and was tradeable, survivorship-free. Every live constraint is then **independently evaluated as of D**. Trading activity must never be allowed to stand in for eligibility.

## 3. Historical mapping of each live constraint

The live filter (`fmp_client.get_stock_screener` + `filter_universe`) must be given an exact as-of definition. Proposals for Neo to rule on:

| Live constraint | As-of definition | Unavailable ⇒ |
|---|---|---|
| **price > $5** | official close for D from grouped daily aggregates (`c`). Split-adjusted as Polygon serves it. Decision-time availability: the pipeline picks on D and enters T+1, so D's close is known at decision time — no lookahead. | exclude, count as `no_price` |
| **volume > 500K** | **shares**, from grouped daily (`v`) for D — matching the live screener's `volumeMoreThan`, which is share volume. Single day, not a window, unless Neo prefers a 20-day average (live uses a single snapshot). | exclude, count as `no_volume` |
| **market cap > $300M** | `/v3/reference/tickers/{t}?date=D` → `market_cap`. Verified genuinely as-of. Where absent, derive from `weighted_shares_outstanding × close(D)`. | **exclude and count as `mcap_unknown`** — never impute from a later value |
| **NYSE / NASDAQ** | `primary_exchange` from the same as-of reference call, mapped via the existing `_EXCHANGE_MAP` | exclude, count as `exchange_unknown` |
| **common stock, not ETF/fund** | `type == "CS"` from the as-of reference call | **exclude and count as `type_unknown`** |

**No current label may be backfilled onto a historical date.** If as-of classification is unavailable for a ticker on D, it is excluded and counted — not resolved with today's value. A company that converted structure, or a ticker reused after delisting, would otherwise be silently misclassified for its entire history. The counts of `*_unknown` exclusions are a headline diagnostic: a large or trending unknown rate invalidates the dataset.

## 4. Data sources — probed 2026-08-11 UTC

| Capability | Endpoint | Result |
|---|---|---|
| As-of membership | `/v3/reference/tickers?date=` | ✅ date-scoped results |
| Delisted names visible | same, `active=false` | ✅ returned AABA, AAC, AACQ |
| **As-of market cap** | `/v3/reference/tickers/{t}?date=` | ✅ AAPL 2024-06-03 $2.975T / 15.334B sh vs 2026-08-01 $4.537T / 14.687B sh — point-in-time, not a current snapshot |
| Survivorship-free bars | `/v2/aggs/grouped/.../{date}` | ✅ everything that traded that day |

`active=false` behaved ambiguously (a 2024 as-of query returned 2019 delistings) and is **not** relied upon.

**Live defect found while probing:** `PolygonClient.get_all_tickers` hardcodes `active="true"` (`polygon_client.py:256`), so the Polygon universe fallback is survivorship-filtered today. Separate ticket.

## 5. Reproducibility — vintages, not identity claims

Re-querying a vendor cannot be expected to reproduce bytes: Polygon restates, and retrieval timestamps differ. So the contract is **replay determinism**, not build determinism:

1. The first build persists **immutable raw-response snapshots** and their content hashes.
2. The manifest records raw-input hashes, normalization version, config hash, code SHA, and the ET date range.
3. **A replay from the same frozen raw snapshot must reproduce identical output hashes.** This is the reproducibility that gets asserted, including from the VPS.
4. A fresh vendor retrieval is a **new input vintage**. It must report a diff against the prior vintage — tickers added/removed, classification changes, market-cap revisions — and may never silently claim identity with it.

## 6. Remaining rulings needed

1. Date range and frequency (proposal: 3Y daily ET, matching the existing cache for comparability)
2. Corporate actions (proposal: keep split adjustment, record `weighted_shares_outstanding`, exclude dividend adjustment — live does not use total-return prices)
3. Symbol aliasing (`BRK.B` vs `BRK-B`; store Polygon-native plus an alias map, never silently drop)
4. Minimum history (proposal: ≥200 prior bars **as of D**, never using bars after D)
5. Delisting policy (proposal: present through final trading day, no forward-fill)
6. Volume definition: single-day share volume (mirrors live) vs 20-day average

## 7. Diagnostic report (compare, never target)

- PIT universe count vs contemporaneous live eligible-universe count
- ATR% distribution vs live observed — quantiles
- market-cap / dollar-volume / sector distribution
- exclusion counts by reason, including every `*_unknown`
- missing-history, alias-collision, delisting counts
- vintage diff, when rebuilt

## 8. Adversarial PIT fixtures (acceptance gate)

Must **fail** if the loader leaks the future:

1. A ticker eligible on D+1 but not D is absent from D.
2. A ticker delisted at D+5 is **present** at D — survivorship.
3. Market cap crossing $300M at D+3 does not qualify the name at D.
4. Bars after D are unreachable when constructing D.
5. An alias introduced later does not retro-resolve at D.
6. A classification known only today does not backfill onto D.
7. A UTC-evening timestamp maps to the correct **ET** market date.

## 9. Acceptance criteria

- No lookahead or survivorship leak, demonstrated against the fixtures above
- As-of classification correct at construction, with `*_unknown` counted and reported
- **ATR distribution is reported and compared with contemporaneous live-eligible observations. No required ATR target, minimum share, or membership selection criterion exists.**
- Replay from the frozen raw snapshot reproduces identical output hashes, including from the clean VPS path
- **Neo and Hawk jointly sign off before any strategy research consumes it**

## 10. Explicit non-goals

No strategy change. No parameter tuning. No performance claim. No backtest result in the delivery. The output is data plus evidence that the data is sound.
