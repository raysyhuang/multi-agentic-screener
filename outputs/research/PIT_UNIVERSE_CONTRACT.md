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
| **market cap > $300M** | **Daily estimate**, not daily retrieval — see §3c. Quarterly as-of `weighted_shares_outstanding`, held forward, × close(D). | **exclude and count as `mcap_unknown`** — never impute from a later value |
| **NYSE / NASDAQ** | `primary_exchange` from reference data, **forward-held monthly** — see §3a | exclude, count as `exchange_unknown` |
| **common stock, not ETF/fund** | `type == "CS"` from reference data, **forward-held monthly** — see §3a | **exclude and count as `type_unknown`** |

### §3a Classification cadence — forward-held monthly, not daily-exact

Draft v2 said classification was evaluated "as of each D" while Appendix A
specified monthly sampling. Those contradict, and the honest one is the cheaper
one. Frozen:

```
Phase A produces a FORWARD-HELD MONTHLY-CLASSIFICATION universe,
not a daily-exact classification universe.

Snapshot:  first ET market date of each month.
Validity:  from that snapshot date through the day before the next.
           Never applied backwards. No label ever comes from the future.
```

This is lookahead-free but it is **an approximation**, and its error mode is not
covered by the `*_unknown` thresholds: those catch *missing* labels, whereas
drift is a *known* label that changed mid-month and is therefore silently wrong
for up to 30 days. Hence the audit below, which is the price of the 20× saving.

### §3c Market cap is a daily ESTIMATE, not a daily retrieval

Same contradiction as §3a, in the constraint where it matters most. §3 said
market cap was read as-of each D via the dated ticker endpoint; Appendix A
budgeted quarterly snapshots. Those are different builds — one costs ~1.9M
calls, the other ~60,000 — and an implementer could satisfy either section while
violating the other. Frozen:

```
quarterly as-of weighted_shares_outstanding
  held forward from its snapshot date
  × daily D close
  = daily ESTIMATED market cap
```

Snapshot on the first ET market date of each quarter, applied forward only,
exactly as §3a. Never applied backwards.

**Shares × close, not the stored `market_cap`.** The endpoint returns both, but
a stored `market_cap` embeds the price on its own snapshot date and would be
stale by up to a quarter. Shares outstanding move on corporate actions and
buyback reporting; price moves daily. Holding the slow term and re-multiplying
by the fast one is the more accurate approximation, and it is the one being
approved.

### §3d Threshold audit (Phase B acceptance)

The approximation only changes an *answer* near the boundary. A name at $2B is
eligible on any shares vintage; a name at $310M may not be.

But a band defined by the estimate is selected by the quantity under test — the
same defect as sampling classification drift from the already-classified
eligible set (§3b), and the same shape as conditioning universe membership on
ATR. **If the stale-share error itself exceeds 20%, a name can sit far outside
the estimated band while its true market cap is on the other side of $300M.**
The band audit would never see it. So the audit has two parts: one measures a
rate, the other tests an assumption.

#### Part 1 — band sample (measures a rate)

```
Population:  every ticker/date pair in the month whose ESTIMATED market cap
             lies within ±20% of $300M on that exact ET market date.
             Membership is per pair, not per ticker.
Sampling:    deterministic seeded sample of up to 50 pairs per month.
             Seed and sampler version recorded in the manifest.
Small pop:   fewer than 50 qualifying pairs => audit all of them.
```

Query date-specific `market_cap` for each pair and compare the **eligibility
verdict** — above or below $300M — against the estimate.

**Halt threshold: >2% disagreement, evaluated per month, not pooled.** A short
concentrated period of share-count drift is exactly the failure worth catching,
and pooling across 36 months would dilute one bad quarter into invisibility.

#### Part 2 — out-of-band sentinel (tests an assumption)

```
Population:  ticker/date pairs whose estimated cap lies OUTSIDE the ±20% band.
Sampling:    deterministic seeded sample of 25 pairs per month,
             stratified roughly evenly above and below $300M.
```

The sentinel exists to falsify one claim: *an out-of-band estimate cannot flip
membership.* That is a binary assumption, not a rate, so it carries **zero
tolerance — any single sentinel flip halts Phase B acceptance**, whatever the
band result shows. A flip means either the band is too narrow or the quarterly
shares cadence is unsafe, and both require revision rather than a wider
threshold.

#### Consequence of any breach

A breach in **either** part explicitly:

- **blocks Phase B acceptance** — the market-cap layer is not signed off;
- **blocks research consumption and sign-off** of the dataset as a whole;
- **requires remediation of cadence, model or band before any rerun** — a rerun
  on the same parameters is not a remedy.

#### Cost

```
band sample      50/month x 36  ~ 1,800 calls
sentinel         25/month x 36  ~   900 calls
                                 ---------------
                                 ~ 2,700 calls, accounted to Phase B
```

### §3b Classification drift audit (mandatory, gates sign-off)

**Sampling population: pre-classification.** Draw the deterministic monthly
sample from names that traded on D and pass the *observable* price and volume
constraints — **before** any forward-held `type` or exchange label is applied.

Sampling from the eligible set, as draft v3 did, makes the audit inherit the
bias it exists to detect. Eligibility already depends on the classification
under test, so one direction of error is structurally invisible:

```
actual common stock  →  forward-held label says ETF/fund/other
                     →  excluded before the sample is drawn
                     →  never tested
                     →  the universe silently loses valid common stocks
```

Contamination (an ETF labelled CS) was catchable because such a name is
included. False exclusion was not. The audit must defend both directions.

**Stratify by forward-held bucket** — at minimum `common stock`,
`ETF/fund/other`, and `unknown` — so excluded names are audited too.

For each month, sample **200 ticker/date pairs deterministically** (seeded by
the month), allocated **roughly equally across the three buckets** rather than
proportionally. The common-stock bucket is by far the largest, so equal
allocation deliberately over-samples the small ones — which is where the newly
covered errors live, and rare errors are exactly what proportional sampling
would miss. Where a bucket holds fewer than its allocation, take all of it and
redistribute the remainder.

Then query date-specific reference data for each pair and compare `type` and
`primary_exchange` against the forward-held label.

| Axis | Tolerance |
|---|---|
| Common-stock vs ETF/fund/other, **either direction** | **zero disagreements** |
| `primary_exchange` label | **≤ 0.5%** of sampled pairs |

Either direction is the point of the pre-classification population: a wrongly
included ETF contaminates the universe, and a wrongly excluded common stock
silently shrinks it. Both are misclassification; only the first was previously
detectable.

Zero tolerance on the security-type axis is not fastidiousness: a mislabelled
ETF is exactly the contamination that put TQQQ into the live candidate pool at
score 97.5 — and the mirror-image error would remove real tradeable names from
every backtest run against this cache, which is harder to notice and worse. One disagreement means the monthly cadence cannot carry that axis,
and the cache is not signed off — return for daily classification or another
source rather than negotiating the threshold afterwards.

Power, stated plainly: 200/month gives 7,200 paired observations over 3 years,
which detects an aggregate disagreement rate around 0.5% comfortably. **Per-month
power is low** — a single bad month could pass unnoticed on the exchange axis.
The zero-tolerance type rule is what carries the safety, not the sample size.

Audit cost: 200 × 36 ≈ **7,200 calls**, counted in Phase A below.

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

0. **Durability.** `outputs/` is gitignored, so a local folder is not an
   artifact and cannot underwrite reproducibility. Each vintage is archived as
   `pit-universe-<vintage>.tar.gz` and attached to a **GitHub Release** tagged
   `pit-universe-<vintage>`; the **manifest is force-added into the repository**
   so the hashes are durably version-controlled even though the payload is not.
   The VPS fetches the release asset through the same authenticated API path it
   already uses for `mas-run-attestation`, verifies every hash against the
   committed manifest, and replays from that. Estimated archive size is
   400–600 MB, within the 2 GB per-asset limit; if a vintage exceeds it the
   manifest records the split parts.
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
- **false-exclusion rate**: audited names whose forward-held label excluded them
  but whose date-specific classification says common stock. Newly measurable
  once the sample is drawn pre-classification; report it whether or not it
  breaches the halt.
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

---

# Appendix A — acquisition budget and call plan

Required before implementation. Frozen rulings from Neo are recorded in §11.

## A.1 What each endpoint can and cannot supply

Probed 2026-08-11 UTC:

| Endpoint | Returns | Cost shape |
|---|---|---|
| `/v2/aggs/grouped/.../{date}` | price + share volume for **every** ticker that traded | **1 call per market date** |
| `/v3/reference/tickers?date=` | ticker, `type`, `primary_exchange`, name, figi — **no market cap, no shares outstanding** | paginated, ~1000/page |
| `/v3/reference/tickers/{t}?date=` | `market_cap`, `weighted_shares_outstanding` | **1 call per ticker per date** |

The middle row is the constraint that shapes everything: classification is cheap and bulk, **market cap is not available in bulk at all**. A naive per-ticker-per-date build is ~2,500 × 750 ≈ **1.9M calls** and is simply off the table.

## A.2 Staged plan, with a hard gate

**Phase A — spine (cheap, proceed without further approval)**

| Item | Calls |
|---|---|
| Grouped daily, 3Y ≈ 750 market dates | **750** |
| Reference list, monthly as-of, ~6 pages each × 36 | **~216** |
| Classification drift audit (§3b), 200/month × 36 | **~7,200** |
| **Phase A total** | **≈ 8,200** |

Yields price, share volume, exchange and security type for every date.

**Classification cadence is monthly, applied forward-only** — frozen in §3a, with the mandatory drift audit in §3b. Daily classification would cost ~4,500 calls; the monthly snapshot plus a 7,200-call audit costs more in raw calls but buys a *measured* bound on the approximation rather than an assumed one. If the audit fails on the security-type axis, the answer is daily classification, not a relaxed threshold.

**Gate.** Phase A ends by reporting the exact count of distinct tickers that pass price + volume + exchange + type. Phase B does not begin until that number is known and approved.

**Phase B — market cap (requires explicit approval)**

Estimated from live universe size (~2,500/day, expected ~4,000–6,000 distinct over 3Y) at **quarterly** shares-outstanding cadence, forward-only like classification, since shares outstanding move on corporate actions and buyback reporting rather than daily:

```
12 quarters × ~5,000 tickers ≈ 60,000 calls
+ threshold audit (§3d: band + sentinel) ≈ 2,700 calls
```

Monthly cadence would be ~180,000. **Quarterly is the proposal; the count is an estimate until Phase A reports the real figure.**

**Abort threshold:** if the Phase A gate shows Phase B would exceed **75,000 calls**, stop and return to Neo/Ray with options rather than proceeding. Do not spend the budget and report afterwards.

## A.3 Rate limits and failure handling

- One request in flight per endpoint family; no burst parallelism.
- Retry on 429/5xx with exponential backoff, capped; a ticker-date that still fails is recorded as an explicit fetch failure, never silently skipped.
- The run is **resumable**: every raw response is written before being parsed, so an interrupted build restarts from the last completed unit rather than re-spending calls.
- The existing FMP daily-budget accounting (`fmp_daily_call_budget`) is the precedent; Polygon needs the equivalent counter for this build.

## A.4 Raw snapshot and cache layout

Feeds §5's replay determinism — the raw layer is the immutable input, and the normalized dataset is a pure function of it.

```
outputs/pit_universe/<vintage>/
  raw/
    grouped/<YYYY-MM-DD>.json.gz
    reference/<YYYY-MM>/page-<n>.json.gz
    details/<YYYY-Qn>/<TICKER>.json.gz
  normalized/
    membership/<YYYY-MM-DD>.parquet
  manifest.json          # raw hashes, normalization version, config hash, code SHA, ET range, counts
```

`<vintage>` is the retrieval date. A rebuild writes a new vintage and diffs against the previous one; it never overwrites.

## A.5 Quantified acceptance thresholds

Replaces "a large or trending unknown rate invalidates the dataset", which cannot survive as an interpretation after the build.

Report **every** rate unconditionally. **Halt research consumption** — the dataset is not signed off — if any of:

| Metric | Halt threshold |
|---|---|
| Security-type unknown, share of daily traded set passing price+volume | **> 1%** on any month |
| Exchange unknown, same denominator | **> 1%** on any month |
| Market-cap unknown (no as-of value and no shares × close) | **> 5%** on any month |
| Monthly unknown rate vs the trailing-12-month median for that metric | **> 2×** |
| PIT daily count vs contemporaneous live eligible count, on dates where live observations exist | **> 15%** divergence, median over the overlap |

The last row is measurable only where live records exist — the candidate table and dashboard history, i.e. recent months — so it validates the recent end of the series and cannot speak for 2023. That limitation is a reported fact, not a caveat to be discovered later.

## §11 — Frozen rulings (Neo, 2026-08-11)

```
Range/frequency: 3 years, daily ET market dates
Price/volume:    D close and D share volume; decision after D close, entry T+1
Market cap:      DAILY ESTIMATE (§3c) — quarterly as-of weighted_shares_
                 outstanding held forward x daily D close. NOT daily market_cap
                 retrieval. Unresolvable => exclude and count unknown.
                 Threshold audit per §3d — band sample AND out-of-band
                 sentinel — gates Phase B acceptance.
Exchange/type:   as-of Polygon reference only; unknown excludes and is counted
Corporate acts:  split-adjusted prices; no dividend/total-return adjustment
Aliases:         native Polygon symbol + explicit dated alias mapping; never drop
History:         minimum 200 prior bars available as of D
Delisting:       include through final trading day; never forward-fill after
Volume:          single-day share volume, matching live screener semantics
```
