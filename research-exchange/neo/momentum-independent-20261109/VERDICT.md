# Neo independent verification — momentum exchange

**Branch reviewed:** `research/momentum-exchange`
**Reviewed commit:** `0a1f27b530b11bdb14d92f7745439dfe49361f57`
**Date:** 2026-11-09
**Scope:** public/read-only research review; no main changes, no live/freeze/size authority.

## Verdicts

| Item | Verdict | Reason |
|---|---|---|
| Annual Top1 holdings and 60.54% result | **MODIFY** | The ten-holding list and price/fee arithmetic reconcile, but Neo could not independently re-select from a second membership + price universe because those raw universe/ranking inputs are not committed in this branch. |
| GPT/Massive annual repricing | **APPROVE, repricing only** | Using the committed Massive prices, Neo independently recomputed `113.85824x`, CAGR `60.5597%`, consistent with GPT `113.86x` and Queen/FMP `113.78x`. This validates repricing of Queen's list, not selection. |
| W10H2 / W12H4 signal claims | **MODIFY** | The ledger is internally rich and 24 additional selected rows were sampled, but the branch does not contain an independently runnable PIT membership/price selector. The sample CSV therefore records ledger observations, not independent re-selection. |
| Identity/rename closure | **MODIFY** | CPRI/KORS duplicate issuer is confirmed in the ledger: 39 config/phase/signal groups contain both, 78 rows total, best rank 2; Top1 unaffected but Top2/Top3 are contaminated until issuer-level dedupe. |

## A. Annual Top1

Queen reports these 2015–2024 picks:

`LUV, NFLX, NVDA, NRG, AMD, AMD, NVDA, DVN, OXY, NVDA`

The branch contains Queen's selected holdings, daily equity, and FMP-vs-Massive price cross-check, but not a complete independent PIT membership and full ranking-price dataset. Therefore Neo does **not** claim independent re-selection.

### Independent arithmetic using committed Massive prices

Neo recomputed the committed list with:

- first-session **close** entry;
- next-year first-session close exit;
- 25 bps on each actual buy/sell;
- no turnover fee when AMD continues from 2019 into 2020;
- terminal liquidation fee.

Result:

- final multiple: **113.8582405x**
- ten-year CAGR: **60.5597%**

This agrees with GPT's committed `113.86x / 60.55%` and differs from Queen's FMP `113.7806x / 60.5383%` only by provider rounding/price precision. The committed reconciliation's explanation is credible.

The claimed `−36.61%` daily MDD comes from Queen's FMP daily ledger and was not independently rebuilt from a second daily source here.

## B. W10H2 / W12H4

Committed artifacts contain:

- 17,135 ranking rows;
- 1,080 selected signal rows;
- 130 distinct signal dates;
- 24 phase paths, daily ledgers, trades, and manifests.

Neo sampled 24 selected rows spanning both configurations, phases, and 2014–2025 signals into `signal_spotcheck_24.csv`. This expands documentary inspection beyond the previously stated eight GPT checks, but is **not** independent re-ranking because the branch lacks a second raw PIT selector dataset.

The reported medians remain branch claims pending independent selection:

- W10H2 Top1: median CAGR 45.24%, median MDD −51.78%
- W12H4 Top1: median CAGR 43.03%, median MDD about −52.00%

## C. Identity and corporate-action findings

### Confirmed

- **CPRI/KORS:** both aliases appear in 39 config/phase/signal groups. Best rank is 2, so current Top1 paths are unaffected. Any Top2/Top3 result must fail closed until issuer-level dedupe is applied before ranking.
- **ANDV/TSO:** no same-signal duplicate was found; best observed rank among the two is 5. Historical issuer mapping still needs stable-ID confirmation.
- **HPE/DXC:** no same-signal duplicate was found; best observed rank is 7. The corrected 0.085904 exchange ratio is documented, but Neo did not independently verify the SEC receipt in this pass.

### Additional risks to close

1. Stable identity should be issuer/event based, not ticker-name based. Aliases must be deduped **before** TopN ranking so two securities of one economic issuer cannot occupy multiple slots.
2. `GOOG/GOOGL` and other legitimate share classes should not be collapsed automatically; policy must distinguish share classes from ticker renames.
3. Corporate-action mappings listed in Queen's `rank_side_CA` field require source receipts and effective dates. A correct Top1 list does not prove near-rank candidates are clean.
4. January is an outcome-selected entry phase. The 60.54% result is descriptive and not an OOS expected return.

## D. Required next evidence

To upgrade annual selection from MODIFY to APPROVE:

1. commit or attach the frozen PIT membership schedule used for each signal;
2. commit the full price/ranking input needed to independently recompute every eligible name;
3. provide issuer/stable-ID and corporate-action mapping with effective-date receipts;
4. run an independent selector that compares full ranked memberships, not only Top1;
5. publish exact differences and fail closed on unresolved identities.

## Hawk status

Hawk/Fable-5 was **NOT RUN**. A hermetic `claude -p ... --model claude-fable-5` smoke test was rejected before inference because the organization has disabled Claude subscription access for Claude Code. No workaround or API purchase was used.
