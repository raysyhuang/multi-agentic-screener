# Neo independent reselection — full inputs pack

**Branch:** `research/momentum-exchange`
**Input commit:** `58e9282973fe49e9ad4e4fff20da5337609bb4b3`
**Date:** 2026-11-09
**Scope:** research only; size 0; no main/PR/live/freeze change.

## Verdicts

| Item | Verdict |
|---|---|
| Annual Top1 reselection and 2015–2024 path | **APPROVE, conditional on the supplied provisional membership/price/corporate-action inputs** |
| W10H2 and W12H4 Top1 selection across all 12 phases | **APPROVE, conditional on the same inputs** |
| Input truth / identity / corporate-action closure | **MODIFY** |

## Independent procedure

Neo fetched the branch before opening Queen's answer-key directory, inspected the bilingual spec and input schemas, then implemented an independent selector from:

- `eligibility_by_signal.csv.gz` (`eligible_final`);
- `rank_prices_month_end.csv.gz` plus signal-scoped overrides;
- `corporate_actions_applied.csv`;
- `trading_sessions.csv`;
- execution price/daily-close/event files for K1 paths.

Only after generating the selections were the outputs compared with `queen/answer-key-20261109`.

The package's own scripts were also run after temporarily copying the separate answer keys beside them.

## A. Annual Top1

Independent selections:

| Holding year | Signal | Winner | 12m return | Runner-up | Runner-up return |
|---:|---|---|---:|---|---:|
| 2015 | 2014-12-31 | LUV | 124.6285% | EA | 104.9477% |
| 2016 | 2015-12-31 | NFLX | 134.3804% | AMZN | 117.7831% |
| 2017 | 2016-12-30 | NVDA | 223.8471% | OKE | 132.8062% |
| 2018 | 2017-12-29 | NRG | 132.3002% | VRTX | 103.4207% |
| 2019 | 2018-12-31 | AMD | 79.5720% | AAP | 57.9496% |
| 2020 | 2019-12-31 | AMD | 148.4290% | LRCX | 114.7316% |
| 2021 | 2020-12-31 | NVDA | 121.9295% | PYPL | 116.5110% |
| 2022 | 2021-12-31 | DVN | 178.6211% | MRO | 146.1769% |
| 2023 | 2022-12-30 | OXY | 117.2818% | HES | 91.5710% |
| 2024 | 2023-12-29 | NVDA | 238.8668% | META | 194.1333% |

All ten selections match Queen's answer key.

Rebuilt path under the supplied close-execution and corporate-action rules:

- final multiple: **113.780624x**
- CAGR: **60.538311%**
- daily MDD: **−36.613739%**

This upgrades Neo's prior annual-selection verdict from MODIFY to **APPROVE, input-conditional**.

The January phase remains outcome-selected. The same pack reports the twelve-phase W12H12 median CAGR as **36.3909%**, so 60.54% is not an OOS expectation.

## B. W10H2 / W12H4

Independent rebuild compared all complete ranking frames and all K1 execution paths:

- ranking frames: **1,584 checked, 0 mismatches**
- maximum absolute return difference: **4.44e−16**
- K1 paths: **36/36 reproduced**

Twelve-phase medians:

- W10H2 Top1 CAGR: **45.2403%**; median MDD **−51.7750%**
- W12H4 Top1 CAGR: **43.0262%**; median MDD **−52.0042%**
- W12H12 Top1 CAGR: **36.3909%**; median MDD **−56.0166%**

Verdict: **APPROVE, input-conditional** for Queen's selections and ledger arithmetic.

## C. Input and identity audit

### Closed or improved

1. **CPRI/KORS:** `eligibility_by_signal.csv.gz` explicitly removes duplicate KORS before ranking on 20 signal dates. CPRI remains the one economic issuer with `ticker_at_signal=KORS`. The previous Top2/Top3 duplicate blocker is mechanically closed in v18 for this identity.
2. **HPE→DXC:** ratio `0.085904` and factor `0.7506542665` are present with the cited SEC 8-K.
3. **Unknown discipline:** exactly twelve rank frames are UNKNOWN—W1–W12 on 2025-11-28—each caused by unresolved DD/Qnity spin treatment. Earlier frames are complete.
4. **Annual signals:** all ten have zero unrankable eligible names under the supplied inputs.

### Remaining limitations

1. Membership remains a provisional public reconstruction with only four official-date patches. The bytes reproduce the engine but do not independently establish historical S&P membership truth.
2. Of the ranking-price table, **4,078 rows** use `other_engine_rule(identity series/successor conversion/deal value/class or receipt units)` rather than a direct Massive grouped close. These are transparent but need row-level source lineage before the input set can be called independently sourced.
3. Several membership identity lines visibly alternate old/new labels across snapshots—for example BHGE/BHI, DXC/CSC and WBA/WAG. The engine-level issuer bridge may be economically intended, but stable CIK/security identifiers should replace ticker-string routing.
4. `CCEP` in 2016 is acknowledged as probably CCE. It did not affect winners, but remains an unresolved historical ticker identity.
5. Corporate-action rows with evidence grade C remain provisional. Selected K1 paths touch PFE/VTRS and PPL/TLN on the ranking side; they were not held, so current Top1 execution P&L is unaffected, but full-universe ranking remains provisional.
6. Price returns omit ordinary dividends. Report them as price/corporate-action-adjusted results, not shareholder total return.
7. The committed `MANIFEST.sha256` no longer matches `README_CN.md` after the answer-key relocation/edit. All load-bearing data files checked against the manifest; `README_CN.md` is the sole mismatch found.

## Final scope

The supplied pack is sufficient to independently reproduce Queen's selection and path outputs. It is not sufficient to prove that the provisional public membership reconstruction and derived corporate-action price series are historically correct versus an independent provider.

Therefore:

- annual selection/result: **APPROVE, input-conditional**;
- W10H2/W12H4 selections/results: **APPROVE, input-conditional**;
- source-level data truth: **MODIFY**;
- live/freeze/size authority: none; size remains 0.

## Hawk status

Hawk/Fable-5 was **NOT RUN**. The hermetic Claude Code smoke test was rejected before inference because the organization has disabled Claude subscription access for Claude Code. No workaround was used.
