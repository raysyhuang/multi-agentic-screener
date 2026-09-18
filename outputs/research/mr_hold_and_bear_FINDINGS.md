# MR hold length and bear cohort — 2026-09-18 (revised after Codex review)

Two questions from the 2026-09-18 model review, both about `mean_reversion|mas_official`:

1. Does holding MR longer help? FORWARD_DECAY_FINDINGS (2026-08-13) measured **+1.11pp** per pick from holding 5 bars instead of taking the live exit (n=35, CI [−0.77, +3.14]) and called it a hypothesis.
2. Is MR's bear cohort structurally bad? Live 90d to 09-17: bear **−0.42%/trade (n=21)** vs choppy +0.99%, and the regime has been bear 12 of the last 14 sessions.

## Revision note (v2, same day)

The first version of this document was reviewed by Codex (2026-09-18) and six defects were fixed before merge. What changed and why:

| defect | fix | effect on the numbers |
|---|---|---|
| Regime label used the **entry day's own close** (an open entry cannot see it) | entries are stamped with the regime as of the **last completed session** (`spy_market_regime(lag_sessions=1)`) | 1,725 of 27,822 labels moved; bear n 4,810 → **4,614**, avg +0.497% → **+0.498%** |
| Bootstrap resampled **trades** iid although thousands share an entry date | added an **entry-date cluster bootstrap**; both intervals reported, the cluster one is decisional | bear CI [+0.39, +0.60] (iid) → **[−0.66, +1.68] (cluster)** — the bear "edge" is not statistically established |
| Headline denominators wrong (said 22 picks / 49-pick window) | every n is the n actually used | OOS row is **n=16** complete 5-bar windows (of 22 new picks); the +1.33pp is the **n=27** overlap; the all-picks delta is **+0.47pp, n=43** |
| Per-horizon paired stats not reproducible from the documented command | `sniper_forward_returns.py` now prints a paired block per horizon (all complete windows + matched cohort) with iid and cluster CIs | none (the 43-pick row was computed by hand before; now it prints) |
| Ticker/SPY horizon windows could use different dates | SPY measured over the ticker's exact bar dates; alpha None if SPY lacks a bar | **0 rows affected** on this cohort |
| Default stdout of `sniper_forward_returns.py` changed | all new report blocks gated behind `--stream/--horizons/--baseline-input` | default **layout** is identical to `origin/main`; the SPY-window alignment fix (row above) is NOT gated — it corrects the benchmark on the default path too, so a default run can print different `spy_*`/`alpha_*` values than `origin/main` whenever a ticker is missing a bar. On this cohort 0 rows are affected, so the reproduced 2026-08-13 sniper run matches; that is a property of the data, not a guarantee |

Pre-registration honesty: the bear rule below was adopted for MR **in the session plan (2026-09-18), not in a prior commit**; it is copied verbatim from `scripts/pead_regime_stamp.py`, where it was pre-registered for PEAD.

## Caveat first

The full-scale backtest population (Arm B) is **edgeless at live selectivity** (MEMORY: MR raw −0.047%/trade; here +0.039%, cluster CI [−0.17, +0.29]). Arm B can only say whether bear is worse **than the rest of that population** and whether hold length changes the **exit walk**. It is not an expectancy claim and it is not the live-selected subset. Arm A is the one that speaks to live picks, and its CIs were expected to cross zero. They do — by both bootstraps.

## Verdicts

| question | verdict |
|---|---|
| Hold MR longer (the +1pp finding) | **DOES NOT SURVIVE out-of-sample.** On the picks added since the August bundle (22 new; **16** with a complete 5-bar window), holding 5 bars was **−0.98pp/pick** (median −0.75, hit 38%, cluster CI [−2.38, +0.33]). The +1.33pp seen on the matched cohort is **exactly the 27 picks** the August finding was already made on, re-measured. |
| Hold length in the engine (3 → 5 → 7 bars) | **IRRELEVANT.** The 0.5/0.3 trail exits 79% of trades at a median 1 bar; only 252 of 27,822 trades reach the 3-bar expiry. `max_hold` does not bind; avg moves +0.039 → +0.041%. Any "hold longer" idea is really "don't trail", which is a different, already-rejected question. |
| Bear-block MR | **NO ACTION** under the pre-registered rule at the live hold (3). Full scale, bear is MR's *best* market regime on the point estimate (+0.498%, n=4,614) vs ex-bear −0.052% — but the **cluster CI [−0.66, +1.68] spans zero**: 4,614 trades on 86 entry dates is ~86 draws of the market, not 4,614. It is also year-unstable: 2025 +1.64% (n=1,877) carries it; 2023 −0.40% and **2026 −0.31% (n=1,598)** are negative. The live bear −0.42% is consistent with the 2026 backtest bear cohort, i.e. this year's bear tape is bad for MR, not bear regimes as a class — and the backtest cannot distinguish either reading. |

Nothing here supports a parameter change. Do not touch the MR stop, trail, hold or regime gate on this evidence.

## Arm A — live-selected picks (updates the forward-decay finding)

Input: frozen dashboard bundle `outputs/research/frozen/data-2026-09-17T230250Z.json`, sha256 `dbd9e336c524a924ecf348db74e02246965d7b4dcc7c0061a2d59cfc78f51ef6` (90d window to 2026-09-17), stream `mean_reversion|mas_official`, n=49 closed trades. Buy-and-hold from the actual T+1 open vs SPY over the **same bar dates**; Polygon strict, no fallback.

Provenance: `{"provider": "polygon", "fallback_reason": null, "requested": 49, "returned": 49, "missing": [], "failures": {}}` (48 tickers + SPY). Ticker/SPY window alignment: 0 horizon-rows would have differed under SPY's own bar count.

Realized as traded: n=49, mean +0.11%, median −0.17%, win 45%.

| horizon | n (complete) | fwd mean | fwd median | win | alpha vs SPY |
|---|---|---|---|---|---|
| 5 bars | 43 | +0.80% | −0.44% | 49% | +0.58% |
| 7 bars | 40 | +0.25% | +0.48% | 52% | −0.32% |
| 21 bars | 27 | +3.96% | +4.08% | 63% | +2.45% |

Paired delta (hold h bars − realized), seeded 10k bootstrap; **iid** resamples picks, **cluster** resamples entry dates (the honest interval — 43 picks sit on 27 dates); split-half by entry date:

| cohort | h | n | dates | mean | iid 95% CI | cluster 95% CI | median | hit | split-half |
|---|---|---|---|---|---|---|---|---|---|
| all with complete window | 5 | 43 | 27 | +0.47pp | [−0.98, +2.08] | [−1.18, +2.48] | −0.11 | 49% | +0.33 / +0.60 |
| all with complete window | 7 | 40 | 25 | −0.13pp | [−2.06, +1.67] | [−2.58, +2.10] | +0.26 | 50% | −0.89 / +0.63 |
| all with complete window | 21 | 27 | 15 | +3.56pp | [−1.15, +8.20] | [−1.35, +7.93] | +2.39 | 56% | +5.72 / +1.55 |
| matched (all 3 windows) | 5 | 27 | 15 | +1.33pp | [−0.84, +3.65] | [−1.09, +4.39] | +1.00 | 56% | +1.02 / +1.61 |
| matched (all 3 windows) | 7 | 27 | 15 | +0.44pp | [−2.24, +2.87] | [−3.10, +3.57] | +0.98 | 52% | −0.41 / +1.22 |

**Out-of-sample split** against the August pin (`frozen/data-2026-08-14T205945Z.json`, keyed on signal_date + ticker): 27 of the 49 picks were already in the bundle the +1.11pp finding was made on; 22 are new (earliest new signal 2026-08-19).

| cohort | h | n | dates | delta mean | iid 95% CI | cluster 95% CI | median | hit | split-half | fwd mean | alpha vs SPY |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **NEW since Aug-14** | 5 | **16** | 12 | **−0.98pp** | [−2.24, +0.13] | [−2.38, +0.33] | −0.75 | **38%** | −0.20 / −1.76 | −0.76% | −0.20% |
| NEW since Aug-14 | 7 | 13 | 10 | −1.31pp | [−3.20, +0.49] | [−3.25, +1.02] | −0.24 | 46% | −0.23 / −2.23 | −0.96% | −0.49% |
| NEW since Aug-14 | 21 | 0 | — | — | — | — | — | — | — | — | — |
| overlap (in Aug-14) | 5 | 27 | 15 | +1.33pp | [−0.84, +3.65] | [−1.09, +4.39] | +1.00 | 56% | +1.02 / +1.61 | +1.73% | +1.05% |
| overlap (in Aug-14) | 21 | 27 | 15 | +3.56pp | [−1.15, +8.20] | [−1.35, +7.93] | +2.39 | 56% | +5.72 / +1.55 | +3.96% | +2.45% |

Read: the matched-cohort +1.33pp is the August finding re-measured on the same 27 picks. The only new evidence is the 16 new picks with a complete 5-bar window (13 at 7 bars), and on those the live exit beat holding at every horizon, with both split-halves negative. n=16 is tiny and both CIs touch zero, but a 58%-hit / +1pp hypothesis that comes back at 38% / −1pp on its first fresh sample has not survived. The August doc's own stress test (sign-stable halves) fails on the 7-bar all-picks row too: −0.89 / +0.63.

## Arm B — full-scale backtest population, hold × MARKET regime

Cache: `outputs/research/ohlcv_polygon_3y.parquet` — 504 tickers (incl. SPY), 374,634 rows, 2023-07-26 → 2026-07-24. **No `.provenance.json` exists beside this cache** (it pre-dates the convention); it is the same Polygon cache every prior MR full-scale study used. Live-faithful `LIVE_MR` params (RSI2 ≤ 10, min_score 50, stop 0.75×ATR, target 1.5×ATR, trail 0.5/0.3, gap-through), unified exit engine, hold ∈ {3, 5, 7}. Regime = SPY SMA20/50 **as of the last completed session before entry** (`spy_market_regime(lag_sessions=1)`), never the per-ticker `regime` field. Lagging moved 1,725 of 27,822 labels vs the entry-day-close stamp of v1.

Hold = 3 (live). `dates` = distinct entry dates in the cell; the **cluster CI resamples those dates** and is the interval to read.

| market regime | n | dates | WR | avg | iid 95% CI | cluster 95% CI | median | per-year avg (n) |
|---|---|---|---|---|---|---|---|---|
| all | 27,822 | 682 | 52.2% | +0.039% | [+0.010, +0.069] | [−0.166, +0.288] | +0.10 | 2023 −0.11 (1,384) · 2024 −0.11 (10,529) · 2025 +0.29 (10,018) · 2026 −0.08 (5,891) |
| bull | 18,201 | 471 | 50.1% | −0.126% | [−0.158, −0.095] | [−0.244, −0.009] | +0.00 | +0.42 · −0.20 · −0.10 · −0.06 |
| choppy | 5,007 | 125 | 57.9% | +0.216% | [+0.149, +0.281] | [−0.063, +0.481] | +0.33 | +0.39 · +0.15 · +0.24 · +0.31 |
| **bear** | 4,614 | **86** | 54.6% | **+0.498%** | [+0.392, +0.605] | **[−0.659, +1.684]** | +0.24 | **−0.40 (887) · +0.30 (252) · +1.64 (1,877) · −0.31 (1,598)** |
| ex-bear | 23,208 | 596 | 51.8% | −0.052% | [−0.081, −0.024] | [−0.162, +0.057] | +0.07 | +0.41 · −0.12 · −0.02 · +0.00 |

Exit mix at hold 3: trail_stop 21,897 · stop 5,498 · expiry 252 · target 175.

Hold sweep (population identical, only `max_hold` changes):

| hold | all avg | bear avg (cluster CI) | expiry exits | trail exits | equity (cap 10, $100k) |
|---|---|---|---|---|---|
| 3 | +0.039% | +0.498% [−0.66, +1.68] | 252 | 21,897 | −55.0%, DD 61.1%, Sharpe −1.20 |
| 5 | +0.040% | +0.498% [−0.66, +1.68] | 41 | 22,063 | −55.8%, DD 61.8%, Sharpe −1.24 |
| 7 | +0.041% | +0.500% [−0.66, +1.69] | 16 | 22,079 | −55.1%, DD 61.2%, Sharpe −1.21 |

The equity row is the concurrency-capped replay of the *whole* population (5.4k taken, 22k skipped); it is reported for completeness and says only what MEMORY already says — the unselected MR population is not tradeable. It is not a statement about live.

**Pre-registered bear rule** (from `scripts/pead_regime_stamp.py`; adopted for MR in the session plan, not a prior commit): bear ≥ +0.5%/trade @ n≥100 → no gate; bear ≤ 0 → propose bear-block; otherwise no action. Outcome at the live hold 3: **NO ACTION** (+0.498% @ n=4,614, a hair under the +0.5 line and nowhere near ≤ 0). Hold 5: NO ACTION. Hold 7 prints "NO GATE" because +0.5003% crosses the line by 0.0003pp — that is a rounding coincidence, not a finding, and hold 7 is not the decisional row. Read the cluster CI instead: the rule's point-estimate test cannot resolve bear in either direction on this population.

## What this changes

- Retire the "MR gains ~+1pp from holding ~7 calendar days" line from the research log; it is now a hypothesis that failed its first out-of-sample sample. Keep the sniper half of that finding (not re-tested here).
- The engine-level hold study is closed: `max_hold` is not a lever while the trail is on.
- Bear: no gate. The live bear −0.42% (n=21) matches the 2026 backtest bear cohort (−0.31%, n=1,598), and the 3Y bear "edge" has a cluster CI spanning zero, so neither "bear is good for MR" nor "bear is bad for MR" is established. Re-read at the next n=30/50 evaluation point of the live stream, not before.

## Reproduce

```bash
# Arm A (strict Polygon; ~1 min). Every row above prints from this one command.
python scripts/sniper_forward_returns.py \
  --input outputs/research/frozen/data-2026-09-17T230250Z.json \
  --stream "mean_reversion|mas_official" --horizons 5,7,21 \
  --baseline-input outputs/research/frozen/data-2026-08-14T205945Z.json
# Arm B (~15 min)
python scripts/mr_hold_and_bear.py --cache-file outputs/research/ohlcv_polygon_3y.parquet \
  --json-out outputs/research/mr_hold_and_bear.json
```

Seeds: bootstrap 20260918 (both arms, both resampling schemes). Arm B aggregates (including the v1 unlagged bear cell under `label_lag`) are in `mr_hold_and_bear.json` beside this file. No per-trade P&L is committed.
