# MR hold length and bear cohort — 2026-09-18

Two questions from the 2026-09-18 model review, both about `mean_reversion|mas_official`:

1. Does holding MR longer help? FORWARD_DECAY_FINDINGS (2026-08-13) measured **+1.11pp** per pick from holding 5 bars instead of taking the live exit (n=35, CI [−0.77, +3.14]) and called it a hypothesis.
2. Is MR's bear cohort structurally bad? Live 90d to 09-17: bear **−0.42%/trade (n=21)** vs choppy +0.99%, and the regime has been bear 12 of the last 14 sessions.

## Caveat first

The full-scale backtest population (Arm B) is **edgeless at live selectivity** (MEMORY: MR raw −0.047%/trade; here +0.039%). Arm B can only say whether bear is worse **than the rest of that population** and whether hold length changes the **exit walk**. It is not an expectancy claim and it is not the live-selected subset. Arm A is the one that speaks to live picks, and its CIs were expected to cross zero. They do.

## Verdicts

| question | verdict |
|---|---|
| Hold MR longer (the +1pp finding) | **DOES NOT SURVIVE out-of-sample.** On the 22 picks added since the August bundle, holding 5 bars was −0.98pp/pick (median −0.75, hit 38%). The +1.33pp seen on the 49-pick window is entirely the 27 picks the August finding was already made on. |
| Hold length in the engine (3 → 5 → 7 bars) | **IRRELEVANT.** The 0.5/0.3 trail exits 79% of trades at a median 1 bar; only 252 of 27,822 trades reach the 3-bar expiry. `max_hold` does not bind; avg moves +0.039 → +0.041%. Any "hold longer" idea is really "don't trail", which is a different, already-rejected question. |
| Bear-block MR | **NO ACTION** under the pre-registered rule. Full scale, bear is MR's *best* market regime (+0.497%, n=4,810, CI [+0.40, +0.60]) vs ex-bear −0.057% — but it is **year-unstable**: 2025 +1.52% (n=2,070) carries it; 2023 −0.40% and **2026 −0.31% (n=1,603)** are negative. The live bear −0.42% is consistent with the 2026 backtest bear cohort, i.e. this year's bear tape is bad for MR, not bear regimes as a class. |

Nothing here supports a parameter change. Do not touch the MR stop, trail, hold or regime gate on this evidence.

## Arm A — live-selected picks (updates the forward-decay finding)

Input: frozen dashboard bundle `outputs/research/frozen/data-2026-09-17T230250Z.json`, sha256 `dbd9e336c524a924ecf348db74e02246965d7b4dcc7c0061a2d59cfc78f51ef6` (90d window to 2026-09-17), stream `mean_reversion|mas_official`, n=49 closed trades. Buy-and-hold from the actual T+1 open vs SPY over the same bars; Polygon strict, no fallback.

Provenance: `{"provider": "polygon", "fallback_reason": null, "requested": 49, "returned": 49, "missing": [], "failures": {}}`

Realized as traded: n=49, mean +0.11%, median −0.17%, win 45%.

| horizon | n (complete) | fwd mean | fwd median | win | alpha vs SPY |
|---|---|---|---|---|---|
| 5 bars | 43 | +0.80% | −0.44% | 49% | +0.58% |
| 7 bars | 40 | +0.25% | +0.48% | 52% | −0.32% |
| 21 bars | 27 | +3.96% | +4.08% | 63% | +2.45% |

Paired delta (hold h bars − realized), seeded 10k bootstrap, split-half by entry date:

| cohort | h | n | mean | 95% CI | median | hit | split-half |
|---|---|---|---|---|---|---|---|
| all with complete window | 5 | 43 | +0.47pp | [−0.98, +2.08] | −0.11 | 49% | +0.33 / +0.60 |
| all with complete window | 7 | 40 | −0.13pp | [−2.06, +1.67] | +0.26 | 50% | −0.89 / +0.63 |
| matched (full 21b window) | 5 | 27 | +1.33pp | [−0.84, +3.65] | +1.00 | 56% | +1.02 / +1.61 |
| matched (full 21b window) | 21 | 27 | +3.56pp | [−1.15, +8.20] | +2.39 | 56% | +5.72 / +1.55 |

**Out-of-sample split** against the August pin (`frozen/data-2026-08-14T205945Z.json`): 27 of the 49 picks were already in the bundle the +1.11pp finding was made on; 22 are new.

| cohort | h | n | delta mean | 95% CI | median | hit | fwd mean | alpha vs SPY |
|---|---|---|---|---|---|---|---|---|
| **NEW since Aug-14** | 5 | 16 | **−0.98pp** | [−2.25, +0.13] | −0.75 | **38%** | −0.76% | −0.20% |
| NEW since Aug-14 | 7 | 13 | −1.31pp | [−3.16, +0.50] | −0.24 | 46% | −0.96% | −0.49% |
| NEW since Aug-14 | 21 | 0 | — | — | — | — | — | — |
| overlap (in Aug-14) | 5 | 27 | +1.33pp | [−0.84, +3.65] | +1.00 | 56% | +1.73% | +1.05% |
| overlap (in Aug-14) | 21 | 27 | +3.56pp | [−1.15, +8.20] | +2.39 | 56% | +3.96% | +2.45% |

Read: the matched-cohort +1.33pp is the August finding re-measured on the same picks (n=27 is exactly the overlap). The only new evidence is the 16–22 recent picks, and on those the live exit beat holding at every horizon with a complete window. n=16 is tiny and the CI touches zero, but a 58%-hit / +1pp hypothesis that comes back at 38% / −1pp on its first fresh sample has not survived. The August doc's own stress test (sign-stable halves) fails too: 7-bar halves −0.89 / +0.63.

## Arm B — full-scale backtest population, hold × MARKET regime

Cache: `outputs/research/ohlcv_polygon_3y.parquet` — 504 tickers (incl. SPY), 374,634 rows, 2023-07-26 → 2026-07-24. **No `.provenance.json` exists beside this cache** (it pre-dates the convention); it is the same Polygon cache every prior MR full-scale study used. Live-faithful `LIVE_MR` params (RSI2 ≤ 10, min_score 50, stop 0.75×ATR, target 1.5×ATR, trail 0.5/0.3, gap-through), unified exit engine, hold ∈ {3, 5, 7}. Regime = SPY SMA20/50 on the trade's **entry date** (`spy_market_regime`), never the per-ticker `regime` field.

Hold = 3 (live):

| market regime | n | WR | avg | 95% CI | median | avg hold | per-year avg (n) |
|---|---|---|---|---|---|---|---|
| all | 27,822 | 52.2% | +0.039% | [+0.010, +0.069] | +0.10 | 1.7 | 2023 −0.11 (1,384) · 2024 −0.11 (10,529) · 2025 +0.29 (10,018) · 2026 −0.08 (5,891) |
| bull | 17,816 | 51.2% | −0.069% | [−0.100, −0.038] | +0.05 | 1.8 | +0.40 · −0.11 · −0.05 · −0.05 |
| choppy | 5,196 | 52.8% | −0.015% | [−0.085, +0.053] | +0.13 | 1.7 | +0.43 · −0.16 · +0.04 · +0.22 |
| **bear** | 4,810 | 55.3% | **+0.497%** | [+0.395, +0.600] | +0.27 | 1.6 | **−0.40 (886) · +0.37 (251) · +1.52 (2,070) · −0.31 (1,603)** |
| ex-bear | 23,012 | 51.6% | −0.057% | [−0.086, −0.029] | +0.07 | 1.8 | +0.41 · −0.12 · −0.03 · −0.00 |

Exit mix at hold 3: trail_stop 21,897 · stop 5,498 · expiry 252 · target 175.

Hold sweep (population identical, only `max_hold` changes):

| hold | all avg | bear avg | expiry exits | trail exits | equity (cap 10, $100k) |
|---|---|---|---|---|---|
| 3 | +0.039% | +0.497% | 252 | 21,897 | −55.0%, DD 61.1%, Sharpe −1.20 |
| 5 | +0.040% | +0.493% | 41 | 22,063 | −55.8%, DD 61.8%, Sharpe −1.24 |
| 7 | +0.041% | +0.496% | 16 | 22,079 | −55.1%, DD 61.2%, Sharpe −1.21 |

The equity row is the concurrency-capped replay of the *whole* population (5.4k taken, 22k skipped); it is reported for completeness and says only what MEMORY already says — the unselected MR population is not tradeable. It is not a statement about live.

**Pre-registered bear rule** (from `scripts/pead_regime_stamp.py`): bear ≥ +0.5%/trade @ n≥100 → no gate; bear ≤ 0 → propose bear-block; otherwise no action. Outcome at hold 3: **NO ACTION** (+0.497% @ n=4,810 — a hair under the +0.5 "no gate" line and nowhere near ≤ 0). The same at hold 5 and 7.

## What this changes

- Retire the "MR gains ~+1pp from holding ~7 calendar days" line from the research log; it is now a hypothesis that failed its first out-of-sample sample. Keep the sniper half of that finding (not re-tested here).
- The engine-level hold study is closed: `max_hold` is not a lever while the trail is on.
- Bear: no gate. The live bear −0.42% (n=21) matches the 2026 backtest bear cohort (−0.31%, n=1,603), so the current pain is the year, not the regime class. Re-read at the next n=30/50 evaluation point of the live stream, not before.

## Reproduce

```bash
# Arm A (strict Polygon; ~1 min)
python scripts/sniper_forward_returns.py \
  --input outputs/research/frozen/data-2026-09-17T230250Z.json \
  --stream "mean_reversion|mas_official" --horizons 5,7,21 \
  --baseline-input outputs/research/frozen/data-2026-08-14T205945Z.json
# Arm B (~15 min)
python scripts/mr_hold_and_bear.py --cache-file outputs/research/ohlcv_polygon_3y.parquet \
  --json-out outputs/research/mr_hold_and_bear.json
```

Seeds: bootstrap 20260918 (both arms). Arm B aggregates are in `mr_hold_and_bear.json` beside this file. No per-trade P&L is committed.
