# Intraday MR probe → discovered intraday VWAP-MOMENTUM (2026-07-19)

`scripts/intraday_mr_probe.py` on real Polygon minute bars: 20 most-liquid S&P
names × 25 days sampled across 2023-05 → 2026-04, session VWAP computed per
ticker-day, observations taken at fixed decision times (11:00–15:00 ET) to avoid
overlap inflation. N=2,420. Forward return measured to the close.

## Result: intraday MEAN-REVERSION is refuted; the effect is MOMENTUM

Forward-to-close is monotonic in VWAP deviation — the opposite of reversion:

| VWAP dev at decision | fwd→close | edge vs base | % up | N |
|---|---|---|---|---|
| ≤ −1.5% | −0.50% | −45 bp | 37% | 170 |
| −1.5 to −1.0% | −0.35% | −30 bp | 42% | 180 |
| −1.0 to −0.5% | −0.24% | −20 bp | 44% | 314 |
| −0.5 to 0% | −0.11% | −7 bp | 50% | 651 |
| 0 to +0.5% | +0.08% | +13 bp | 58% | 622 |
| +0.5 to +1.0% | +0.19% | +24 bp | 59% | 277 |
| **≥ +1.0%** | **+0.40%** | **+45 bp** | **64%** | 206 |

Base rate (all obs): −0.05%. Price *relative to session VWAP* predicts continuation
through the close: below-VWAP keeps falling, above-VWAP keeps rising. This is the
classic intraday VWAP-momentum / trend-persistence effect — and it is clean and
monotonic across all seven buckets, not a single lucky bucket.

## The promising signal: long strongly-above-VWAP into the close

The ≥+1% bucket: +0.40% avg to close, 64% up, N=206 — a real, sizeable edge over
the −0.05% base rate. A long-only intraday rule (enter names ≥1% above VWAP after
~11:00, exit at close) is the natural expression.

## Caveats before ANY excitement (the whole point of this session's discipline)

1. **Gross returns, no costs.** +0.40% to close on a liquid name, entering intraday
   and exiting at close, loses ~0.1–0.2% to spread+slippage → net ~+0.2–0.3%.
   Still positive but thinner; needs the unified-engine backtest with real fills.
2. **Mega-cap, bull-window sample.** The 20 most-liquid names in 2023–2026 are
   AAPL/MSFT/NVDA-type mega-caps in a strong uptrend. Intraday momentum there may
   be regime- and universe-specific. Widen the ticker sample and split the window.
3. **Well-known / competitive.** VWAP momentum is HFT-adjacent; the edge at
   minute-entry/close-exit granularity for a slower book is the open question.
4. Probe uses fixed decision times and one observation per (ticker, time) — low
   overlap, but still not a portfolio backtest.

## Verdict & next step

Intraday MR (candidate 2) is refuted; **intraday VWAP-momentum is the most
promising signal found this session** and supersedes it as the candidate to pursue.
Next: a proper intraday VWAP-momentum backtest through the unified exit engine
(minute-resolution fills, real costs, concurrency cap), on a WIDER liquid universe
and split by sub-period, before any paper trial. Do not promote on the strength of
a gross-return probe — that is exactly how the 82% sniper illusion happened.

Two hypotheses refuted by honest testing this session (gap-continuation,
intraday-MR); one real directional signal discovered (intraday VWAP-momentum).

## Addendum — the proper backtest KILLS VWAP-momentum too (2026-07-19)

`scripts/intraday_vwap_backtest.py`: single actionable entry at 12:00 ET (dev ≥
threshold above session VWAP), ride to the close through the unified exit engine
(minute ExitBars, expiry), net of realistic per-side cost. 40 liquid tickers × 40
sampled days.

| test (dev≥+1%, ride to close) | N | WR | avg net% |
|---|---|---|---|
| net of 5 bp/side | 78 | 41% | **−0.26** |
| **zero cost** | 78 | 45% | **−0.16** |
| below-VWAP control | 158 | 48% | −0.28 |
| early / mid / late third | 25/22/31 | — | +0.11 / −0.53 / −0.37 |

**Verdict: refuted.** The signal is negative even before costs; the below-VWAP
CONTROL is equally negative (so there is no *directional* edge — both sides lose);
and it is positive in only one of three sub-periods. The +0.40% probe number was an
**intraday-overlap artifact**: the probe pooled 5 decision times per day, so an
all-day-strong name was counted ~5× and inflated the mean. Counting each ticker-day
once — a real tradeable entry — erases the edge. Classic why-you-must-backtest-not-
probe. Do not pursue.

### Session scorecard on new alphas (all liquid large-cap technical signals)
- Gap-continuation: dead (daily + intraday).
- Intraday mean-reversion: refuted (effect is momentum).
- Intraday VWAP-momentum: refuted (overlap artifact; no edge single-entry, control
  also negative, unstable across sub-periods).
- Post-earnings drift: still untested (needs point-in-time earnings dates).

Read-through: simple price-derived technical signals on the most-liquid large-caps
are efficiently priced — three plausible ones all died under honest testing. The
value delivered is the TESTING FRAMEWORK (unified engine + realistic costs +
controls + sub-period splits + overlap-aware sampling) that reliably kills
artifacts — exactly what was missing when the 82% sniper backtest was believed.
Genuine alpha more likely needs a data edge the $199 plan enables but we haven't
mined (short interest, options flow), a less-efficient universe (small/mid-caps),
or an event edge (post-earnings drift) — not more technical signals on mega-caps.
