# Exit-layer study — trail guard + MR stop width (2026-08-04)

**Status: both hypotheses REJECTED. No code change recommended.**

Scripts: `scripts/exit_trail_intraday_study.py`, `scripts/mr_stop_fullscale.py`.

## Why this was investigated

Exit-reason breakdown across every live stream showed the exit layer, not the
entry layer, is where the money moves:

| Stream | n | WR | trail_stop | stop / time_stop | median hold | % of peak captured |
|---|---|---|---|---|---|---|
| MR official | 31 | 54.8% | 74% | 16% @ −2.97% | 1d | 65% |
| MR sleeve | 74 | 50.0% | 68% | 23% @ −2.52% | 1d | 54% |
| Sniper | 70 | 57.1% | 86% | 14% @ −5.68% | 1d | 51% |
| PEAD neglected | 6 | 50.0% | 67% | 33% @ −5.87% | 1d | 20% |

74–86% of ALL exits are `trail_stop`, and 77–86% of losers were up >0.5% before
dying. Median hold is 1 day on *every* stream — including sniper (designed 7d)
and PEAD (designed 20d).

## H1 — the same-bar trail guard. Mechanism CONFIRMED, P&L REFUTED.

`exit_engine.py:169-182` forbids a trail from enforcing on the bar it arms
(daily OHLC can't prove high-before-low). With median hold of 1 day the arming
bar IS the entry bar, so single-bar round trips are forced to the hard stop:

```
RARE 2026-07-20 hold=0 reason=stop pnl=-2.48% MFE=+2.28% MAE=-2.93%
NET  2026-07-28 hold=0 reason=stop pnl=-3.12% MFE=+1.83% MAE=-4.91%
```

RARE peaked at 4.5× the 0.5% activation threshold and still booked the hard stop.

**Ordering (Polygon 1-minute):** of 38 resolvable cases, **25 high-first vs 13
low-first** — the engine's forced assumption is wrong ~2:1. (PR #21's minute
study measured only *stop-vs-target* ties, which genuinely are negligible. The
trail-arm ambiguity is the frequent one and had never been measured.)

**But enforcing the trail intraday over ALL trades, winners included:**

| stream | WR | avg/trade | sum |
|---|---|---|---|
| MR official | 54.8% → **93.5%** | +0.510% → **+0.236%** | +15.8 → +7.3pp |
| sniper | 57.1% → **90.0%** | +0.714% → **+0.024%** | +50.0 → +1.6pp |
| MR sleeve | 50.0% → 82.4% | +0.085% → +0.049% | +6.3 → +3.6pp |

Losers improve exactly as expected (+21 / +78 / +44pp) — but winners lose far
more (sniper winners +161pp → +35pp). A 0.3% trail that actually enforces caps
every winner at ~+0.3%.

> **Win rate is purchasable and nearly worthless.** A 90%+ WR book is available
> on demand at ~zero profit. Same lesson as the 82% sniper fill-realism artifact,
> reached from the opposite direction. Optimize avg P&L per trade, not WR/hit rate.

**Do not "fix" the guard.** It is conservative in the right direction, which also
means recorded live P&L is the pessimistic branch — the dashboard is not flattered.

Methodological trap worth naming: replaying only the LOSERS (the obvious move,
since they look like the victims) showed a fake +110pp gain. The bias only
disappears when winners are replayed under the same rule.

## H2 — widen the MR stop. REJECTED at full scale (replicates the 2026-07 result).

3Y Polygon cache, 504 tickers, live execution config, varying ONLY the stop:

| min_score | config | n | WR | avg/trade | 95% CI |
|---|---|---|---|---|---|
| **75 (LIVE)** | live tiers | 7889 | 53.35% | **−0.019%** | [−0.066, +0.027] |
| 75 (LIVE) | tiers ×2.0 | 7889 | 55.91% | +0.004% | [−0.046, +0.053] |
| 75 (LIVE) | flat 2.0×ATR | 7889 | 55.99% | −0.002% | [−0.052, +0.048] |
| 50 (low) | flat 0.75×ATR | 27822 | 52.23% | +0.039% | [+0.009, +0.069] |
| 50 (low) | flat 2.0×ATR | 27822 | 55.94% | +0.062% | [+0.030, +0.094] |

1. The monotonic direction IS real at both selectivity levels (live: −0.028 →
   −0.002, WR 51.7% → 56.0%). It just moves MR from slightly-negative to zero.
2. **The low-selectivity "significance" is a one-year artifact and the pooled CI
   hides it.** Year splits: 2023 −0.12 / 2024 −0.11 / **2025 +0.29** / 2026 −0.09.
   Three of four years negative; the CI excludes zero only by treating 27k trades
   as independent draws when they are really four regime-years.
3. MR's raw signal population has no edge at live selectivity under realistic
   execution. Live +0.38–0.51%/trade is the ranking + correlation filter + gate
   funnel, which this backtest does not model.

**Rule adopted: always print the per-year split beside any CI.** The CI alone
would have sold this change.

## Standing conclusions

- Stop looking for parameter alpha in MR: score IC ≈ 0, trail rejected (2026-07),
  stop rejected twice (2026-07, 2026-08).
- Trail OFF is catastrophic for sniper: −1.429%/trade, 20% WR, −100pp. The trail
  is not a refinement there, it IS the strategy.
- Sniper `act=2.0 + stop=0.75×ATR` (+1.069%/trade on n=70) is a two-parameter
  joint optimum whose stop change alone is negative (−0.068pp) — treat as overfit.
- Open, untested: PEAD's horizon mismatch (median hold 1d on a 20-day drift
  strategy, winners capturing 20% of peak). Structural, not a tuning question.
