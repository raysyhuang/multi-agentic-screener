# Post-Earnings Drift (PEAD) probe — the first candidate to SURVIVE (2026-07-19)

`scripts/pead_probe.py` on cached daily bars (503 S&P names, 2023-05→2026-04) +
point-in-time FMP earnings (EPS actual vs estimate, cached via
`src/data/earnings_cache.py`). Entry is the first trading day strictly AFTER the
announcement day (post-reaction, look-ahead-safe); forward drift measured to
+5/+10/+20 trading days, bucketed by EPS surprise, vs the unconditional base rate.
5,848 events across 500 tickers.

Base rate (bull window): +5d +0.37%, +10d +0.74%, +20d +1.45%.

| surprise bucket | +5d edge | +10d edge | +20d edge | N |
|---|---|---|---|---|
| big miss (<−25%) | +79 bp | +156 bp | +165 bp | 153 |
| miss | −1 | +6 | −25 | 608 |
| inline | +15 | +24 | +4 | 1,177 |
| beat | +7 | +26 | +58 | 3,232 |
| big beat (>+25%) | +16 | +75 | **+148** | 576 |

## Verdict: PEAD is present on the beat side — the first real, surviving signal

The beat side shows classic PEAD: **monotonic and horizon-growing** — bigger beat
→ larger forward drift, reaching +148 bp over base at +20d for big beats (N=576),
+58 bp for beats (N=3,232). This is the well-documented underreaction-to-earnings
anomaly, and unlike the three refuted technical signals it SURVIVES the first honest
diagnostic.

Why it's more promising than the intraday candidates:
- **Costs are negligible here** — a 20-day hold pays ~10–20 bp round-trip against a
  +148 bp edge, vs the intraday strategies where per-side cost was the whole edge.
- Large sample, monotonic in surprise magnitude, and a real economic mechanism.

## Caveats (still required before promotion)
1. **Edge vs beta.** Absolute big-beat drift is +2.93%/20d but the bull-window base
   is +1.45%, so ~half is market beta; the tradeable ALPHA is the +148 bp excess.
2. **Overlap.** 20-day holds overlap heavily → autocorrelated returns overstate
   significance. Effect size + monotonicity + N + known-anomaly prior keep it credible.
3. **Bull window** (2023–2026) — needs a sub-period split and, ideally, a bear slice.
4. The big-MISS bounce (+165 bp) is post-selloff mean-reversion, NOT drift — a
   different (contrarian) effect; irrelevant to a long-only beat strategy.

## Next step
Proper PEAD backtest through the unified exit engine: long big-beats (and beats) at
T+1 open, ~20-day hold with an ATR stop, realistic costs, concurrency-capped equity,
sub-period split. Given the diagnostic passed cleanly, this is the candidate with a
genuine shot at promotion — the first of the session.

Session tally: gap-continuation ✗, intraday-MR ✗, VWAP-momentum ✗, **PEAD ✓ (survives
diagnostic; backtest next).**

## Addendum — PEAD SURVIVES the proper backtest (2026-07-19)

`scripts/pead_backtest.py`: long an earnings beat at T+1 open (via the shared
simulate_trade / unified exit engine), 20-day hold, stop 3×ATR, target 6×ATR, net
of per-side cost, concurrency-capped equity, sub-period split. Cached data only.

Threshold sweep (net 7.5 bp/side):

| beat threshold | N | WR | avg/trade | PF | per-trade Sharpe | equity× | CAGR | eqDD |
|---|---|---|---|---|---|---|---|---|
| >2% | 3,748 | 55.3% | +1.35% | 1.58 | 1.23 | 1.10× | 3.4% | 30% |
| **>10%** | 1,558 | 57.1% | **+1.80%** | 1.73 | **1.50** | 1.76× | 21% | 21% |
| >25% | 552 | 55.2% | +1.57% | 1.59 | 1.24 | 1.31× | 10% | 12% |

- **Survives costs:** +1.20%/trade even at 15 bp/side (30 bp round-trip); a 20-day
  hold barely notices execution. (VWAP-momentum was negative before costs.)
- **Positive per-trade in ALL sub-periods:** early/mid/late = +2.11 / +1.01 / +0.93%
  — never flips negative (VWAP-momentum was negative in 2 of 3). BUT it DECAYS:
  per-trade Sharpe 2.09 → 0.85 → 0.84; the mid-third concurrency-capped equity even
  lost (−12.7% CAGR) on clustering/drawdown timing.
- **Healthy exits:** 2,367 expiry / 937 stop / 444 target — captures drift by
  holding, not by manufacturing wins with trailing stops (the sniper failure mode).

### Verdict: PEAD earns a PAPER TRIAL — the first strategy this session to clear the bar
Best variant: long beats >10%, T+1 entry, 20-day hold, 3×ATR stop. Real, cost-
surviving, sub-period-stable per-trade edge with a documented mechanism. Caveats
before production capital: (1) edge decays across the window (alpha decay / crowding /
regime — needs a longer & older, incl. bear, sample); (2) concurrency-capped equity
is sizing-sensitive because earnings cluster in seasons (concentration risk — needs a
sector/season cap in sizing); (3) 20-day overlap overstates significance. So: paper-
first on the VPS mirror, forward-tracked against this backtest, before any capital.

### Session outcome
Four candidates tested with one honest framework; three refuted, ONE survives:
gap-continuation ✗, intraday-MR ✗, VWAP-momentum ✗, **PEAD ✓ → paper trial.**
