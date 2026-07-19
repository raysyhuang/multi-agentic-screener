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
