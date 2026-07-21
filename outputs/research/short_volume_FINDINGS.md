# Short-volume-ratio probe (untried edge) — no tradeable edge (2026-07-19)

`scripts/short_volume_probe.py` on real Polygon daily short-volume (FINRA) for 502
S&P names, 2023-05→2026-04. Bucket every (ticker, day) by short_volume_ratio;
forward 5/10/20-day return vs base rate. Look-ahead-safe. **276,362 observations.**

Base rate: +5d +0.37%, +10d +0.74%, +20d +1.45%.

| short-volume ratio | +5d edge | +10d edge | +20d edge | N |
|---|---|---|---|---|
| <35% | +8 bp | +7 bp | +4 bp | 38,581 |
| 35-45% | −1 | −4 | −13 | 57,130 |
| 45-50% | −4 | −10 | −19 | 35,271 |
| 50-55% | −3 | −11 | −21 | 35,449 |
| 55-65% | −8 | −13 | −31 | 61,762 |
| ≥65% | −7 | −10 | **−38** | 45,670 |

## Verdict: no tradeable edge. Mildly (but trivially) bearish in high short volume.

The pattern is monotonic-ish and the OPPOSITE of a squeeze: more short volume →
slightly WORSE forward returns (short selling is weakly informed). But the effect is
tiny — ≤38 bp over 20 days (~2 bp/day), noise-level against a +1.45%/20d base and
gone after costs. For a long-only book the low-short-volume bucket (+4 bp/20d) is
trivially small.

Note: an 8-ticker smoke run had suggested a +99 bp/20d contrarian squeeze in the
≥65% bucket — that was pure small-sample noise, erased by the full 502-ticker
universe. Same probe→confirm discipline that killed VWAP-momentum.

Untried-edge tally so far: short-volume ratio ✗. Short-interest days-to-cover
(bi-monthly squeeze factor, distinct data) and small/mid-cap universe still open.
Read-through remains: on efficient large-caps, price/flow technical signals don't
carry tradeable edge — the one survivor (PEAD) is an event edge, not a technical one.

---

## Addendum — days-to-cover (short interest squeeze) also null (2026-07-19)

`scripts/days_to_cover_probe.py` on Polygon bi-monthly short interest, 502 tickers,
95,043 settlement observations, publication-lag corrected (enter settlement +8
trading days, since FINRA publishes ~8 business days late — entering at settlement
is look-ahead).

| days-to-cover | +5d edge | +10d edge | +20d edge | N |
|---|---|---|---|---|
| 1-2 | +117 bp | −15 bp | +221 bp | 23,326 |
| 2-3 | +56 | −98 | +81 | 29,871 |
| 3-5 | +41 | −114 | +82 | 27,526 |
| 5-8 | +22 | −154 | +80 | 9,765 |
| ≥8 | +11 | −235 | +162 | 4,058 |

**Verdict: no coherent edge.** The sign FLIPS across horizons (+5d weakly favors low
short interest, +10d uniformly negative, +20d uniformly positive and non-monotonic) —
the signature of noise, not a factor. A real squeeze/informed-short effect would be
stable and monotonic.

## Untried-edge tally (short data): both null
- Short-volume ratio ✗ (tiny, weakly bearish, untradeable)
- Days-to-cover / short interest ✗ (incoherent across horizons)

Both confirm the session read-through: on efficient large-caps, price/flow/short
technical factors do not carry tradeable edge. Remaining untried: options flow
(IV/OI — likely also efficient on mega-caps, heavier to test) and a small/mid-cap
universe (the one genuinely less-efficient population — the real remaining
hypothesis, but needs a dedicated data pull).
