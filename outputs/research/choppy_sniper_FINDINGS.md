# Choppy sniper ×0.6 — CANDIDATE, not licensed. Shadow-track it. (2026-08-08)

Tier-1 item 2.3. Live has taken **0 sniper picks across 10 choppy runs**: the
ranker's `REGIME_MULTIPLIERS[choppy][sniper] = 0.6` acts as a hard switch
(sniper 77×0.6 = 46 vs MR 85×1.1 = 94), so MR takes every choppy slot. The review
argued Run E shows choppy is sniper's *best* regime and the multiplier
contradicts the engine's own data. Script: `scripts/choppy_sniper_regime_test.py`.

## First: the review's comparison measured the wrong variable

The `regime` column in the truth-matrix cohorts is `classify_regime(df)` over
**the ticker's own price history** — a per-stock trend label. The live gate uses
the **market** regime from SPY/QQQ. Different variables; the review's numbers
were not measuring what the multiplier controls.

Re-stamped with the SPY-based market regime per entry date:

| cohort | bear | bull | **choppy** |
|---|---|---|---|
| Sniper (Run E) | +0.394% (n=214) | +0.125% (n=846) | **+0.604% (n=275)** |
| MR (3Y) | +0.497% (n=4810) | −0.069% (n=17816) | **−0.016% (n=5196)** |

Sniper choppy per-year: 2024 +0.18 / 2025 +0.23 / 2026 +1.16 — **positive in
every year**, which is more than most candidates this month managed.

- Sniper choppy − MR choppy = **+0.619pp, 95% CI [+0.067, +1.165] — significant.**
- The review's actual claim (choppy is sniper's best regime): choppy − bull =
  +0.479pp, CI [−0.190, +1.126] — **not significant.** Directionally lucky,
  not established.

## Then: the confound that decides it

Both cohorts above are **unselected** populations. The live pipeline runs a
selection funnel, and for MR that funnel is worth about a full percentage point:

| | backtest (unselected) | live (through the funnel) |
|---|---|---|
| MR choppy | −0.016% (n=5196) | **+0.972% (n=16)** |
| Sniper choppy | +0.604% (n=275) | **no data — never traded** |

So the real question is not "unselected sniper vs unselected MR" (sniper wins,
significantly). It is "**selected** sniper vs **selected** MR" — and live MR
choppy at +0.972% is already *above* unselected sniper's +0.604%. Whether
sniper's funnel gives it a comparable lift is unmeasured, because the multiplier
has ensured sniper never trades in choppy.

## Verdict: do not flip the multiplier. Shadow-track instead.

Flipping it would displace a live stream running +0.972%/trade on the strength of
a comparison that ignores the selection step worth ~1pp to its incumbent. That is
the same error class as every rejected candidate this session — a real-looking
effect measured on the wrong population.

The right move is to generate the missing data. **PR #54 shipped exactly the
mechanism**: persist choppy sniper candidates with a shadow `skip_reason` so they
are tracked and evaluated but never traded and never counted in the official
record. After ~30 shadow choppy sniper trades the comparison becomes
selected-vs-selected and the decision makes itself.

Choppy is ~16% of runs (10 of the last 64), so expect roughly a quarter's worth
of data — slow, but the alternative is guessing with live capital.

## Standing caveats

- Sniper absolutes from the 3Y cohort are untrustworthy (universe under-samples
  ATR%≥5 — see `research-sniper-backtest-universe`); relative arms only.
- Live MR choppy is n=16. It is the incumbent, but it is not a settled number
  either.
