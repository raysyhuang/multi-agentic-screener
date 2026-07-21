# Gap-Continuation Alpha — Findings (2026-07-19)

First Phase-4 candidate. Data: 503-ticker S&P 500, 2023-05 → 2026-04 (cached
parquet). All runs through the unified exit engine with realistic gap-through
fills + a concurrency-capped equity curve (`scripts/gap_continuation_research.py`).

Thesis: a stock that gaps up ≥3% on heavy volume and closes near its high tends
to continue higher. Enter T+1 open, ATR stops/targets/trail.

## Verdict: NO robust tradeable alpha on daily bars. Do NOT promote.

### 1. Stop-based configs all lose
Every parameter config (gap threshold, target, hold, trend filter, close-position
filter) lands at zero-to-negative expectancy:

| config | N | WR | avg% | PF | Sharpe | equity× |
|---|---|---|---|---|---|---|
| baseline | 896 | 57.9% | −0.15 | 0.88 | −0.31 | 0.88 |
| gap_min=5 | 431 | 58.0% | −0.02 | 0.98 | −0.05 | 1.01 |
| hold=5 | 896 | 58.0% | −0.12 | 0.90 | −0.26 | 0.90 |
| tight close≥0.7 | 644 | 58.4% | −0.15 | 0.88 | −0.32 | 0.86 |

Signature everywhere: **~58% win rate but payoff < 1.** The entry has a slight
directional edge; ATR-width stops (~1.5×ATR ≈ 3–4% for these names) throw it away.
Changing `target_atr` does nothing — targets are almost never reached.

### 2. The entry edge is real but tiny (forward-drift diagnostic)
Raw forward returns from T+1 open, no stops, vs the universe base rate:

| horizon | signal avg | base avg | edge |
|---|---|---|---|
| 1d | +0.112% | +0.071% | +4 bps |
| 3d | +0.350% | +0.218% | **+13 bps** |
| 5d | +0.372% | +0.366% | +0.6 bps |
| 10d | +1.191% | +0.736% | +45 bps (mostly bull beta) |

There IS a ~13-bps excess drift over 3 days, but it's far too thin to survive
ATR stops + slippage — which is exactly why every stop-based config lost.

### 3. Drift-capture (time exit, no stops) — best config is marginal and suspect
Monetizing drift correctly (exit at close on a time stop, catastrophe stop only):

| config | N | WR | avg% | PF | Sharpe | equity× | CAGR | DD |
|---|---|---|---|---|---|---|---|---|
| drift hold=3 (all) | 896 | 51.4% | +0.07 | 1.04 | 0.09 | 0.87 | −4.9% | 29.7% |
| **drift hold=2, gap≥5** | 431 | 54.3% | +0.26 | 1.18 | **0.41** | 1.14 | +4.9% | 11.5% |

The `gap≥5 / hold=2` config is weakly positive (Sharpe 0.41, +4.9% CAGR). **But
it is the best of ~20 configs tried** — after a deflated-Sharpe / multiple-testing
correction it is almost certainly not significant, it is mostly bull-market beta,
and it is unvalidated out-of-sample. Promoting it would repeat the exact data-mining
that produced the 82% sniper illusion. Classify as WATCH, not promote.

## Conclusion & next step

Daily gap-continuation is not a tradeable strategy. The genuine edge (short-term
post-gap drift) is small and lives at a resolution daily bars can't exploit —
concrete, evidence-backed justification for wiring the **Polygon $199 minute bars
(Phase 3)** before pursuing the momentum-continuation family further. Re-test the
`gap≥5 / 2-day drift` config at minute resolution AND out-of-sample before any
paper trial.

The other two Phase-4 candidates remain data-gated: intraday mean-reversion needs
minute bars (VWAP deviation is meaningless daily); post-earnings drift needs
point-in-time earnings dates.

## Addendum (2026-07-19) — intraday probe REFUTES the "edge lives intraday" guess

With the Polygon $199 minute feed now working, `scripts/gap_intraday_probe.py`
pulled real minute bars for the T+1 entry day of 149 sampled gap signals and
built the average intraday return path from the regular-session open (timezone
converted UTC→America/New_York, DST-correct):

| checkpoint | avg % | median % | % up |
|---|---|---|---|
| open+15m | −0.08 | −0.07 | 46% |
| +1h | −0.19 | −0.13 | 47% |
| +2h (midday) | −0.38 | −0.41 | 43% |
| close | −0.20 | −0.27 | 45% |

The entry day **fades**: price drifts down from the open, bottoms midday, recovers
slightly into the close but ends negative (−0.20% avg, only ~44% of days up). An
opening-buy / hold-intraday rule loses money. **The momentum-continuation thesis is
refuted at the very resolution where I hypothesized it might live.** The tiny
multi-day drift the daily study found accrues overnight (gap→next-open), not in the
session.

**Verdict: gap-continuation is dead at BOTH daily and intraday resolution. Retire it.**

Silver lining / direction: the open→midday fade-then-recover shape is an intraday
MEAN-REVERSION pattern, not continuation — i.e. the minute data points toward
**candidate 2 (intraday mean-reversion)** as the more promising family to test next,
now that the minute pipeline is proven. (149-signal probe, one window — directional,
not a validated strategy.)
