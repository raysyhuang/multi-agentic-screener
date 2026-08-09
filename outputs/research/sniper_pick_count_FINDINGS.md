# Sniper pick-count widening — REJECTED at the live configuration (2026-08-08)

Direct follow-up to `rank_quality_FINDINGS.md`, which found the top-2-of-N choice
is close to random and noted that **summed** per-trade P&L nearly doubled from
k=2 to k=6 (+353 → +524pp). That looked like "if selection is noise, just take
more". Script: `scripts/sniper_pick_count.py`.

**It was an artifact of ignoring capital.** A real account has slots. Replayed
through the same concurrency-capped equity simulator the dashboard uses:

## At the LIVE cap (`sniper_max_positions = 3`)

| k | signals | taken | **skipped** | avg/trade | return | maxDD | Sharpe |
|---|---|---|---|---|---|---|---|
| 1 | 445 | 445 | 0 | +0.489% | +93.2% | 19.0% | 0.99 |
| **2 (live)** | 749 | 748 | **1** | +0.471% | **+180.5%** | **26.3%** | **1.18** |
| 3 | 945 | 926 | 19 | +0.366% | +146.4% | 35.0% | 0.96 |
| 4 | 1067 | 931 | 136 | +0.451% | +151.5% | 34.1% | 0.97 |
| 6 | 1212 | 933 | 279 | +0.432% | +152.5% | 33.8% | 0.97 |
| 10 | 1303 | 933 | 370 | +0.340% | +152.5% | 33.8% | 0.97 |

**k=2 is the best cell on every axis** — highest return, lowest drawdown,
highest Sharpe. Widening costs **−29pp return AND +7.8pp drawdown**.

The mechanism is visible in the `skipped` column: 1 → 136 → 279. With only 3
slots, a wider daily quota fills them with the day's marginal candidates and then
**skips genuinely better signals later**, because the slots are already occupied.
More picks does not mean more trades — `taken` plateaus at ~933 regardless of k.
It only changes *which* trades, for the worse.

## Widening only helps if the CONCURRENCY cap moves too

| cap | best k | return | Sharpe |
|---|---|---|---|
| 3 (live) | **2** | +180.5% | 1.18 |
| 5 | **4** | +135.6% | 1.23 |
| 10 | 6 (return) / 4 (Sharpe) | +63.3% / +57.8% | 1.17 / **1.25** |

So the real lever was never `max_final_picks` — it is `sniper_max_positions`.
And raising that is a materially different decision: it increases concurrent
sniper exposure, the 18%-drawdown risk profile these caps were set from assumed
3, and every drawdown in this table (19-35%) dwarfs the live book's realized
3.4%. **Not licensed by this test.**

## Verdict

**Do not widen sniper picks/day.** The live `max_final_picks=2` is, at the
current concurrency cap, the optimum on this cohort — arrived at by accident or
by good judgement, but correct either way.

Caveats that keep this a relative result: the 3Y universe under-samples sniper's
ATR%≥5 population (see `research-sniper-backtest-universe`), so absolutes are
untrustworthy; these are sniper-only fully-invested sims, not the blended book;
and 2023 is negative at every k.

## Method note worth keeping

This is the third time this session a "more/looser is better" reading died once
the test was made capital-aware or CI-aware:
- the time_stop relaxation (+0.31pp claimed → real fired stops saved money),
- the rank-ordering edge (+0.47pp claimed → CI spans zero, flips in 2025),
- and now pick-count widening (+171pp summed → −29pp compounded).

**Summed per-trade P&L is not a portfolio result.** Any future proposal that
cites a sum should be re-run through `simulate_book` before it is believed.
