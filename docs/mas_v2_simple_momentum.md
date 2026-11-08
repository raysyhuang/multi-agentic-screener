# MAS v2 — Simple Momentum

Status: **baseline rejected; not an official pick or production strategy**.

## Why reset

The legacy MR and Sniper strategies did not establish reliable selection alpha.
They remain in the repository for audit history and shadow observation, but MAS
v2 does not combine with them, average their scores, or attempt to preserve the
old strategy identity.

## The complete strategy

1. Universe: the certified point-in-time S&P 500 membership on signal day.
2. Signal: last market session of each month.
3. Score: close 21 sessions ago / close 252 sessions ago - 1.
4. Eligibility: price at least $5 and sufficient observed history. Missing data
   is excluded, never imputed.
5. Portfolio: top 20, equal weight.
6. Execution: next market session; rebalance only the required difference.
7. Cost: 10 bp per side on actual traded value.
8. Exit: the next monthly rebalance. No stop, target, trail, market-regime gate,
   quality score, valuation score, or discretionary override.

That is the entire economic claim: medium-term winners tend to continue winning.

## What is deliberately not inherited

- the generic MAS 0-100 ranker;
- confluence bonuses and cross-model deduplication;
- MR/Sniper ATR exits;
- bull/bear/choppy multipliers;
- the current live FMP universe replayed into history;
- trade-return sums presented as account drawdown.

## Required test before activation

Use the same certified PIT dataset and account engine as the independent momentum
study. Report 2005-2015 and 2015-2025 separately, yearly results, daily equity,
CAGR, daily Sharpe, maximum drawdown, turnover, time invested, SPY excess, and
excess over an equal-weight portfolio of the same eligible universe.

The result artifact must include code SHA, PIT vintage ID, data hashes, this rule
object, every monthly target portfolio, and the daily equity curve.

Activation requires positive net excess over the equal-weight universe in both
ten-year windows. A positive point estimate with a confidence interval including
zero remains research-only.

## Change discipline

This document freezes the baseline before its MAS-native result is observed.
Changing Top20, 12-1, the monthly cadence, price floor, or cost creates a new
registered variant. Quality and value may later be tested as separate filters;
they are not bundled into v1.

## First frozen read — 2026-11-08

The rule above was evaluated once on the existing frozen public PIT dataset,
with 10 bp per side and twelve start-month paths. Eligibility was measured at
the month-end signal date; execution still required a real price on the next
session. These are price-return results:

| Window | Momentum CAGR, median | SPY CAGR, median | Momentum max drawdown |
|---|---:|---:|---:|
| 2005-2015 | 4.99% | 5.53% | -60.97% |
| 2015-2025 | 10.16% | 11.41% | -39.40% |

The baseline trails SPY in both decades and has worse drawdown. It therefore
fails before the still-required equal-weight-universe comparison and confidence
interval. It must not be activated.

This result does not establish that momentum is absent. It rejects this exact
standalone implementation as the new MAS core. No parameter is changed in this
document after seeing the result.
