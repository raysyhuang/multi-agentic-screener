# H5 — "good company, mispriced after a real drawdown": REJECTED at G1 (2026-09-19)

**Verdict: REJECTED.** Names 25% or more below their 52-week high that still show revenue growth ≥ 10%, positive TTM EPS and P/E ≤ 20 earn **+1.09% excess over 60 sessions** (n=1,417), with a moving-block 95% CI of **[−1.27, +3.12]**. With the corrected raw-price screen that figure is **+0.34%**, block [−1.90, +2.24] (§ Raw-price screen).

Beating the drawdown-only control was the claim that separated H5 from MR-style rebound. It came out at **+0.94pp**, with a block CI of **[−1.21, +2.77]**. Both point estimates have the hypothesised sign. Neither is established.

This was the Drift desk of the four-desk research-pipeline document. Ray chose to test the idea rather than build the desk. It now has a recorded answer.

## Pre-registration

The criteria are in the docstring of `scripts/h5_quality_drawdown.py`. That file was committed in `c0a10a0` and pushed at 2026-09-19T09:16:42Z. The run's `generated_at` is 09:17:41Z. The inputs already existed at commit time: the wide price parquet and the earnings cache with fingerprint `5262c738a5f4…`. What the timestamps prove is that no H5 computation came before the criteria.

- **Signal day S.** Each condition is evaluated at the close of S:
  - drawdown: close ≤ 0.75 × the 252-session max close, with at least 200 sessions of history;
  - revenue YoY ≥ +10%;
  - TTM EPS > 0;
  - close / TTM EPS ≤ 20.
- **Fundamentals.** Taken from the FMP per-ticker earnings cache, because FMP's quarterly key-metrics and ratios return 402 on the Starter plan. A report dated D becomes usable at the second session on or after D.
- **Cohorts.** QUALITY is all four conditions true. CONTROL is the drawdown condition true with fundamentals available, but not all quality conditions.
- **Events.** One event per episode, re-armed after 20 sessions off. Entry is the open of S+1.
- **Primary.** Excess over 60 sessions against the same-day, same-liquidity-tercile base rate.
- **G1 pass needs all four:**
  - (a) mean ≥ +1.50%;
  - (b) moving-block CI lower bound > 0;
  - (c) positive in at least 2 years with n ≥ 30;
  - (d) QUALITY − CONTROL block CI lower bound > 0.

## Result (`outputs/research/h5_quality_drawdown.json`)

| 60 sessions | n | Entry dates | Excess | Date-cluster CI | Moving-block CI |
|---|---|---|---|---|---|
| QUALITY | 1,417 | 398 | +1.09% | [−0.53, +2.69] | [−1.27, +3.12] |
| CONTROL | 5,599 | — | +0.15% | — | [−0.66, +0.94] |
| **QUALITY − CONTROL** | | | **+0.94pp** | | **[−1.21, +2.77]** |

| G1 condition | Result |
|---|---|
| (a) mean ≥ +1.50% | ✗ (+1.09%) |
| (b) block CI lower bound > 0 | ✗ |
| (c) positive in ≥ 2 years (n ≥ 30) | ✓ — 2024 +0.34% (n=337), 2025 **+2.12%** (n=673), 2026 −0.01% (n=407) |
| (d) QUALITY − CONTROL block CI lower bound > 0 | ✗ |
| **PASS** | **false** |

## Descriptive cells

These are counted as variants. None of them can rescue the verdict.

| Cell | QUALITY excess | QUALITY block CI | Minus CONTROL | Block CI |
|---|---|---|---|---|
| 20 sessions (primary definition) | +1.11% | [−0.00, +2.08] | +0.77pp | [−0.38, +1.81] |
| drawdown ≥ 35%, 60 sessions | +1.28% | [−3.12, +5.44] | +1.50pp | [−1.91, +4.71] |
| P/E ≤ 15, 60 sessions | +2.10% | [−0.35, +4.39] | +1.78pp | [−0.64, +4.02] |
| **P/E ≤ 15, 20 sessions** | **+1.64%** | **[+0.52, +2.70]** | **+1.21pp** | **[+0.08, +2.33]** |

## Reading

- **The direction holds in every cell.** Quality drawdowns beat drawdown-only names at both horizons, at both drawdown depths and at both P/E caps. They beat the matched base rate too. That consistency is why H5 was worth testing. It is not the same as an established effect.
- **Most of the 60-session result comes from one year.** 2025 contributes +2.12% on n=673; 2024 and 2026 are near zero. A single-year edge is the pattern that has repeatedly failed forward in this registry.
- **The cheaper cut cleared its CIs under the adjusted screen, but only as a post-hoc result.** At P/E ≤ 15 over 20 sessions, both the level and the control difference excluded zero. It is one of seven descriptive cuts, on the wrong horizon, chosen after the results were visible. Under the corrected raw screen its difference from CONTROL no longer clears zero (§ Raw-price screen). It is kept as an observation, not a lead.
- **The window is short.** The 252-session lookback means events begin in mid-2024, so this is roughly two calendar years of events. The pre-registration stated this caveat in advance.

## Raw-price screen (v3; `h5_quality_drawdown_rawscreen.json`)

While running H3 we found that the $5 floor was applied to split-adjusted prices. That let in penny stocks which later reverse-split; see `h3_dividends_splits_FINDINGS.md` § 3. With the floor corrected:

| Cell | Adjusted screen | **Raw screen** |
|---|---|---|
| QUALITY, 60 sessions (primary) | +1.09%, block [−1.27, +3.12] | **+0.34%, block [−1.90, +2.24]**; minus CONTROL +0.48pp, block [−1.58, +2.27] |
| QUALITY by year, 60 sessions | 2024 +0.34, 2025 +2.12, 2026 −0.01 | 2024 −0.59, 2025 +1.04, 2026 −0.07 |
| P/E ≤ 15, 20 sessions (the § Reading lead) | +1.64%, block [+0.52, +2.70]; minus CONTROL block [+0.08, +2.33] | +1.33%, block [+0.22, +2.40]; **minus CONTROL block [−0.09, +2.14]** |

The verdict is unchanged. The P/E ≤ 15 lead no longer beats the drawdown-only control.

## Caveats

- **EPS and split adjustment.** P/E divides a split-adjusted price by FMP's reported EPS. If FMP's historical EPS is not split-adjusted, the P/E of any name that split inside the window is off by the split ratio. This was not verified, and it affects few names.

- **Look-ahead in the fundamentals.** None by construction: a report is usable from the second session on or after its date. EPS is FMP's reported actual (adjusted, not GAAP). P/E uses TTM adjusted EPS.
- **Universe and timing.** The same survivorship-reduced universe as H1 (`h1_pead_wide_FINDINGS.md` § 6). Current index membership is not used.
- **Stage 0 only.** No stops, no costs.

## Provenance

- **Prices.** `ohlcv_polygon_wide_3y.parquet`, sha256 `2d04af2db42ab0519614522a7e8cbd0f38d9532c01b26c34f4e3933e07d25e0c`. Provider `polygon`, no fallback; 2 of the 4,724 requested tickers are missing.
- **Earnings.** 4,721 files, fingerprint `5262c738a5f4…`.

## Reproduce

```bash
python scripts/h5_quality_drawdown.py --json-out outputs/research/h5_quality_drawdown.json
```
