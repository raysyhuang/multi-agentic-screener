# H5 — "good company, mispriced after a real drawdown": REJECTED at G1 (2026-09-19)

**Verdict: REJECTED.** The quality cohort — names at least 25% below their 52-week high that still show revenue growth ≥ 10%, positive TTM EPS and P/E ≤ 20 — earns a **positive but unestablished** excess return:

| 60 sessions (primary) | Adjusted-price screen | **Raw-price screen** |
|---|---|---|
| QUALITY | +1.42% (n=1,347), block [−0.95, +3.41] | **+0.56%** (n=1,345), block [−1.70, +2.47] |
| QUALITY − CONTROL | +0.98pp, block [−1.19, +2.84] | +0.57pp, block [−1.58, +2.44] |

What separates this idea from MR-style rebound is that it must beat a drawdown-only control. It does not beat the control with a CI clear of zero under either screen, and most of the effect comes from a single year (§ Result).

This was the Drift desk of the four-desk research-pipeline document. Ray chose to test the idea rather than build the desk; this is the result.

## Pre-registration

The criteria are in the docstring of `scripts/h5_quality_drawdown.py`, committed in `c0a10a0` and pushed at 2026-09-19T09:16:42Z. The inputs already existed at that point: the wide price parquet and the earnings cache with fingerprint `5262c738…`. The push time therefore proves only that no H5 computation came before the criteria.

- **Signal day S**, evaluated at S's close:
  - close ≤ 0.75 × the 252-session max close, with at least 200 sessions of history;
  - revenue YoY ≥ +10%;
  - TTM EPS > 0;
  - close / TTM EPS ≤ 20.
- **Fundamentals** come from the FMP per-ticker earnings cache, because FMP's quarterly key-metrics and ratios return 402 on the Starter plan. A report dated D becomes usable at the second session on or after D.
- **Cohorts.** QUALITY = all four conditions true. CONTROL = drawdown true and fundamentals available, but the quality conditions not all met.
- **Events.** One per episode, re-armed after 20 sessions off. Entry is S+1's open.
- **G1 conditions**, all required:
  - (a) QUALITY mean 60-session excess ≥ +1.50%;
  - (b) moving-block CI lower bound > 0;
  - (c) positive in at least 2 years with n ≥ 30;
  - (d) QUALITY − CONTROL block CI lower bound > 0.

**Amendment after the Codex review of PR #123.** The criteria are unchanged; the implementation now matches its registered text. The first run had two defects:
- It treated report *position* as a fiscal quarter. When a quarter was missing, "YoY" silently spanned two years and "TTM" six quarters. For example, STLA on 2024-05-02 got a P/E of 1.7 this way.
- It checked only the two endpoint revenues, although the registered text requires all five to be positive.

Reports k−4 through k must now be consecutive quarters: every gap between reports is 45–140 days, and report k is 300–430 days after report k−4. All five revenues must also be positive. Every number in this document comes from the corrected code.

## Result

Adjusted-price screen (`h5_quality_drawdown.json`):

| G1 condition | Result |
|---|---|
| (a) QUALITY ≥ +1.50% | ✗ (+1.42%) |
| (b) block CI lower bound > 0 | ✗ ([−0.95, +3.41]) |
| (c) positive in ≥ 2 years | ✓ 3/3: 2024 +0.16%, 2025 **+2.72%**, 2026 +0.31% |
| (d) QUALITY − CONTROL block lower bound > 0 | ✗ ([−1.19, +2.84]) |
| **PASS** | **false** |

Raw-price screen (`h5_quality_drawdown_rawscreen.json`): +0.56%. Condition (c) also fails here, 1/3: 2024 −0.76%, 2025 +1.63%, 2026 −0.12%. The raw screen fixes a look-ahead in the $5 floor, which had admitted penny stocks that later reverse-split; see `h3_dividends_splits_FINDINGS.md` § 3.

## Descriptive cells

These are variants and cannot rescue the verdict. Each cell reads: QUALITY excess, its block CI; QUALITY − CONTROL, its block CI.

| Cell | Adjusted screen | Raw screen |
|---|---|---|
| 20 sessions (primary definition) | +1.17%, [+0.04, +2.15]; +0.73pp [−0.48, +1.83] | +0.82%, [−0.31, +1.80]; +0.58pp [−0.60, +1.65] |
| drawdown ≥ 35%, 60 sessions | +1.94%, [−2.63, +6.03]; +1.67pp [−1.75, +4.73] | +1.18%, [−3.38, +5.20]; +1.14pp [−2.42, +4.29] |
| P/E ≤ 15, 60 sessions | +2.57%, [+0.22, +4.67]; +1.97pp [−0.33, +4.04] | +1.70%, [−0.54, +3.78]; +1.56pp [−0.74, +3.68] |
| **P/E ≤ 15, 20 sessions** | **+1.79%, [+0.64, +2.90]; +1.25pp [+0.03, +2.45]** | **+1.41%, [+0.25, +2.51]; +1.08pp [−0.11, +2.26]** |

## Reading

- **The direction holds in every cell.** Quality drawdowns beat drawdown-only names at both horizons, at both drawdown depths, at both P/E caps and under both screens. That consistency is why the idea deserved a test, but it is not an established effect.
- **Most of the effect is 2025.** Under the raw screen, 2024 and 2026 are negative. A single-year edge is the pattern that has repeatedly failed forward in this registry.
- **The cheaper cut is the strongest cell, and it is post-hoc.** At P/E ≤ 15 over 20 sessions, the level clears zero under both screens. Its difference from CONTROL clears zero only under the adjusted screen. It is one of seven descriptive cuts, on a non-primary horizon, chosen after the results were visible. It is recorded as an **observation**, not a lead. Taking it further would need its own pre-registration and data this study did not see.
- **The window is short.** The 252-session lookback puts the first events in mid-2024, so the sample covers about two calendar years. This was stated in advance.

## Final review: the control is not depth-matched

The registry text calls for a control with *equal* drawdown. As run, CONTROL met the same ≥ 25% threshold but was not matched on depth, and QUALITY's drawdowns were deeper:

| Mean peak ratio (raw screen) | Mean | Median |
|---|---|---|
| QUALITY | 0.673 | 0.725 |
| CONTROL | 0.704 | 0.736 |

When CONTROL is reweighted to QUALITY's drawdown-depth distribution (Codex, final review):
- CONTROL's 60-session excess moves from −0.01% to **+0.27%**.
- QUALITY − CONTROL shrinks from +0.57pp to **+0.29pp**.

Part of the "quality" advantage is therefore simply "deeper drawdown". The verdict (REJECTED) is unchanged, and the mechanism estimate is confounded. A retest should match on depth.

## Caveats

- **EPS basis.** P/E divides a split-adjusted price by FMP's reported EPS. Codex spot-checked NVDA, AVGO, SMCI and CMG and found historical EPS is split-adjusted, so the two are consistent. That check does not certify every ticker.
- **Look-ahead in the fundamentals.** A report is used only from the second session on or after its **recorded** date. Both that date and the report contents come from today's FMP cache, not point-in-time vintages, so later revisions to report dates, actuals or estimates could leak in; this is unquantified. EPS is FMP's adjusted actual, not GAAP. The P/E numerator is a split-adjusted price, and the code does not verify that every ticker's EPS is split-adjusted too.
- **Universe.** The same survivorship-reduced universe as H1 (`h1_pead_wide_FINDINGS.md` § 6). Current index membership is not used.
- **Scope.** Stage 0 only: no stops, no costs.

## Provenance

- **Prices.** `ohlcv_polygon_wide_3y.parquet`, sha256 `2d04af2db42ab0519614522a7e8cbd0f38d9532c01b26c34f4e3933e07d25e0c`.
- **Earnings.** Fingerprint `5262c738a5f4…`.
- **Splits** (raw screen only). `data/cache/corp_actions/splits.json`, sha256 `3319d67b…`.

## Reproduce

```bash
python scripts/h5_quality_drawdown.py --json-out outputs/research/h5_quality_drawdown.json
python scripts/h5_quality_drawdown.py --raw-price-screen --json-out outputs/research/h5_quality_drawdown_rawscreen.json
```
