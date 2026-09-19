# H4 — drift after a second consecutive big beat: REJECTED at G1 (2026-09-19)

**Verdict: REJECTED.** A second consecutive ≥10% EPS beat drifts exactly as much as a first-time one. The difference is **+0.005 percentage points**, with a date-cluster 95% CI of **[−0.53, +0.53]**. On its own, the consecutive cohort's 20-session excess is **+0.35%**, CI [−0.05, +0.74]. That fails both the +0.50% threshold and the CI-above-zero test.

## Pre-registration

The criteria are in the docstring of `scripts/h4_consecutive_beats.py`. That file was committed in `6e1faaa` and pushed at **2026-09-19T05:02:36Z**, while the wide price parquet did not yet exist. The parquet's `generated_at` is 05:06:08Z. Verify with `git log --format='%h %ad' --date=iso -- scripts/h4_consecutive_beats.py`.

H4 has a reason to exist separately from PEAD only if condition (d) holds: consecutive beats must drift *more* than first beats.

## Result (`outputs/research/h4_consecutive_beats.json`)

| Cohort | n | Entry dates | Excess 20d | Cluster 95% CI | Hit |
|---|---|---|---|---|---|
| Consecutive (previous report also ≥10%) | 6,806 | 550 | +0.35% | [−0.05, +0.74] | 48.4% |
| First (previous report <10%) | 5,563 | 552 | +0.34% | [−0.04, +0.72] | 48.3% |
| **Consecutive − first** | | | **+0.005pp** | **[−0.53, +0.53]** | |

| G1 condition | Result |
|---|---|
| (a) mean excess ≥ +0.50% | ✗ |
| (b) cluster CI lower bound > 0 | ✗ |
| (c) positive in ≥ 2 of 3 years | ✓ (3 of 4 years with n ≥ 30) |
| (d) consecutive − first, CI lower bound > 0 | ✗ |
| **PASS** | **false** |

## Reading

Neither hypothesis in the docstring wins. Anchoring would have made the second beat under-reacted to again. "Serial beaters are known and priced" would have made it drift less. The two cohorts are indistinguishable at n ≈ 6,000 each.

Both cohorts sit at +0.34 to +0.35%, which is the same level as the whole-universe raw-beat result in H1 (+0.29%). A big beat is worth about that much on this universe, whatever the previous quarter did.

## Caveats

- **Previous-report window.** The previous report must fall 60 to 130 calendar days before the current one. 1,005 of the 13,506 eligible beats had no previous report in that window and are in neither cohort. The n values in the table count only events with a complete 20-session window.
- **Same universe, prices, earnings cache and timing caveats as H1.** See `h1_pead_wide_FINDINGS.md` § 6.
- **Stage-0 only.** No stops, costs or execution model.

## Provenance

- Prices: `outputs/research/ohlcv_polygon_wide_3y.parquet`, sha256 `2d04af2db42ab0519614522a7e8cbd0f38d9532c01b26c34f4e3933e07d25e0c`. Provider `polygon`, no fallback; 2 of 4,724 requested tickers are missing.
- Earnings: FMP per-ticker cache.

## Reproduce

```bash
python scripts/h4_consecutive_beats.py --json-out outputs/research/h4_consecutive_beats.json
```
