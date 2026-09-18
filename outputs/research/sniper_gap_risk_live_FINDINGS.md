# Sniper ex-ante gap-risk filter on the LIVE cohort — FINDINGS (2026-09-18)

## Pre-registered read (stated before the run)

> the filter is worth a paper re-entry test only if a single threshold removes ≥5 of the 7 `time_stop` losers AND ≤3 of the 25 `trail_stop` exits, AND that same threshold is stable by year on the backtest cohort. n=32 → descriptive only, no decision.

## Verdict: **FAIL — no threshold comes close.**

The best any threshold does is remove **2 of 7** `time_stop` losers, and only by also removing **7 to 10 of 25** `trail_stop` exits (live p70 4.7%: 2 time_stops / 7 trail_stops; backtest p80 4.7%: 2 / 8; backtest p70 4.2%: 2 / 10). Every other threshold removes **0** time_stops. Exhaustively: across all 32 possible cutoffs of the stored gap-vol scores, none removes ≥5 time_stops while removing ≤3 trail_stops. The 7 day-1 gap-throughs that account for the whole live sniper loss are **not** gappy names by the trailing-60d measure: 5 of the 7 sit *below* the live cohort's median gap-vol, and the calmest quintile holds 2 of them. The gappiest live quintile is the cohort's *best* bucket (WR 75%, +1.60%/trade). Filtering on trailing gap-vol makes the live cohort **worse** at every threshold tried.

This is a mechanism finding, not a tuning miss: the losses come from names whose overnight behaviour looked ordinary before entry. Trailing gap-vol is not the ex-ante variable. **The gap-risk lever is closed for sniper unless a different ex-ante feature is proposed and pre-registered.**

Consistent with the backtest cohort (n=1357): quintile expectancy is non-monotonic in gap-vol and no threshold raises the average (p70 +0.234 vs baseline +0.236), so the live result is not a small-n surprise.

## Question

Sniper's live loss (n=32, −0.34%/trade) is entirely 7 `time_stop` exits at −6.46% avg (MAE −7.9%, MFE +0.24%) — day-1 gap-throughs. Does `gap_vol(ticker, asof)` = trailing-60d 90th-pct |open/prev-close − 1| (bars strictly before the signal date) identify them ex ante without removing the 25 `trail_stop` exits (+1.37% avg)?

## Live cohort — `sniper|mas_official`, 32 closed trades

Exit mix: 25 `trail_stop`, 7 `time_stop`. Baseline WR 40.6%, avg −0.344%. All 32 rows scored (feature available for 32/32).

Expectancy by gap-vol quintile (Q1 calmest):

| bucket | gap_vol range | N | WR | avg % | time_stops in bucket |
|---|---|---|---|---|---|
| Q1 | 2.2–2.9% | 6 | 33% | −1.64 | 2 |
| Q2 | 3.1–3.5% | 6 | 17% | −2.49 | 3 |
| Q3 | 3.5–4.0% | 6 | 50% | −0.26 | 0 |
| Q4 | 4.1–5.3% | 6 | 17% | +0.41 | 1 |
| Q5 | 5.4–8.2% | 8 | 75% | +1.60 | 1 |

Filter "drop gap_vol > threshold". Backtest-derived thresholds are what a live filter would actually be set from; live percentiles are shown for comparison only.

| threshold | kept | dropped | dropped `time_stop` | dropped `trail_stop` | kept WR | kept avg % | dropped avg % |
|---|---|---|---|---|---|---|---|
| backtest p95 (6.7%) | 30 | 2 | 0 | 2 | 36.7% | −0.49 | +1.84 |
| backtest p90 (6.0%) | 27 | 5 | 0 | 5 | 33.3% | −0.59 | +0.97 |
| backtest p80 (4.7%) | 22 | 10 | 2 | 8 | 31.8% | −0.72 | +0.48 |
| backtest p70 (4.2%) | 20 | 12 | 2 | 10 | 30.0% | −1.51 | +1.60 |
| live p95 (7.7%) | 31 | 1 | 0 | 1 | 38.7% | −0.43 | +2.38 |
| live p90 (6.1%) | 29 | 3 | 0 | 3 | 34.5% | −0.54 | +1.59 |
| live p80 (5.4%) | 26 | 6 | 0 | 6 | 30.8% | −0.87 | +1.94 |
| live p70 (4.7%) | 23 | 9 | 2 | 7 | 30.4% | −0.79 | +0.80 |

Read across: the dropped set has a *positive* average at every threshold, the kept set gets worse at every threshold. The pre-registered bar (≥5 of 7 time_stops, ≤3 of 25 trail_stops) is not met by any row.

## Backtest cohort (reference thresholds + stability), n=1357

Live sniper config (`SNIPER_ENTRY`: min_score 70, ATR% floor 5, stop 1.5×/target 3.0×ATR, hold 7, time_stop 1d, trail 0.5/0.3, gap-through fills) on `ohlcv_3y_cache.parquet` (501 tickers). Baseline WR 52.4%, avg +0.236%, worst-5% sum −715%.

| quintile | gap_vol range | N | WR | avg % | worst-5% sum |
|---|---|---|---|---|---|
| Q1 | 1.1–2.8% | 271 | 55.0% | +0.50 | −115 |
| Q2 | 2.8–3.5% | 271 | 56.1% | +0.21 | −149 |
| Q3 | 3.5–3.9% | 271 | 48.3% | −0.18 | −128 |
| Q4 | 3.9–4.7% | 271 | 50.6% | +0.23 | −132 |
| Q5 | 4.7–9.6% | 273 | 52.0% | +0.41 | −167 |

| threshold | kept | dropped | WR | avg % | worst-5% | yr1 | yr2 | yr3 |
|---|---|---|---|---|---|---|---|---|
| p95 (6.7%) | 1295 | 62 | 52.5% | +0.215 | −688 | +0.077 | +0.201 | +0.252 |
| p90 (6.0%) | 1225 | 132 | 52.6% | +0.209 | −649 | +0.077 | +0.172 | +0.255 |
| p80 (4.7%) | 1092 | 265 | 52.4% | +0.186 | −541 | +0.057 | +0.175 | +0.224 |
| p70 (4.2%) | 950 | 407 | 52.8% | +0.234 | −466 | −0.003 | +0.274 | +0.289 |
| baseline | 1357 | — | 52.4% | +0.236 | −715 | +0.077 | +0.332 | +0.224 |

No threshold improves the average; the worst-5% sum shrinks only because fewer trades are taken. Nothing here is stable enough to carry a filter.

## What this does and does not say

- It says trailing overnight-gap volatility is not the ex-ante signature of sniper's day-1 gap-throughs, live or in backtest.
- It does not say the gap-throughs are unpredictable — only that this feature does not predict them. Any replacement feature needs its own pre-registered read before it is run on the live cohort (n=32 will "find" something otherwise).
- n=32 is descriptive. The bar was set so a pass would be a reason to *test* on paper, never to change the book; the fail means the shadow-stream decision for sniper proceeds on the existing evidence stack, not on a pending filter.

## Provenance

- **Live cohort input:** `outputs/research/frozen/data-2026-09-17T230250Z.json`, sha256 `dbd9e336c524a924ecf348db74e02246965d7b4dcc7c0061a2d59cfc78f51ef6`, 255,290 bytes, `generated_at` 2026-09-17T23:02:50Z, window_days 90 (manifest: `MANIFEST-2026-09-17T230250Z.json`).
- **Live prices:** `fetch_ohlcv(..., source="polygon", strict=True, no_cache=True)`, 2025-03-20 → 2026-09-18. `get_last_ohlcv_provenance()` = `{"provider": "polygon", "fallback_reason": null, "requested": 26, "returned": 26, "missing": [], "failures": {}}` (25 distinct tickers across the 32 rows, + SPY = 26 requests).
- **Backtest cohort:** `outputs/research/ohlcv_3y_cache.parquet` (501 tickers, 2023-07 → 2026-07; gitignored, not in the PR). No beside-file provenance manifest exists for this cache; it predates the manifest convention.
- **Machine-readable:** `outputs/research/sniper_gap_risk_live.json` (both cohorts' tables, the live per-row scores, provenance dict, input sha).

## Reproduce

```bash
PYTHONPATH=. python scripts/sniper_gap_risk.py \
  --cohort outputs/research/frozen/data-2026-09-17T230250Z.json \
  --cache-file outputs/research/ohlcv_3y_cache.parquet \
  --json-out outputs/research/sniper_gap_risk_live.json
python -m pytest tests/test_sniper_gap_risk_cohort.py -q
```

The default path (no `--cohort`) keeps its analysis formulas and printed formats unchanged, with one exception: on `origin/main` it crashes with a `TypeError` in the per-year stability split (a `datetime.date` window compared against the pandas `Timestamp` signal date; pandas ≥ 2 refuses the comparison). This PR normalises the signal date to a plain `date` in `run_backtest_cohort` (`scripts/sniper_gap_risk.py`), so the default run now completes; its numbers are the backtest-cohort tables above.

**Pre-registration timing:** the read in § "Pre-registered read" was stated in the session plan before the run, but the read and the result landed in the same commit, so git history cannot show that it preceded the run. Treat the timing as asserted, not independently verifiable. The arithmetic does not depend on it: no cutoff of the 32 stored scores satisfies the rule (see Verdict).
