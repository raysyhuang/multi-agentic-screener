# Pre-registration — momentum + quality core (monthly, long-only)

Registry row: `R-2026-10-momentum-quality-core`. Written 2026-10-05, **before any data for this hypothesis has been pulled or looked at**. Changing anything below after data exists requires a new registry row and an incremented variant count. That is the point of writing it now.

**Status: BLOCKED on data.** It runs only on a certified point-in-time universe (`outputs/research/PIT_RULINGS_REQUEST_2026-10.md`). Running it on today's universe replayed backwards is prohibited: that is the survivorship defect that inflated the PEAD sleeve's evidence and the Reclaim replay.

## G0 — who is on the other side, and why do they lose

- **Momentum (12-1):** investors under-react to gradual information, and disposition-prone holders sell winners early, so prices adjust over months rather than days. The counterparty is the early seller and the anchored buyer. This is a large, persistent premium in the published record across markets and decades. It is not a pattern found in our own data.
- **Profitability screen:** momentum's worst crashes concentrate in unprofitable, lottery-like names that attention-driven buyers overpay for. Requiring positive trailing earnings is a crash filter, not a second alpha claim.
- **Why it suits this book:** monthly turnover makes 10 bp/side costs nearly irrelevant, and it uses daily closes only. There's no intraday fill problem, no gap-through stop and no score-tiered exit, which is where every daily model here has bled.

## Primary specification (P), the only arm that decides

| Item | Rule |
|---|---|
| Formation date F | last XNYS session of each month |
| Universe | PIT-certified eligible set on F, estimated mcap ≥ $1B (contract §3c), close ≥ $5, ≥ 273 sessions of history as of F |
| Momentum score | close(F − 21 sessions) / close(F − 252 sessions) − 1 (skips the most recent month) |
| Quality screen | the four most recent quarterly `epsActual` with report date ≤ F sum to > 0 (report-dated FMP earnings cache). Fewer than four known quarters → excluded, never imputed |
| Portfolio | top **50** by momentum among screened names, equal weight |
| Execution | enter at the open of the first session after F; hold to the open of the first session after the next F; no stops, no targets, no trail |
| Costs | 10 bp per side on traded weight (`settings.slippage_pct`) |
| Delisting | a holding that stops trading exits at its last close, then takes a **−30%** delisting return (primary; Shumway 1997's performance-delisting estimate). **0% and −100%** are reported as bounds |
| Prices | split-adjusted, price-only. No dividends anywhere, consistent with MAS (contract §11) |
| Benchmark B1 (decides) | equal-weight portfolio of the same filtered universe (mcap / price / history, no momentum, no quality screen), same rebalance, no costs |
| Benchmark B2 (reported) | SPY, price-only |

**Primary statistic:** monthly net excess return of P over B1.

## Pass / fail, all pre-registered

**Power precondition.** At least **84 monthly observations**. With fewer, the verdict is **UNDERPOWERED**: report descriptively and render neither PASS nor FAIL. (At ~2%/month tracking error, 36 months gives a 95% CI half-width of ~8%/yr, too wide to detect the premium being tested.)

**PASS** (all required):
1. Mean monthly net excess vs B1 > 0, with the 95% moving-block bootstrap CI (block = 3 months, 10,000 resamples, seed 20261005) lower bound **> 0**.
2. Mean excess positive in **≥ 2 of 3** equal-length sub-periods.
3. Mean excess still **> 0** (point estimate) under the −100% delisting bound.
4. The deflated Sharpe is computed with `variants_tested = 3` and **reported**. Like G3, it feeds fragility and is not a fourth blocking check.

**FAIL / REJECTED:** mean excess ≤ 0, **or** the CI upper bound ≤ 0.

**Anything else:** WATCH. Neither direction is established; no promotion and no tuning.

## Variants: three, declared now

| Arm | Definition | Role |
|---|---|---|
| **P** | momentum top-50 among the profitable | **decides** |
| S1 | momentum top-50, no quality screen | attribution only |
| S2 | equal-weight all profitable names, no momentum | attribution only |

`variants_tested = 3`. No other lookback, portfolio size, screen, holding period or universe cut may be tried under this row. A new idea is a new row.

## If it passes

G3: `generate_validation_card(..., variants_tested=3)` + `run_validation_checks`. Then G4: a quarantined paper sleeve (`signal_source="mq_core_paper"`) under `docs/paper_sleeve_acceptance_criteria.md`, outside the book until Ray decides otherwise.

## Implementation notes (not decision rules)

- Reuse the PIT vintage, `src/research/` toolkit conventions, and the trading calendar. Rebalance dates come from XNYS sessions, not calendar month-ends.
- Report a per-year table and turnover beside every CI (standing method rule).
- The report must stamp PIT vintage ID and hashes, code SHA, and `get_last_ohlcv_provenance()`.
