# NDX100 6m-mean Top3: holdings ledger + Ray's 10% stop / buy-back test (Queen, 2026-11-15, Asia/Taipei)

Research only. No trades, capital 0, not a Cat pick card. Not pushed to GitHub.

## Engine and baseline check
Imports the verified read-only engine `/workspace/ndx100-fill-20261114-v1.1/` (engine.py / lib.py, unmodified; system python3).
Rule: at each month-end close rank NDX100 point-in-time members by mean of the last 6 monthly returns, buy Top3 equal weight,
hold to next month-end close, price only (Massive split-adjusted closes), 25 bp per side. Window 2014-12-31 -> 2026-11-13 (143 months, last month partial).

Reproduction: engine `run()` CAGR 37.72%, daily MDD -49.69%, monthly MDD -45.66%, x45.32. The new simulator `sim.py` with stop disabled
reproduces the engine NAV to 6e-13 on every day (out/run_stops_stdout.txt, first row).

## Part 1 (holdings.py)
- `out/holdings_ledger_2015_2026.csv`: one row per month per holding (signal date, holding month, rank, ticker, 6m-mean score, split-adjusted entry/exit close, month return).
  Prices are split-adjusted (not the raw quoted price). Exit close is the last close inside the same data segment (engine cash-out rule for delistings).
- `out/monthly_summary_2015_2026.csv`: basket gross/net, rebalance cost, QQQ price return, NAV, month-end drawdown, names changed vs prior month, one-way turnover.
- `out/ticker_stats.csv`: months held, avg month return, hit rate, contribution. `contrib_log` = (w*r/R)*ln(1+R) per month (sums exactly to ln gross NAV);
  `contrib_arith_sum_w_r` = simple sum of 1/3*r.
- `out/yearly_returns.csv`; charts `out/charts/a_..e_*.png`.

## Part 2 (sim.py, run_stops.py)
Per holding, daily closes inside the holding month, relative to the month-start entry close:
- from-entry: stop line = entry*(1-x). trailing: stop line = post-entry peak close*(1-x) (peak kept through stop/re-entry).
- Stop: close <= line -> sell. Re-entry (Ray's literal rule, default): a later close >= the executed sell price -> buy back. Unlimited repeats.
  Sensitivity `reenter@line`: re-entry only when close >= max(sell price, stop line) (differs only when the stop close gapped through the line).
- Each stop sale and each buy-back pays 25 bp on the sleeve. Cash earns 0. No stop/re-entry action on the last day of the month
  (the normal month-end rebalance handles it; a stopped sleeve re-picked next month pays the normal buy cost). Rebalance cost uses actual end-of-month stock weights.
- Execution: `same-close` acts at the triggering close (OPTIMISTIC: you cannot both observe a close and trade at it);
  `next-open` acts at the next trading day's open. Adjusted open = adjusted close x raw open/raw close of that day from the Massive day bars
  (same files the engine uses; rename stitching mirrored). 0 missing opens; out/check_opens_stdout.txt lists 4 |open gap|>25% days (ZM 2020-09-01, NVDA 2023-05-25, APP 2025-02-13, INTC 2026-10-30).
- Metrics: CAGR (months/12, engine convention), daily and month-end MDD, multiple, worst month, stop events, re-entries,
  whipsaw = buy-back price > sell price, stops helped/hurt = month-end close below/above that sell price, stop cost drag (% of NAV per year),
  % of sleeve-days in cash, calendar-year returns, 2020/2022 year return and intra-year daily MDD.
- "best variant" in the chart = highest CAGR/|daily MDD| among literal-rule variants: in-sample selection, not a recommendation.
- Outputs: `out/stop_variants_summary.csv`, `out/stop_variants_yearly_returns.csv`, `out/stop10_from_entry_event_log.csv`, `out/charts/stop_nav_drawdown.png`.

## Result in one line
No stop variant reduced the full-period daily MDD below the baseline -49.7% (stop variants range -55.9% [from-entry 20% same-close] to -72.7% [trailing 10% next-open]),
and every variant lost 3.9-24.3 pp of CAGR. Stops do cut some single-month losses and the 2020 crash, but whipsaw (sell low, buy back higher, 25 bp twice)
compounds into deeper multi-year drawdowns (2022-2025). See out/run_stops_stdout.txt.

## Caveats
Price only (no dividends); PIT membership from the n100tickers package as used by the engine; split-adjusted Massive data with the engine's
corporate-action fixes; same-close execution is optimistic; next-open ignores slippage beyond 25 bp; no intraday stops (close-based only);
INTC Oct-2026 path (-28.6% on 2026-10-30) comes from the engine's Massive data and was not separately re-verified here.
Nov-2026 is partial to 2026-11-13. Baseline is at -46.7% from its 2026-06-22 peak as of 2026-11-13 (out/current_drawdown_stdout.txt).

Reproduce: `python3 holdings.py; python3 run_stops.py; python3 check_opens.py` (from this folder, system python3).
