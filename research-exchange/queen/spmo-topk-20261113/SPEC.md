# SPMO 规则 TopK 策略说明 / "SPMO-rule TopK" Strategy Spec

Status: research only, PROVISIONAL. Written by Queen 2026-11-13 (Asia/Taipei). No live trading, nothing in Cat/Bee/MAS touched.

## 一句话（人话）
照搬 SPMO 背后指数（S&P 500 Momentum Index）的**选股打分法**：看每只股票"去掉最近一个月的过去 12 个月涨幅"，再除以它这段时间每天涨跌的波动；
打分最高的不是买约 100 只，而是**只买前 1 / 2 / 3 名，等权**。每年 3 月和 9 月第三个星期五收盘换仓，单边 25bp 费用。
另外测：去掉 NVDA（让第 K+1 名补上）、加 30% 移动止损（Queen 自选的处理：触发后这一仓拿现金等到下次换仓）、含股息版本。

## 1. 官方方法来源 / Source (verified this run)
S&P Dow Jones Indices, *S&P Momentum Indices Methodology*, November 2026 edition,
https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-momentum-indices.pdf
(fetched 2026-11-13; text saved as `source/methodology_webfetch_2026-11-13.txt`). Key lines quoted verbatim:

- Objective: "Index constituents are weighted by the product of float-adjusted market capitalization (FMC) and momentum score".
- Universe: "each index universe is all constituents of the underlying index as of the rebalance reference date"; S&P 500 Momentum → underlying S&P 500, measurement period 12 months.
- Momentum value (Appendix A): "The momentum value is computed as the 12-month price change, excluding the most recent month of the security in local currency. If 12 months of price history is not available, momentum value is calculated from nine months of price history."
- Lag example (Appendix A, NOTE 1): "if the effective rebalancing date is on 03/24/2014, the reference date is 02/28/2014, and the momentum value will be calculated based on the prices from 01/31/2014 (priceM-2) and 01/31/2013 (priceM-14)."
- NOTE 2: "If there is no price available on day M-2 or day M-14, the price from the day prior will be used. If there is no price available on any of the ten days prior, the momentum value will be calculated using formula (b)…"
- NOTE 3: "it must be trading for at least ten months prior to the rebalancing reference date."
- Risk adjustment: "Risk-Adjusted Momentum Value = [formula image] where: [σ] = Standard deviation of daily price returns for the same date period used in Step 1 above."
- Z-score (Appendix B): "calculated using the mean and standard deviation of the relevant variable within each of the index universes"; "the winsorized z-score of a security is capped at ± 3."
- Score: "If Z > 0, Momentum Score = 1 + Z; If Z < 0, Momentum Score = (1 / (1 – Z)); If Z = 0, Momentum Score = 1".
- Selection: "securities are first ranked in descending order by momentum score into five quintiles. Securities with the highest scores (the first quintile) are then selected".
- Weight cap (current, changed effective 09/18/2026 per Appendix D): "Lower of 9% at the company level and three times the security's market capitalization weight in the index." Previous: "Lower of 9% and three times its market capitalization."
- Liquidity eligibility: "minimum three-month average daily value traded (ADVT) of US$ 3 million."
- Rebalancing: "The following indices rebalance semi-annually, effective after the close on the third Friday of March and September, with a reference date of the last business day of February and August, respectively: … S&P 500 Momentum …"

**Corrections to what Queen said earlier (t1648):** volatility is measured over the **same 12-month window** (M-14 → M-2), not 3 years; the lag is exactly "skip the month before the reference month" (prices at end of M-2 and M-14, M = effective month). Rebalance months March/September are confirmed.

**Not verifiable from the text (marked):**
- The risk-adjusted formula itself is an image that did not extract. The text implies RAMV = MV / σ; Queen uses that. (Ranking is unchanged by annualising σ.)
- Formula (b) (9-month fallback) is an image; Queen uses P(M-2)/P(M-11) − 1 and σ over that window.
- z-score std: population vs sample not specified; Queen uses population (pstdev). Daily-return σ uses sample stdev.
- The 20% buffer paragraph sits under the GCC/Saudi section; it is not clear it applies to S&P 500 Momentum. Not used (irrelevant for Top 1–3 anyway).
- The sentence "Weights calculated as a result of the reference date data are implemented in the indices using closing prices as of the rebalancing reference date" appears at the end of the rebalancing list; Queen uses it only in the Top-100 sanity rebuild.
- If the 3rd Friday is a market holiday (e.g. Good Friday 2008-03-21), the rule text does not say; Queen executes at the last session before it.

## 2. Precise rules (English)
For each year y and month m ∈ {3, 9}:
1. `ref` = last trading session of month m−1. `exec` = 3rd Friday of month m (or last session before it).
2. `pe` = last session of month m−2; `ps` = last session of month m−2 of year y−1 (M−14); fallback `ps9` = last session of month m−11 (M−11).
3. Universe = S&P 500 members on `ref` (point-in-time; see §4), with a price on or within 10 sessions before `pe`, and first trade ≤ last session of month (m−1)−10.
4. MV = P(pe)/P(ps) − 1 (or 9-month formula). Spin-offs/special dividends between ps and pe: start price × F (F from the Queen packs' `corporate_actions_applied.csv`), unless the vendor series is already spin-adjusted (detected by matching the pack's F-adjusted rank-price return within 2%).
5. σ = sample stdev of daily close-to-close returns between the two price dates (spin ex-day return de-biased by F when the series is not vendor-adjusted).
6. RAMV = MV/σ. z = (RAMV − mean)/pstdev over the universe; z_w = clip(z, −3, 3); score = 1+z_w (z_w>0), 1/(1−z_w) (z_w<0), 1 (z_w=0).
7. **TopK (Queen choice):** rank by score desc; ties (several names capped at score 4.0 are common) broken by unwinsorised RAMV desc, then ticker. So Top K = Top K by RAMV. Hold Top K, equal weight.
8. Trade at the close of `exec`; equal-weight targets; one-way cost c = 25 bps on traded value, fixed-point `Ep = E0 − c·Σ|T_i − h_i|` (same as Queen v18). Start of window: buy the latest selection whose exec ≤ start at the start close. End: liquidate at END close with fee.
9. Price basis: split-adjusted, price-only (Queen v18 convention). TR variant: FMP dividend-adjusted closes for holdings (selection still price-based) — 2015-era only.
10. Corporate actions while held: spin/special dividend → shares ÷ F (value kept). A pick whose daily series ends inside the hold (acquired/delisted) → cashed out at last close with fee (flagged in events.csv).
11. **No-NVDA:** NVDA removed from the universe before ranking (next name moves up; z-stats recomputed without it).
12. **30% trailing stop (Queen choice for this test):** peak = highest close since the position's entry close (continues while the name is re-selected); trigger close ≤ 70% × peak on any session that is not a rebalance day or END → sell at that close (fee); that slot stays **cash until the next scheduled rebalance**. This is not the undecided A/B rule; it is a third, simplest treatment.

## 3. Windows
- Main: 2015-01-02 → 2025-01-02 (official window; initial holding = Sep-2014 selection).
- Academic main result (provisional membership): 2005-01-03 → 2015-01-02 (initial = Sep-2004 selection).
- Extension: 2015-01-02 → 2026-11-12 and 2025-01-02 → 2026-11-12.
- Benchmarks same dates: SPY (price / TR), SPMO (from 2015-10-12 inception).

## 4. Inputs (read-only) and how to reproduce
- Membership 2015-era: `/workspace/momentum432-inputs-pack-full-20261109/membership_intervals.csv` (frozen 2026-11-09 pack; public reconstruction + 4 official patches). After 2025-11-28: snapshots of `/workspace/prior-quarter-winner-check-2026-10-20/sp500_hist_upd.csv` (same source, last snapshot 2026-08-18).
- Membership 2005-era: `/workspace/w10h2-w12h4-oos-2005-2015-20261112/pack/membership_intervals.csv` (+ its `alias_rules.csv` for symbol routing).
- Month-end rank prices + corporate actions (cross-check and F): each pack's `rank_prices_month_end.csv.gz`, `corporate_actions_applied.csv`.
- Daily prices: `data/fmp_px_2013_2026/` (FMP stable `historical-price-eod/full`, pulled 2026-11-13 by `data/pull_fmp_px.sh`), `data/fmp_divadj_2013_2026/` (dividend-adjusted), `data/fmp_mcap_2014_2026/` (market cap), 2005-era FMP daily from `/workspace/w10h2-w12h4-oos-2005-2015-20261112/raw/fmp/` plus `data/fmp_px_2003_gap/`.
- Massive daily (`/v2/aggs/ticker/…/range/1/day`, adjusted=true) for 47 names (48 era files) whose FMP series was missing or failed the pack check: `data/massive_daily/` (raw dumps + `code/ingest_massive_daily.py`).
- Run: `python3 code/run_all.py` (stdlib only; ~2 min) → `out/`; `python3 code/top100_sanity.py` → `out/top100_sanity_*`.
- Others can re-implement from §2 and compare `out/holdings_top10_by_rebalance.csv` and `out/score_tables_full.csv.gz` line by line.
