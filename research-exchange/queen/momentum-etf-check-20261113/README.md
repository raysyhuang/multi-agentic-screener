# US equity momentum ETF check

Computed 2026-11-13 (Asia/Taipei). Last US session close used: **2026-11-12**. Research only; no live changes.

This is a data check of listed US equity momentum ETFs against SPY and QQQ, and against Queen's official *price-only* 2015–2025 anchors (annual champion 60.5383% CAGR / −36.61% MDD; W10H2 45.2403% and W12H4 43.0262% 12-phase median CAGRs; complementary-sleeve 108-path median daily MDD about −54%). Those Queen numbers are single-name, price-only. These ETFs typically hold ~80–200 names and are reported here as both total return (dividends reinvested) and price-only.

Not investment advice.

## Data and method

- **Total return:** FMP `historical-price-eod/dividend-adjusted` `adjClose` (split- and dividend-adjusted). Long histories were requested in chunks because this endpoint caps at 5,000 rows.
- **Price-only:** FMP `historical-price-eod/full` `close` (split-adjusted, dividends *not* reinvested).
- **Cross-check:** Massive `adjusted=true` daily closes are split-adjusted *price* returns, not total return. On official-window endpoints, FMP price-only and Massive agreed within half a cent for MTUM, SPY, SPMO, PDP, and QMOM (e.g. SPY 2015-01-02 both 205.43; 2025-01-02 both 584.64; 2026-11-12 both 757.12). Massive was *not* used as the primary return engine because several funds (SPMO, ONEO, MMTM, XMMO under the current ticker) are missing low-volume days or, for XMMO/XSMO, start only at the 2019 ticker change.
- **Inception:** FMP `ipoDate`. First close is the first FMP bar (noted when it differs, e.g. QQQA ipoDate 2021-05-18, first close 2021-05-20; SPY ipoDate 1993-01-22, first TR/price close 1993-01-29).
- **AUM:** FMP profile `marketCap` as of the 2026-11-12 quote, used as an AUM proxy. Not an official issuer AUM filing.
- **Windows:** CAGR uses actual calendar days / 365.25 between the start and end closes. Official window is **close 2015-01-02 to close 2025-01-02** (exactly 10.00 years) when both dates exist. Funds that listed later are labelled from first close to 2025-01-02 and are *not* a full official window. “2025–now” is 2025-01-02 to 2026-11-12 (1.86 years), not calendar 2026 YTD. Max drawdown is peak-to-trough on that same daily series. Calendar-year returns use the last session of the prior year to the last session of the year; 2022 for SEIM is a partial year from 2022-05-18 (marked incomplete).
- **MFMO total-return path:** FMP dividend-adjusted history came back truncated (45 of 234 days). TR path stats for MFMO are **UNKNOWN**. Price-only is complete. FMP `lastDividend` is 0, and the 45 overlapping adjCloses match price closes within $0.0023, so price-only is a close proxy, but it is still labelled separately.

## Important caveat: XMMO and XSMO

Effective the open on **2019-06-24**, these were not a continuous momentum strategy:

- PXMG (Invesco Russell Midcap Pure Growth) → XMMO (S&P MidCap 400 Momentum)
- PXSG (Invesco Russell 2000 Pure Growth) → XSMO (S&P SmallCap 600 Momentum)

Source: Invesco press release 2019-03-11 and NYSE Arca RB-19-095. Same CUSIP, new index and ticker. **2015–2025 and 2005–2019 figures mix growth and momentum.** Pure current-strategy figures (2019-06-24 to 2025-01-02, total return): XMMO 15.35% CAGR / −36.72% MDD; XSMO 12.05% CAGR / −39.37% MDD. Their 2008 calendar losses are from the old growth indexes, not from S&P momentum.

## Compact results (total return unless noted)

AUM is FMP marketCap as of 2026-11-12. “vs SPY” is SPY total-return CAGR over the *same dates*.

**Full official window 2015-01-02 to 2025-01-02**

| Ticker | Name | Inception | AUM | TR CAGR | TR MDD | vs SPY | Excess vs SPY | Price-only CAGR / MDD | 2025-01-02 to 2026-11-12 total | Full-history TR CAGR (from) | Worst complete year | 2022 TR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MTUM | iShares MSCI USA Momentum Factor ETF | 2013-04-18 | $26.71B | 13.40% | −34.07% | 12.97% | +0.42 pp | 11.83% / −34.08% | +40.09% | 15.07% (2013-04-18) | 2022 −18.29% | −18.29% |
| PDP | Invesco Dorsey Wright Momentum ETF | 2007-03-01 | $1.37B | 10.54% | −34.70% | 12.97% | −2.43 pp | 10.19% / −34.80% | +21.29% | 9.32% (2007-03-01) | 2008 −46.20% | −24.49% |
| XMMO | Invesco S&P MidCap Momentum ETF | 2005-03-03 | $6.46B | 15.37%* | −36.72%* | 12.97% | +2.40 pp* | 14.69% / −37.01% | +12.02% | 11.32%* (2005-03-03) | 2008 −44.07%* | −16.01% |
| XSMO | Invesco S&P SmallCap Momentum ETF | 2005-03-03 | $2.12B | 11.28%* | −39.37%* | 12.97% | −1.70 pp* | 10.53% / −39.78% | +22.88% | 8.58%* (2005-03-03) | 2008 −37.85%* | −15.45% |
| MMTM | SPDR S&P 1500 Momentum Tilt ETF | 2012-10-25 | $162.5M | 13.24% | −33.85% | 12.97% | +0.27 pp | 11.57% / −34.21% | +14.75% | 13.62% (2012-10-25) | 2022 −16.12% | −16.12% |
| DWAS | Invesco Dorsey Wright SmallCap Momentum ETF | 2012-07-19 | $805.0M | 9.58% | −46.17% | 12.97% | −3.39 pp | 9.07% / −46.16% | +13.97% | 10.95% (2012-07-19) | 2022 −18.51% | −18.51% |
| SPY | SPDR S&P 500 ETF (benchmark) | 1993-01-22 | $803.0B | 12.97% | −33.72% | — | — | 11.02% / −34.10% | +32.05% | 10.75% (1993-01-29) | 2008 −36.80% | −18.18% |
| QQQ | Invesco QQQ Trust (benchmark) | 1999-03-10 | $472.4B | 18.29% | −35.12% | 12.97% | +5.31 pp | 17.36% / −35.62% | +36.02% | 10.54% (1999-03-10) | 2008 −41.74% | −32.58% |

\*XMMO/XSMO official and full-history numbers mix Russell Pure Growth until 2019-06-21 with S&P momentum after that. See caveat above.

**From first close to 2025-01-02 (not a full official window)**

| Ticker | Name | Inception / first close | AUM | TR CAGR | TR MDD | SPY same dates | Excess vs SPY | Price-only CAGR / MDD | 2025-01-02 to 2026-11-12 total | Full-history TR CAGR | Worst complete year | 2022 TR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPMO | Invesco S&P 500 Momentum ETF | 2015-10-12 | $13.92B | 17.05% | −30.94% | 14.20% | +2.85 pp | 15.55% / −31.31% | +42.66% | 17.71% | 2022 −10.44% | −10.44% |
| QMOM | Alpha Architect U.S. Quantitative Momentum ETF | 2015-12-02 | $337.1M | 11.62% | −39.16% | 14.00% | −2.37 pp | 11.07% / −39.92% | +6.76% | 10.22% | 2018 −11.76% | −7.01% |
| JMOM | JPMorgan U.S. Momentum Factor ETF | 2017-11-09 | $2.70B | 13.93% | −34.29% | 13.98% | −0.05 pp | 12.67% / −34.30% | +37.91% | 14.94% | 2022 −20.81% | −20.81% |
| FDMO | Fidelity Momentum Factor ETF | 2016-09-15 | $812.1M | 14.27% | −33.92% | 14.76% | −0.49 pp | 13.08% / −34.17% | +28.93% | 14.34% | 2022 −19.33% | −19.33% |
| VFMO | Vanguard U.S. Momentum Factor ETF | 2018-02-16 | $1.88B | 12.72% | −36.77% | 13.56% | −0.83 pp | 11.62% / −36.91% | +31.58% | 13.39% | 2022 −12.83% | −12.83% |
| ONEO | SPDR Russell 1000 Momentum Focus ETF | 2015-12-03 | $114.7M | 10.27% | −40.86% | 14.18% | −3.91 pp | 7.79% / −41.23% | +33.83% | 11.38% | 2018 −12.41% | −12.01% |
| SEIM | SEI QiM U.S. Large Cap Momentum Active ETF | 2022-05-18 | $842.5M | 19.19% | −14.12% | 18.26% | +0.92 pp | 18.08% / −14.39% | +31.33% | 17.77% | 2023 +16.22% (2022 partial −2.36%) | −2.36% (partial) |
| QQQA | ProShares Nasdaq-100 Dorsey Wright Momentum ETF | 2021-05-18 / 2021-05-20 | $9.3M | 3.08% | −38.42% | 11.55% | −8.47 pp | 2.91% / −38.50% | +29.27% | 6.91% | 2022 −29.08% | −29.08% |
| JOET | Virtus Terranova U.S. Quality Momentum ETF | 2020-11-19 | $229.9M | 11.74% | −26.57% | 14.35% | −2.61 pp | 10.71% / −26.58% | +19.96% | 11.28% | 2022 −18.05% | −18.05% |

JOET is quality *plus* momentum, not pure price momentum. QQQA is a Nasdaq-100 momentum slice and is very small. SEIM’s 19.19% starts in May 2022, after a large part of the 2022 drawdown; do not read it as a 10-year official-window result.

**Launched after the official window**

- **FMTM** (MarketDesk Focused U.S. Momentum, 2025-03-20, AUM $46.3M): no official-window figure. From inception to 2026-11-12, total return +49.88% / 27.83% CAGR / short sample only.
- **MFMO** (Motley Fool Momentum Factor, 2025-12-09, AUM $14.2M): official window n/a. Price-only from inception to 2026-11-12: 3.06% CAGR. Total-return path UNKNOWN (incomplete download).

## Closed or inactive (FMP `isActivelyTrading=false`); not scored

AMOM (QRAFT AI-Enhanced U.S. Large Cap Momentum, ipoDate 2019-05-21), OMOM (Invesco Russell 1000 Momentum, 2017-11-08), DWAQ (Invesco DWA NASDAQ Momentum, 2004-01-02), PMOM (Principal Sustainable Momentum, 2017-10-18), MOM (AGFiQ U.S. Market Neutral Momentum, 2011-09-07 — also market-neutral, out of scope), BEMO (Aptus Behavioral Momentum, 2008-06-10).

Left out of the live universe even though they exist: international momentum (IMTM, IMOM, IDMO, EEMO, PIZ, etc.), sector Dorsey Wright funds (PTF, PRN, …), value-with-momentum blends (SPVM, XMVM, XSVM, ULVM, USVM), leveraged MTUL, and multi-asset/tactical products (GMOM, VAMO, EZMO, SAMM, …).

## Takeaways

1. **None of these funds looks like Queen’s 60% / 43–45% backtest.** Over the official 2015–2025 window, the large, clean US momentum ETFs earned total-return CAGRs in a roughly 10–13% band (MTUM 13.40%, PDP 10.54%, MMTM 13.24%), versus SPY 12.97% and QQQ 18.29%. Price-only is about 1–2 pp lower (MTUM 11.83%, SPY 11.02%). That is a diversified 100-name factor, not a single-name annual champion. It does not corroborate the 60.5383% price-only champion figure, and it should not be used to argue that figure was “normal momentum.”

2. **SPMO is the best-known S&P 500 momentum product, and it did beat SPY — but only from its October 2015 inception, not over a full official window.** 17.05% TR CAGR / −30.94% MDD from 2015-10-12 to 2025-01-02, versus 14.20% for SPY on the same dates (+2.85 pp). After 2025-01-02 it also led (total +42.66% vs SPY +32.05%). That is useful context, not a 10-year official-window result, and it is still far below Queen’s single-name medians.

3. **Drawdowns are material and 2022 was not uniquely fatal for every fund.** Official-window TR MDDs clustered around −31% to −46% (MTUM −34.07%, SPMO since inception −30.94%, DWAS −46.17%), similar in size to the champion’s −36.61% and milder than the 108-path median ~−54%. 2022 TR: MTUM −18.29% (almost identical to SPY −18.18%), SPMO only −10.44%, QMOM −7.01%, PDP −24.49%, QQQ −32.58%. For funds with 2008 history, 2008 was worse (PDP −46.20%, SPY −36.80%, QQQ −41.74%).

4. **XMMO/XSMO must not be read as 20-year momentum evidence**, and several late-launch CAGRs (SEIM 19.19%, FMTM 27.83%) are short-sample. The only clean read-through for Ray’s locked annual-champion sleeve is that live, diversified US momentum ETFs did *not* produce anything like 40–60% CAGR over 2015–2025, and even the best large-cap product (SPMO) was a mid-teens total-return equity factor with a ~31% drawdown.

## Files

- `out/momentum_etf_compact_display.csv` — human-readable compact table
- `out/momentum_etf_compact.csv` — machine-readable decimals
- `out/momentum_etf_summary.csv` — total-return and price-only rows, all windows
- `out/momentum_etf_windows.csv` — per-window start/end prices, years, MDD dates, SPY/QQQ same-date stats
- `out/momentum_etf_calendar_years.csv` — calendar-year returns (complete vs partial)
- `out/closed_or_inactive_tickers.csv`, `out/excluded_tickers_notes.csv`
- `raw/*_tr_daily.csv`, `raw/*_px_daily.csv` — cleaned daily series used in the math
