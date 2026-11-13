# Claude: how real momentum ETFs have done (2026-11-13, Asia/Taipei)

Research only, not investment advice. Data pulled from FMP on 2026-11-13 and saved in this folder, so the tables can be rebuilt with `python3 summarize.py` (standard library only).

## Result

Diversified momentum ETFs have performed roughly in line with the S&P 500. Over 10 years, 3 of the 9 funds with a full history beat SPY; over the last 3 months none did. Nothing investable came near the 40%+ a year of the Top1 backtest.

### Cumulative price return to the 2026-11-12 close (dividends excluded)

| ETF | 3 months | 1 year | 5 years | 10 years | 10 years, per year |
|---|---|---|---|---|---|
| SPMO | −12.0% | +14.2% | +106% | +408% | 17.6% |
| XMMO | −16.1% | +2.2% | +43% | +344% | 16.1% |
| MTUM | −12.2% | +12.9% | +47% | +274% | 14.1% |
| **SPY** | −2.7% | +12.7% | +62% | +250% | 13.3% |
| MMTM | −5.6% | +2.6% | +45% | +225% | 12.5% |
| PDP | −7.6% | +15.7% | +32% | +217% | 12.2% |
| QMOM | −11.1% | +8.7% | +24% | +185% | 11.1% |
| DWAS | −11.2% | +15.8% | +7% | +170% | 10.5% |
| IMTM | −7.4% | +3.6% | +24% | +109% | 7.7% |
| FFTY | −22.5% | −15.1% | −42% | +27% | 2.4% |
| FDMO | −8.0% | +8.3% | +65% | under 10 years | |
| JMOM | −7.1% | +17.7% | +63% | under 10 years | |
| VFMO | −9.4% | +16.2% | +53% | under 10 years | |

Funds beating SPY: 0 of 12 over 3 months, 6 of 12 over 1 year, 5 of 12 over 3 years, 3 of 12 over 5 years, 3 of 9 over 10 years.

### Same window as the backtest (2015-01-02 to 2025-01-02, dividends included)

| | Per year |
|---|---|
| XMMO | 15.4% |
| MTUM | 13.4% |
| SPY | 13.0% |
| PDP | 10.5% |
| DWAS | 9.6% |
| Top1 backtest W10H2 (median of 12 phases, for comparison) | 45.2% |

### Bad years (dividends included)

- 2008: PDP −46.2%, SPY −36.8%. PDP is the only one of these funds that existed.
- 2022: MTUM −18.3%, SPMO −10.4%, SPY −18.2%.

## Reading

- Real momentum funds land within a few points a year of the index, either side. That matches the literature (a modest premium for diversified momentum) and our own finding that returns shrink fast as the number of holdings grows.
- The gap between these funds and the Top1 backtest is concentration in one stock, which is the part that failed in 2005–2015.
- Fund construction matters more than the label: 10-year results run from +27% to +408%.
- Momentum funds fell with or harder than the market in 2008 and in the last three months.

## Limits

- The first table is price only (FMP `quote-change`), so dividends are excluded; this understates SPY slightly more than the momentum funds.
- Only funds that still exist are listed. Closed momentum ETFs are missing, which flatters the group.
- FDMO, JMOM and VFMO have under 10 years of history (launch years 2016–2018 are from memory); FMP's "10Y" for them equals their since-launch figure, so it is not shown.
- Calendar-year results only; no within-year drawdowns.
- The fund list is Claude's selection of the main US-listed momentum ETFs, not an exhaustive screen.

## Files

- `fmp_quote_change_20261113.csv` — raw FMP price-change figures for the 13 symbols.
- `fmp_adjclose_spots_20261113.csv` — dividend-adjusted closes on the dates used for the matched-window and bad-year checks.
- `summarize.py`, `summary_stdout.txt` — rebuilds and records the tables above.
