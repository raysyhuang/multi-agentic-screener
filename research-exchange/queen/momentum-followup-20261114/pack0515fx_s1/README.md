# PROVISIONAL 2005–2015 input pack (Queen v18 cross-period test) — built 2026-11-12 (Asia/Taipei)

**PROVISIONAL. Research only. Not certified, not verified, not for live use.**
Membership is a public reconstruction (`sp500_hist_upd.csv`, /workspace/prior-quarter-winner-check-2026-10-20).
This is a cross-period test, not a blind OOS (the 2005–2015 decade had already been cited by GPT before this run).

Same file formats and same engine as the frozen pack `/workspace/momentum432-inputs-pack-full-20261109` (read-only, untouched).
`rebuild_2005_2015.py` is a byte-for-byte copy of the frozen `rebuild_full.py`. Only three things change: the docstring, `first_td(2015→2005, ph)` and `first_td(2025→2015, ph)`.
The rules are unchanged: Queen v17 strict + v18 KORS/CPRI dedupe. That dedupe has no effect in 2005–2015.
- Month-end signal, execution at the close of the first session of the month. K=1, ties broken by ticker ascending.
- c = 0.0025 one-way on traded value. Liquidation at END with fee.
- 365-day continuous membership; price-return lookback.
- Spin / special-dividend F applied to the lookback start price when B < ex ≤ S.
- Deal untradable rule: deal in (d−60d, d]. K=1 post-check: deal in (S, d].
- UNKNOWN propagates. CAGR years = days/365.25.
- No 30% trailing stop.

Usage: `cd pack && python3 rebuild_2005_2015.py W H K phase` (stdlib only), or `python3 run_all.py` (25 paths → ../results/).

## Files (same schema as the frozen pack)
| file | content / source |
|---|---|
| trading_sessions.csv | NYSE sessions 2003-10-01..2016-01-29, taken from the FMP SPY daily calendar (3,104 sessions). The 2004-06-11, 2007-01-02 and 2012-10-29/30 closures are correctly absent. |
| membership_intervals.csv | intervals derived from the public-reconstruction snapshots (1,317 snapshots 2004–2015, 494–504 names each) |
| eligibility_by_signal.csv.gz | 130 signals 2004-12-31..2015-09-30. 456–485 eligible names per signal (365-day continuous membership). All 717 labels are kept; there is no frozen-name restriction and no 150-name subset. |
| rank_prices_month_end.csv.gz | 70,336 (ticker, month-end) prices. 70,291 come from Massive grouped closes, reconstructed as raw × company split factors; 43 are FMP segment repairs and 2 are FMP point repairs (see repairs.csv). No missing cell for any eligible name at any needed month-end. |
| rank_price_overrides.csv | AGN issuer routing as in the frozen pack: signals ≥ 2015-03-17 use Actavis plc (Massive ACT) month-ends before 2015-06-15 |
| corporate_actions_applied.csv | 84 rank-side events: 71 spins, 2 special dividends, 10 not_applicable, 1 unresolved_BLOCKER (WIN, copied from the frozen pack). Grades: A 9, B 21, C 43. |
| execution_daily_closes.csv.gz / series_info.csv | 71 held-segment series (9,770 rows) for the 61 pick labels on the 25 paths, plus 2 acquirer series (acq:ACT; acq:S_SPRINTCORP_2013 = Massive S.WD on 2013-07-11, then S). Source per segment: Massive raw daily aggs × company split factor, or FMP daily. A source is accepted only if its month-end ratio to the rank price stays within 0.5% across the whole segment and it has no missing session (coverage/execution_series_selection.csv). Nothing is interpolated or fabricated. |
| execution_availability.csv.gz | 268 (trade date, pick, window end) rows: 267 ok, 1 not_tradable (FRX@2014-07-01) |
| holding_events.csv | MMI cash $40 (2012-05-22); UST cash $69.50 (2009-01-06); ROH cash $78.97 (2009-04-01); FRX→ACT $26.04 + 0.3306 ACT (2014-07-01); S (Sprint/SoftBank, no-election default) $5.647658 + 0.261744048 new S (processed 2013-07-11); AIG warrants 0.533933 × 15.77 (2011-01-20, grade C) |
| untradable_deal_dates.csv | MMI, UST, ROH, FRX, S, HSP |
| repairs.csv / alias_rules.csv / ca_candidates.csv | vendor-data repairs, symbol aliasing (no rename bridges needed; the hist file labels lineages by their later ticker), and the CA candidate list |
