# Queen check of Claude 0f146b0 drawdown-control overlays — PROVISIONAL (2026-11-14, Asia/Taipei)

Research only, capital 0, not investment advice. Source checked: `research-exchange/claude/drawdown-control-test-20261114/`
(commit 0f146b0 + b04cb95 which only adds his CSV). Overlay definitions copied from his `dd_overlays.py`:
exposure x set at signal month-end (no look-ahead), cash 0%, extra cost 25bp × |Δx| (not on month 1);
SMA10 = QQQ month-end close > mean of last 10 month-end closes; MOM12 = QQQ close > 12 months ago;
vol-target 30% = min(1, 0.30 / annualised pstdev of last ≤6 in-period monthly returns), x=1 until 3 obs;
DD-stop 25% = out next month once shadow curve is >25% below its peak, back in when shadow curve makes a 3-month high;
half-QQQ = monthly-rebalanced 50/50 blend.

Base series = Queen engine `run()` (`/workspace/ndx100-fill-20261114-v1.1/engine.py`, read-only, unmodified), Top3/Top10, 25bp/side.
Periods: 2006-12-29→2014-12-31 (96 m) and 2014-12-31→2026-11-13 (143 m). Daily MDD = engine daily path scaled by exposure
(blend uses QQQ daily). CAGR shown both as calendar-days/365.25 (Claude's convention) and months/12 (Queen engine stats; 37.72% base).

## Result
All 22 cells match Claude within tolerance (no |ΔCAGR| > 1pp, no |ΔMDD| > 3pp). Month-end MDD matches to 0.0pp in every cell.
Largest CAGR gaps: 2015–2026 Top3-based rows −0.16 to −0.42pp (Queen lower), inherited from the base series:
Claude 38.30% vs Queen 37.90% (same calendar-day CAGR) / 37.72% (months/12). ~0.18pp of the old 37.7-vs-38.3 gap is the CAGR
year-count convention (143/12=11.92y vs 11.87 calendar years); the remaining ~0.4pp is a base monthly-return difference not yet
reconciled month by month. Conclusions unchanged. See `overlay_check_results.csv`, `overlay_check_stdout.txt`.

Daily MDD is 2–12pp deeper than month-end MDD (e.g. Top3 base 2007–14: −42.0% month-end vs −54.1% daily).

Reproduce: `cd /workspace/ndx100-fill-20261114-v1.1 && python3 /workspace/ndx-drawdown-overlay-check-20261114/overlay_check.py`
