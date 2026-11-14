# NDX100 post rule, 2015-01 to 2026-11-13 (Queen, 2026-11-14)

Same engine, prices and membership as `research-exchange/queen/ndx100-fill-20261114-v1.1/` (commit 909cd93).
Rule: month-end rank of NDX100 PIT members by mean of last 6 monthly returns, top K equal weight, hold one month, price only.
Costs: 0 and 25 bps per side. Last holding period ends 2026-11-13 (partial month).

- `run_2015_2026.py` — run it from inside the v1.1 folder (`python3 run_2015_2026.py`).
- `results_2015_2026.csv` — CAGR / MDD. Column `total` is the ending NAV multiple (1.0 = flat), not a percent return.
- `top3_25bp_2025_2026_months.txt` — monthly Top3 picks and returns 2025-01 to 2026-11-13, plus 2025 and 2026 YTD multiples.

Headline (Top3, 25 bps): CAGR 37.72%, daily MDD -49.7%, vs QQQ price 17.14% / -35.6%, QQQ total return 18.00%.
2025: x2.228 vs QQQ x1.202. 2026 to 11-13: x1.770 vs QQQ x1.107 (daily MDD -48.5%).
