# What a new versioned pack would contain (proposal only; not built)

Suggested name: `inputs-pack-full-20261112-v19` (new directory). Leave 2026-11-09 untouched.

rank_prices_month_end changes (scale the base value for month-ends on or before the date):
| ticker | on/before | multiply by | source |
|---|---|---|---|
| GOOGL (and GOOG) | 2014-03-31 | 0.50077 (Massive 2014-04-03 closes; 0.5005 FMP ratio also acceptable) | class C dividend |
| TMK | 2014-06-30 | 2/3 | 3:2 split (Massive record under GL) |
| DISCA | 2014-07-31 | 0.5074 (Massive closes) or 0.5 (Massive record) | class C dividend |
| BF.B | 2016-07-29 | 0.5 (cumulative 0.4 with the next row) | 2:1 |
| BF.B | 2018-02-28 | 0.8 | 5:4 |
| BLL, BALL | 2017-04-28 | 0.5 | 2:1 |
| BKNG | 2018-01-31 | 1/25 | 25:1 applied to PCLN-era bars |
| ARNC, HWM | 2016-10-31 | 1/0.801 (remove Massive's spin adjustment; keep F=0.7116) — or add a QADJ row | NEW: double adjustment |
| EQT | 2018-10-31 | 1837/1000 (remove vendor spin adj; keep F=0.5262) — or add a QADJ row | NEW: latent double adjustment |

Identity and eligibility:
- TT: back-fill month-end prices on or before 2020-02-28 as old IR / 1.2878 (or keep old IR prices and add a spin row with F = 1/1.2878 at ex 2020-03-02). Mark TT eligible for signals 2020-03-31..2021-02-26.
- IR: eligible_final = False for signals 2020-03-31..2021-02-26 (new issuer joined on 2020-03-03). Split the membership interval at 2020-03-02.
- Add a corporate_actions row documenting each fix, with Massive raw/adjusted evidence.

Quality gate to add: for every Massive split record (any adjustment_type) under the engine symbol, alias or successor, assert that the pack's month ratio is not ≈ the factor. For every spin with a Massive spin-type record, assert that a QADJ row exists.

Expected effect: with Claude's nine only, no Top5 change. With ARNC added, 5 Top3 tables in W4–W7 change (2017-02-28 W4/5/6, 2017-04-28 W6/7), and up to 364 answer-key paths would need rerunning. No W9–W12 anchors are affected.
