# Taiwan W1–12 × H1–12 × Top3 × 12 start-month grid (2026-11-12)

Point-in-time FTSE TWSE Taiwan 50 / Taiwan 150 / Mid-Cap 100 (Wikipedia/FTSE reconstruction) with FinMind prices (single vendor, through 2026-10-21). 365-day seasoning. Buy next open, sell last close of the hold. Costs: buy 0.3%, sell 0.3% + 0.3% STT. lo/hi bounds for unknown stock/cash dividend splits (lo is primary). Official phase is December-end signals; the other 11 months are sensitivity only. Benchmark is 0050 total return on the same windows, no costs. Membership and prices are a public reconstruction plus a single vendor: keep the PROVISIONAL label.

This packet does not change the locked semi-annual Taiwan routine (L6m Top3, Jun 30 and Dec 31 close). 2026 incomplete holds are censored and are not in the headline multiples. No synthetic 1900 back-fill was used for MC100.

## Windows

- Full: TW50 / TW150 from 2004-12-31 to 2025-12-31 (~21y); MC100 from 2005-12-30 to 2025-12-31 (~20y, seasoning only from 2005-12).
- 10y: start 2015-12-31 (2016–2025).
- 5y: start 2020-12-31 (2021–2025).

## Official December-phase lo / TR / net of costs (5y / 10y / full), Queen-verified

- TW50 W12H12: 1.55x / 2.13x / 0.89x vs 0050 2.49 / 5.83 / 10.79, DD −74.7%
- TW150 W12H12: 2.82x / 4.74x / 1.60x vs same, DD −83.3%
- MC100 W12H12: 5.52x / 9.00x / 3.46x vs 2.49 / 5.83 / 9.80, DD −81.9%
- TW50 W6H6: 1.66x / 3.19x / 4.08x vs 2.48 / 5.69 / 10.40, DD −66.4% (loses all three windows)
- TW150 W6H6: 9.92x / 22.39x / 22.63x vs same, DD −85.8% (sanity exact match to the published Oct 21 figures)
- MC100 W6H6: 10.48x / 32.85x / 14.17x vs 2.48 / 5.69 / 9.41, DD −85.2%

W12H12 full: 0/12 start months beat 0050 in all three universes. W6H6 TW150 5y Dec/Jun is the best phase (9.92x) vs Feb/Aug 0.99x. Dropping the two best periods vs 0050 turns W6H6 and W12H12 excess negative. W9H4 Dec TW150 is 7.56x / 40.35x / 161.75x vs 0050 2.52 / 6.05 / 10.17, DD −69.6%; it is data-mined from 5,184 cells and is not a new freeze.

Read `SPEC.md` (frozen before the run) and `REPORT.md` first. `period_ledger_lo.csv.gz` is the compressed period ledger. `SHA256SUMS.txt` covers the original run outputs; this README was added for the GitHub handoff and is not in that manifest.
