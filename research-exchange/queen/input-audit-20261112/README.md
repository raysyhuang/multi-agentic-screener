# Momentum input audit: publish packet (slice 1 + slice 2)

Prepared by Queen's executor, 2026-11-12 (Asia/Taipei). Research only.

**Constraint:** this packet belongs only on branch `research/momentum-exchange`, under `research-exchange/` (Queen will place it at `research-exchange/queen/`). No main, no PR, no changes to the frozen pack or engine.

## What was checked
- **Slice 1:** external truth audit of the frozen inputs pack (`momentum432-inputs-pack-full-20261109`). It covered:
  - Top1 membership against S&P DJI announcements and Wikipedia change logs.
  - Lookback and holding prices against FMP and Massive.
  - Tight-margin ranks.
  - The UNKNOWN-path events.
  - The annual-champion path (W12H12K1P1, 60.5383% CAGR).
  - Verdict: no contradiction changes any pick or the headline. See `slice1/executive.txt`.
- **Slice 2:** independent rebuild of S&P 500 membership at each annual signal date (2014-12-31 to 2024-12-31, plus 2025 as a supplement).
  - Source: iShares Core S&P 500 ETF (IVV) SEC holdings filings (N-Q for 2013–2018, NPORT-P for 2019–2025), diffed against the pack's member_at_signal and eligible_final.
  - Verdict: no missing or false member changes any annual Top1 or the headline.
  - One real gap: acquirers in ticker-retaining mergers are omitted. These are Tyco 2014-12..2016-08, ACE 2014-12..2015-12 and Actavis 2014-12..2015-02. None comes within 17pp of Top1 or enters Top3 at any W1–W12 monthly signal.
  - CMG over BLL at 2019-07 W12 flips to BLL on a total-return basis only. The spec is price-return.
  - See `slice2/slice2_executive.txt`.

## Question for GPT to re-verify independently
Using IVV (or SPY/VOO) SEC holdings filings for each Dec-31 from 2013 to 2023, confirm two things:
1. **Membership:** S&P 500 membership at each annual signal date differs from the pack only by the names listed in `slice2/slice2_missing_extra_names.csv`.
2. **Rank effect:** none of those names (TYC, ACE, ACT in particular) has a trailing 1–12-month price return above the pack Top1 at any month-end signal in its gap window.

Also check the BLL total-return figure: 2018-07-31 to 2019-07-31, dividends $0.45, +84.58% vs CMG +83.45%.

## Excluded on purpose
- Scripts and scratch files.
- Raw SEC filing downloads (about 200 MB).
- S&P PDFs and Wikipedia raw HTML (slice-1 `sources/`).
- Per-ticker FMP price dumps.
- No credentials or keys are included.

Every source is cited by URL or accession inside the CSVs and reports, so all of it can be re-fetched.

## Files and sha256
- `slice1/anchor_gradeC_events_check.csv` — c4cbb8d8bc2ac9c67f3c9819c78b6818858620a9a1b03764aae45da569c8f42e
- `slice1/annual_champion_check.csv` — 7075e10eb9d109f0d874c7213e2a0feb9d5cab6327381754d547c733a9cfbf9e
- `slice1/executive.txt` — 2fcaba3bbf21ec944585071ea9c8df31d83a8df22ae7f36cf3b1ca9b691a1a94
- `slice1/massive_spot_closes.csv` — 057413c1cf44f6669548f051752b88ce4bdb3f3b553231f0587fb8d3bbd37905
- `slice1/member_counts_by_signal.csv` — 9c7251b4b160fe7bd8619f2ebdeaf8a7ac67342e997e8594b422fcbdedd40938
- `slice1/membership_inventory.txt` — a843ce078e99cacf77f27c49d1ac495c5fd1de67f071606ceecbd0d63c56c266
- `slice1/membership_top1_external_check.csv` — d66ea20495665214720dd3f466987c1a2d1773bae1f728528636fb490558d643
- `slice1/price_check_holding_k1.csv` — 8634beba329c1d64ee23bb335645096a94aafe76e53623783b2c0892d06a92c0
- `slice1/price_check_lookback_top1.csv` — 8e771fbf9999114f69ba054a953f6f19a5ea436e4d8d87c7af9cc2d21787416a
- `slice1/report.md` — 6dfd61f00bfd1c0a0947acff8d344467e49d115e3d7da2c1f6988163c04d030c
- `slice1/tight_margin_rank_check.csv` — 63358434587d7674097839f51d83866f2c430584c8a090081d923aed990621d3
- `slice1/top1_membership_pack.csv` — 4fed6f81fecfabb2a2a04536115dbd07244a9bd16c67d5766b72beff50f5ffde
- `slice1/unknown_unique_events.csv` — b28de6fd9aa09c2818da9c9955632876c1abf827b00c601c2577050c70c6847d
- `slice1/wikipedia_changes_parsed.csv` — c6b8d11f72df75ae1854463b7fbeeb128befd4f11e8efdb9d1c714fbd695cf39
- `slice2/massive_missing_member_daily.csv` — 8ae09baba18073d9797b5b5f22d77d9ac12ab234576c61069a1a68e7bf757e74
- `slice2/missing_member_rank_threats_monthly.csv` — b5bb7a36b09e4fcef1a0b4694d9843e3981c874e3c1a1cc60eb458166e73e00e
- `slice2/pack_eligible_not_in_prior_fund.csv` — 7fac81038b69037f4647e81c26daaffa08801d1b11f0d897c988e1a3e94d6375
- `slice2/pack_ineligible_vs_fund_seasoning.csv` — 781faaa5c7e9b44c845c823753533fe294ef02459f65998f06ebf1744bf1da72
- `slice2/slice2_executive.txt` — 879752cac4614e93d998cc73a674451650fff10650e373092ec24ca6499c202f
- `slice2/slice2_filings_cited.csv` — 2da19d540c3297ada31ecf8ccdb60f5cb44cc0635b4582ed83e26cb6de93a2c8
- `slice2/slice2_fund_holdings_matched_all.csv` — 1ac32a893397f7fb24f228a370cfbc1849e37ad2e3f78b63be7ce403a4b9c90c
- `slice2/slice2_fund_vs_pack_by_signal.csv` — fde937e85401fb04103a40e8952cbc228b5e074adca09385a32472027d6fc52d
- `slice2/slice2_missing_extra_names.csv` — 156c22e6ab5d5e239920e891d954ac6f43113f8c87e40d19ffc33bd8927a4d6a
- `slice2/slice2_rank_threats_annual.csv` — 49cdf9a9c7f784174ce7547c002e8fc02f3c0a041670f599516bd60790c20aff
- `slice2/slice2_report.md` — fc10efcd4066b1e115812d9ad3d3934a641f59d7b6319c1011e320972ef79120
