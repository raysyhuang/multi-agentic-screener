# spmo-topk-20261113 — how to reproduce
Research only (Queen, 2026-11-13 Asia/Taipei). Read SPEC.md (rules + S&P source quotes) and REPORT.md (results, Chinese).
1. Inputs already on the box (read-only): frozen pack /workspace/momentum432-inputs-pack-full-20261109, provisional 2005 pack
   /workspace/w10h2-w12h4-oos-2005-2015-20261112/pack (+ raw/fmp), membership file /workspace/prior-quarter-winner-check-2026-10-20/sp500_hist_upd.csv.
2. Price data in data/ (re-pull FMP with data/pull_fmp_px.sh, pull_fmp_more.sh, *_extra.sh; needs env FMP_API_KEY).
   Massive dumps in data/massive_daily/raw_dumps -> python3 code/ingest_massive_daily.py.
3. python3 code/run_all.py && python3 code/top100_sanity.py && python3 code/spmo_dates.py && python3 code/make_report_tables.py
   (Python 3.11+, stdlib only.)
4. Compare your own implementation against out/score_tables_full.csv.gz (every name, every rebalance) and out/picks_top3_by_rebalance.csv.
5. Shareable compact inputs (what the engine actually used, after Massive overlays and split repairs):
   data/derived_daily_closes_2015_px.csv.gz, data/derived_daily_closes_2015_tr.csv.gz, data/derived_daily_closes_2005_px.csv.gz.
   For GitHub, push everything except the raw FMP JSON folders data/fmp_px_2013_2026, data/fmp_divadj_2013_2026,
   data/fmp_mcap_2014_2026, data/fmp_px_2003_gap (~1.1 GB; re-pullable with the scripts). Market caps for the Top-100 check
   are only in data/fmp_mcap_2014_2026 (re-pull with data/pull_fmp_more.sh).
