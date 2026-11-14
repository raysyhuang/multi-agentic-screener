# ndx100-fill-20261114 (Queen)
Read REPORT.md (Chinese). Reproduce, from this folder, with python3 and n100tickers/ on the path:
  python3 live_track.py; python3 run_post_rule.py; python3 run_claude_universe.py; python3 grid.py; python3 export_prices.py; python3 fmp_crosscheck.py; python3 audit_vs_packs.py
Prices are Massive daily aggregates (split-adjusted, no dividends) in raw/; all fixes are in lib.py.
Membership: jmccarrell/n100tickers @16eaf93 (clone in n100tickers/, not part of the deliverable).
Claude's repo (sparse clone @4547367) is in repo/ for claude_mask.py / claude_engine_dump.py, read-only; not part of the deliverable.
