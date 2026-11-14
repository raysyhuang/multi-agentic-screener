# ndx100-fill-20261114-v1.1 (Queen)

v1.1 修正版（研究用途）。相对已推送的 v1.0（提交 8dd0b85，目录 ndx100-fill-20261114，请勿改动）只改了三处：新进股票归因改为按只数平分；网格 W1–3/W4–12 中位数改按格子中位数汇总；补上 BIIB 2017-02-02 的 BIVV 分拆调整（F≈0.9216）。说明见 REPORT.md 文首。帖子规则 Top3 27.7%、网格中位数、2026 实盘跟踪这些主要数字没有变。

Read REPORT.md (Chinese). Reproduce, from this folder, with python3 and n100tickers/ on the path:
  python3 live_track.py; python3 run_post_rule.py; python3 run_claude_universe.py; python3 grid.py; python3 export_prices.py; python3 fmp_crosscheck.py; python3 audit_vs_packs.py
Prices are Massive daily aggregates (split-adjusted, no dividends) in raw/; all fixes are in lib.py.
Membership: jmccarrell/n100tickers @16eaf93 (clone in n100tickers/, not part of the deliverable).
Claude's repo (sparse clone @4547367) is in repo/ for claude_mask.py / claude_engine_dump.py, read-only; not part of the deliverable.
