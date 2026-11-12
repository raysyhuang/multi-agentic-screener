# Command log (box, Asia/Taipei, 2026-11-12 ~15:55–16:10 UTC+8)
1. find / -name membership_intervals.csv / eligibility_by_signal.csv.gz / check_membership_intervals.py   (located packs; script not yet on box)
2. sha256sum on all pack copies (all identical)
3. git clone --single-branch --branch research/momentum-exchange https://github.com/raysyhuang/multi-agentic-screener.git repo-readonly   -> HEAD 8074d871540d00497644c6bac36b3f7a6e3444ca
4. cd repo-readonly/research-exchange/shared/inputs-pack-full-20261109 && sha256sum -c MANIFEST.sha256   -> all OK
5. cd repo-readonly && ( time python3 research-exchange/gpt/check_membership_intervals.py ) > ../script_stdout.txt 2> ../script_stderr.txt   -> exit 0
6. python3 independent_recheck.py repo-readonly/research-exchange/shared/inputs-pack-full-20261109 .   -> exit 0 (pandas 3.0.5)
7. FI rank-price lookback at 2025-11-28 computed from rank_prices_month_end.csv.gz -> fi_fisv_lookback_20251128.txt
8. git -C repo-readonly status --short -> empty (no modifications)
Python: 3.13.5 (main, Aug 10 2026, 12:06:59) [GCC 14.2.0]
