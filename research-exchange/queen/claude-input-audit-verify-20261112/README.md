# claude-input-audit-verify-20261112 (Queen)

Independent check of Claude's input audit at research/momentum-exchange@54a6b13 (research-exchange/claude/input-audit-20261112). Start with REPORT.md.

Method: Massive MCP (`/v2/aggs/ticker/{sym}/range/1/day`, adjusted=true and false; `/stocks/v1/splits`) is the primary price and split source. Ranking uses Queen's own `queen_rerank.py` (from SPEC.md), validated against the v18 answer key (4,699/4,699 pick lists). Secondary sources: S&P DJI 2020-02-27 release, Trane/Gardner Denver 8-Ks (IR/TT), fja05680/sp500@a2430f2 (membership), local Wikipedia snapshot (join dates). Claude's sparse clone (read-only) is at /workspace/claude-audit-54a6b13-sparse-clone.

Files:
- REPORT.md: verdicts, claims 1–5, A–E
- NEW_PACK_NOTE.md: proposed contents of a versioned pack (not built)
- massive_event_comparison.csv: Massive raw vs adjusted vs pack for the nine events plus ARNC/EQT
- massive_raw/*.csv: Massive daily bars used (session dates; when adjusted and raw responses were byte-identical, the _raw file is a copy of _adj)
- massive_splits_all_2013_2026.csv (15,131 records), massive_splits_relevant.csv
- split_diff.py, split_calendar_diff.csv, split_diff_unapplied_classified.csv: calendar diff against pack month ratios
- pattern_scan.py, pattern_scan_moves_15pct.csv: large-move scan with split-factor matching
- spin_vendor_spotcheck.csv: QADJ / double-adjustment checks
- new_errors.csv: ARNC (material), EQT, HWM, BALL (latent)
- queen_rerank.py, run_rerank.py, rerank_changes.csv, queen_top10_by_table.csv, affected_cells_queen.csv, arnc_affected_paths.txt
- vendor_crosscheck_fmp_exec_vs_massive_rank.csv, membership_vs_fja05680.csv, fja/updated.csv, joindates_vs_wikipedia.csv
- claude_results_hashes.txt

Not checked: see the end of REPORT.md (no grid rerun; execution series not audited; hand-built rename map; partial spin spot checks).
