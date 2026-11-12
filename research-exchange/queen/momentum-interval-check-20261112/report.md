# Membership-interval consistency check — 2026-11-12 (research only)

Prepared by Grok Bot for Ray. Read-only check. The frozen 2026-11-09 pack was not modified. Nothing was committed, pushed, or merged.

## Inputs
- Repo `raysyhuang/multi-agentic-screener`, branch `research/momentum-exchange`, fresh clone at `repo-readonly/`, HEAD `8074d871540d00497644c6bac36b3f7a6e3444ca` (this is still the branch tip). The working tree was clean after the run.
- Script: `research-exchange/gpt/check_membership_intervals.py`, last changed in commit `012406e732ff14c5808263d04b7a4a4e93bcf9c7`. A copy is in `script_as_run/`.
- Pack: `research-exchange/shared/inputs-pack-full-20261109/`. The script hardcodes this path relative to its own location. `sha256sum -c MANIFEST.sha256` passed for all 19 files.
  - membership_intervals.csv `aa688ffbcc799065025534529423e5e45d00f0755193730ac9909a4ac1d34863`
  - eligibility_by_signal.csv.gz `d57eb4fbeb321cbb6b985330b2857345b1cc73c5a659e8c3703d41805188bb6e`
  - These are byte-identical to the copies in /workspace/momentum432-inputs-pack-20261109, /workspace/momentum432-inputs-pack-full-20261109, the non-full shared pack, and the stale /workspace/gpt-verify-20261112 checkout.

## GPT script run
`cd repo-readonly && python3 research-exchange/gpt/check_membership_intervals.py` with Python 3.13.5. Exit code 0. stderr is empty except for `time` output. It covered 132 signals and 778 interval rows, and found 9 mismatched signal days. `missing_but_eligible=[]`.

## Independent recompute (`independent_recheck.py`, pandas, the same rule start <= S < end_exclusive)
- 132 signal dates, from 2014-12-31 to 2025-11-28. 123 match and 9 mismatch: every month-end from 2025-03-31 to 2025-11-28. 2025-01-31 and 2025-02-28 match.
- Every mismatch goes one way: the name is in eligibility but missing from intervals. No name is in intervals but missing from eligibility on any date.
- The per-day counts match GPT's printout exactly:

| signal | intervals | eligibility | missing |
|---|---:|---:|---:|
| 2025-03-31 | 499 | 503 | 4 |
| 2025-04-30 | 499 | 503 | 4 |
| 2025-05-30 | 498 | 503 | 5 |
| 2025-06-30 | 498 | 503 | 5 |
| 2025-07-31 | 495 | 503 | 8 |
| 2025-08-29 | 493 | 503 | 10 |
| 2025-09-30 | 490 | 503 | 13 |
| 2025-10-31 | 489 | 503 | 14 |
| 2025-11-28 | 486 | 503 | 17 |

- The 17 names at 2025-11-28 (CONFIRMED): APP, COIN, DASH, DDOG, EME, EXE, FISV, HOOD, IBKR, PSKY, Q, SNDK, SOLS, TKO, TTD, WSM, XYZ.
- 503 vs 486 at 2025-11-28: CONFIRMED. The interval file has 778 rows: CONFIRMED.
- On every mismatched date, every mismatched name has member_at_signal=True, eligible_365d=False, and eligible_final=False. No mismatched name is eligible_final=True on any date: CONFIRMED, and this extends GPT's 2025-11-28 statement to all 9 dates.
- The latest interval start_date is 2024-12-23: CONFIRMED.
- The 2014–2024 year-end signals (11 dates) match one-for-one: CONFIRMED. All 2014-12 to 2025-02 signals also match.
- Sanity checks: the eligibility file has no member_at_signal=False rows and no duplicate (date, ticker) pairs. No ticker has overlapping intervals.

## Corrections and nuances to GPT's writeup
1. **The gap affects only additions.** The interval file still records 2025–2026 removals: 36 end_date_exclusive values fall on or after 2025-01-01, and the latest is 2026-08-18. Only post-2024-12-23 starts are missing. The file is not simply truncated at the end of 2024. That one-sided update is why the reconstructed count falls from 503 to 486.
2. **Not all 17 are new 2025 index additions.**
   - FISV is Fiserv's ticker change back from FI. The FI interval runs 2023-06-07 to 2025-11-11, and FI appears in the 15-name rename bridge for FISV→FI. At 2025-11-28 the eligibility file lists FISV with eligible_365d=False, and FI is no longer present. So Fiserv, a continuous member, is excluded from the 2025-11-28 eligible pool because the second rename (FI→FISV) has no bridge.
   - PSKY succeeds PARA after the Skydance merger. The PARA interval ends 2025-08-08.
   - SNDK has an old SanDisk interval (2006–2016) but is a different, new company.
   - The rest are genuine 2025 additions or spin-offs (Q, SOLS).
3. **Fiserv pick risk at 2025-11-28: none.** From the pack's rank prices, FI's lookback returns at 2025-11-28 are negative for every W from 1 to 12 (−7.83% to −73.92%; see fi_fisv_lookback_20251128.txt). FI/FISV has no corporate-action rows. Also, the 2025-11-28 signal is already UNKNOWN for all 12 W because of DD. The rename-bridge gap is still worth recording as a pack-definition issue.
4. **Small limitation in GPT's script.** It checks eligible_final only for names "missing from intervals", not for "extras". There were zero extras, so this has no effect here.
5. **Framing.** SPEC.md describes membership as the "last snapshot on or before S". The interval rule gives the same result wherever the file is complete.

No supplement pack was written. If one is needed, it should be a new versioned file and must not overwrite the frozen pack.
