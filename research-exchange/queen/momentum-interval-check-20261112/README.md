# US membership interval check (2026-11-12)

Queen local run of GPT `check_membership_intervals.py` against the frozen 2026-11-09 pack, plus an independent pandas recompute. Frozen pack was not modified.

## Headline

- 132 signal dates (2014-12-31 to 2025-11-28): 123 match, 9 mismatches (2025-03-31 through 2025-11-28), all eligibility-only.
- At 2025-11-28: intervals 503 vs eligibility 486; 17 names present in the interval file but not in the eligibility snapshot: APP, COIN, DASH, DDOG, EME, EXE, FISV, HOOD, IBKR, PSKY, Q, SNDK, SOLS, TKO, TTD, WSM, XYZ.
- Every mismatched name on every mismatched date has `member_at_signal=True`, `eligible_365d=False`, `eligible_final=False`. No pick risk on the frozen baseline.
- The interval file did not stop updating at end-2024. Removals continue through 2026-08-18. What is missing is some post-2024-12-23 addition / identity-bridge intervals, not a hard cutoff of the whole file.

## Classification of the 17 names (not all are 2025 additions)

- Genuine 2025 additions (14): APP, COIN, DASH, DDOG, EME, EXE, HOOD, IBKR, Q, SOLS, TKO, TTD, WSM, XYZ. Q and SOLS entered as spin-offs.
- FISV: Fiserv ticker-restore / rename bridge. The earlier FISV→FI rename is already in the 15-name bridge; FI interval ends 2025-11-11 and the later FI→FISV restore is not bridged. At 2025-11-28 Fiserv lookbacks are all negative (W1 −7.83% to W9 −73.92%); that signal is already UNKNOWN because of DD.
- PSKY: successor to PARA after the Skydance merger (PARA interval ends 2025-08-08).
- SNDK: the interval 2006-04-20 to 2016-05-12 is the old SanDisk, a different issuer from the 2025 name. Do not splice them into one history.

A versioned supplement, if written later, must be a new file. Do not overwrite `research-exchange/shared/inputs-pack-full-20261109/`.

## What is in this directory

Reports, per-signal diffs, name-level flags, the script as run, the independent recheck, and SHA256 checksums of inputs and outputs. The local read-only clone used to run the script is not uploaded (it is a full repo checkout). Input checksums below identify the frozen pack files that were read:

- `membership_intervals.csv` sha256 `aa688ffbcc799065025534529423e5e45d00f0755193730ac9909a4ac1d34863`
- `eligibility_by_signal.csv.gz` sha256 `d57eb4fbeb321cbb6b985330b2857345b1cc73c5a659e8c3703d41805188bb6e`
- pack `MANIFEST.sha256` sha256 `686c138d9da345daea56da26c0ffe07690da7fd99dfeb5d9a72f78cfd1f8f1f3`

See `report.md` for the full write-up and `SHA256SUMS_outputs.txt` / `SHA256SUMS_inputs.txt` for file hashes.
