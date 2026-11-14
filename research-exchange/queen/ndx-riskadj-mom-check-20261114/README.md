# ndx-riskadj-mom-check-20261114 (Queen)

Research only. This folder replicates the third-party post "momentum-lab.net ndx-grjmom" (risk-adjusted 12−1 momentum on NDX100) on Queen's verified NDX100 point-in-time engine. **Every figure labelled "theirs" comes from the post and is third-party and unverified.** Figures labelled "Queen" were computed here.

## Engine and data
- Imports `/workspace/ndx100-fill-20261114-v1.1/` read-only. It uses Massive split-adjusted prices with all of v1.1's fixes, and n100tickers PIT membership.
  - engine.py sha256 `7e23b73c…fd24`
  - lib.py sha256 `1a15013a…a736`
- Nothing in that folder was changed. Bytecode writing was disabled.
- Sanity check (`verify_baseline.py`): the new runner reproduces the verified post rule exactly. 2015-01 to 2026-11-13, Top3, 25 bps, month-end close: CAGR 37.72% (months/12), daily MDD −49.69%, ×45.32.

## Their rule as implemented
- Signal at each month-end close.
- Momentum is the return from ME[t−12] to ME[t−1] (12−1). Vol is the RMS of daily returns over the same window × √252. Score = ret / vol^N.
- Top3, equal weight.
- Execution: buy at the **open of the next trading day** and exit at the next rebalance open. Opens are Massive raw open/close ratios applied to v1.1's adjusted close, so all split and spin fixes carry over.
- Cost 0.15% per side on traded weight, the same convention as engine.py.
- CAGR = calendar days / 365.25. Vol = daily std × √252. Monthly Sharpe uses rf = 0. Daily MDD and month-end MDD are both reported.

## Substitutions
1. **Prices are not dividend-adjusted.** Our Massive data is price-only; they use dividend-adjusted prices. QQQ is shown both as price and as total return.
2. Month-only symbols (21 legacy or illiquid names, e.g. BEAS, LMCK, CRWV, NBIS):
   - vol uses RMS of monthly returns × √12 (135 signal evaluations);
   - when held, they trade month-end close to month-end close (93 holding instances, all runs combined).
3. 229 holding instances had no open on the exit or entry day (delisting or vendor gap). These fall back to the last close in the segment, which is the engine's cash-out rule.
4. The last period of the windows that run to 2026-11-13 exits at the 2026-11-13 close, because there is no next open.
5. Windows:
   - A = their window: signals 2008-01-31 to 2026-07-31, exit at the 2026-09-01 open.
   - B = 2015-01 to 2026-11-13.
   - C = 2008-01 to 2026-11-13.
   - Our PIT data does support 2008-01 (12−1 lookback back to 2007-01).

## Key comparison: window A, next-open, 0.15%/side
| Item | Theirs (unverified) | Queen | Verdict |
|---|---|---|---|
| N=1 CAGR / vol / MDD | 36.74% / 37.03% / −52.91% | 29.23% / 35.9% / −61.4% (month-end −59.2%) | **mismatch** CAGR −7.5pp, MDD −8.5pp |
| N=0 (12−1) | 31.63 / 40.61 / −60.85 | 28.85 / 40.1 / −61.4 | CAGR mismatch (−2.8pp); vol and MDD match |
| 6m-mean (post rule) | 35.76 / 39.27 / −50.94 | 29.12 / 38.3 / −51.3 | CAGR mismatch (−6.6pp); vol and MDD match |
| QQQ | 16.92 / 22.39 / −49.38 | TR 16.95 / 22.4 / −49.4 (price 16.01) | match (TR) |
| N=0.5 | 36.36 | 32.61 | mismatch |
| N=1.5 | 24.19 | 24.32 | match |
| N=2 | 14.42 | 14.34 | match |
| N=1 cost 0 → 0.5% per side | 38.97 → 31.67 | 31.33 → 24.45 (0.15: 29.23, 0.25: 27.85) | levels mismatch; slope similar (−7.3 vs −6.9pp) |
| Sharpe diff N1−N0, 95% CI | [−0.043, 0.442] | point +0.074; CI [−0.134, 0.306] (block 6) / [−0.111, 0.308] (block 12) | qualitative match: CI includes 0 |
| Daily corr N1 vs N0 | 0.911 | 0.900 | match |
| Average Top3 overlap N1 vs N0 | 65.6% | 65.5% | match |
| 6−1 and 9−1: N1 Sharpe < N0 | yes | 6−1: 0.818 < 0.915; 9−1: 0.695 < 0.867 | match |
| 2026-09-30 FTNT | 103.29%, 42.42%, 2.4352, rank 12→7 | 103.29%, 42.42%, 2.4352, rank 12→7 | **exact match** |
| 2026-09-30 INTC | 166.80%, 77.99%, 2.1387, rank 7→8 | 166.80%, 77.99%, 2.1387, rank 7→8 | **exact match** |
| 2025Q1 N1 return / in-quarter MDD | −3.18% / −37.02% | −3.19% / −37.02% | match |
| 2025Q1 QQQ | −8.14% / −13.19% | TR −8.14% / −13.19% | match |
| Max-DD episode | peak 2008-05, recovered 2011-07, 1,155 days | peak 2008-05-20, trough 2010-07-02, recovered 2013-07-26, 1,893 days, −61.4% | mismatch (peak date matches) |
| Longest recovery | 2021-02-04 → 2024-05-24, 1,205 days, DD 49.91% | peak 2021-02-04 (same), recovered 2025-02-04, 1,461 days, DD −53.3%; overall longest is the 2008 episode | mismatch (peak date matches) |

Likely cause of the CAGR gaps. The exact match on the signal example, on 2025Q1, on the overlap and on QQQ shows that the signal definition, execution and cost match. The gaps sit almost entirely before 2015, plus part of 2021–2024.
- Our window-A sub-period CAGRs:
  - N1: 13.8% (2008-02 to 2014) and 39.4% (2015 to 2026-08)
  - mean6: 11.4% and 41.1%
  - N0: 8.7% and 42.6%
- Our N1 path in 2008–2011 is much worse than theirs (they recover by 2011-07; we do not recover until 2013-07).
- Candidate causes:
  - different PIT membership or price sources before 2015 (theirs are unknown);
  - delisted-name handling;
  - dividends (likely under ~1pp a year for these names; not quantified).
- These cannot be resolved without their pick list.

## Window B, 2015-01 to 2026-11-13 (next-open, 0.15%), for comparison with our baseline
| Variant | CAGR | Vol | Daily MDD | Month-end MDD | Sharpe |
|---|---|---|---|---|---|
| N=0 12−1 | 37.69% | 43.1% | −58.2% | −52.4% | 0.892 |
| N=0.5 | 41.81% | 41.5% | −53.5% | −47.7% | 0.961 |
| **N=1 12−1** | **34.09%** | 38.9% | −58.6% | −48.6% | 0.898 |
| N=1.5 / N=2 | 29.72% / 17.64% | | | | |
| 6m-mean post rule | 37.56% | 42.3% | −51.3% | −44.5% | 0.979 |
| No NVDA: N0 / N1 / 6m-mean | 26.13% / 25.31% / 29.60% | | | | |
| QQQ price / TR | 17.21% / 18.08% | 22.0% | −35.6% / −35.1% | | |

- Sharpe diff N1 − 6m-mean in B: −0.082, CI [−0.459, 0.270].
- Sharpe diff N1 − N0 in B: +0.006, CI [−0.214, 0.243].

## Conclusion
- **On our data, risk-adjusted N=1 is not robustly better than the 6m-mean post rule or than N=0.**
- Window A:
  - N1 is within 0.4pp of both on CAGR (29.23% vs 29.12% vs 28.85%).
  - Its daily MDD is 10pp worse than the 6m-mean (−61.4% vs −51.3%).
  - Sharpe: N1 0.877 vs 6m-mean 0.902 vs N0 0.803.
  - Every bootstrap CI includes 0.
- Window B: N1 is the worst of the three on CAGR (34.1% vs 37.6% vs 37.7%).
- N=1 is fragile:
  - N=0.5 is best in every window.
  - In 6−1 and 9−1, N=1 is far below N=0.
  - With a 2/5/10-day execution lag, N1 drops 3–6pp.
- Without NVDA, the 6m-mean stays ahead of N1 in both windows.
- Their headline 36.74% is not reproduced (we get 29.23%). The structure they describe is reproduced: N1 has lower vol than N0, the CI includes 0, overlap and correlation match, and N1 is worse in 6−1 and 9−1.

## Files
- `riskadj.py`: library (signals, next-open runner, stats).
- `run_all.py`: everything in out/results_*.csv and out/example_*.
- `yearly.py`: calendar-year and sub-period returns.
- `episodes.py`: drawdowns deeper than 25%.
- `verify_baseline.py`: baseline sanity check.
- `out/stdout_*.txt`: console logs.
- `out/picks_next_open_015.csv`: monthly picks for all variants (window C).
- Reproduce: `python3 verify_baseline.py; python3 run_all.py > out/stdout_run_all.txt; python3 yearly.py; python3 episodes.py` (about 1 minute).
