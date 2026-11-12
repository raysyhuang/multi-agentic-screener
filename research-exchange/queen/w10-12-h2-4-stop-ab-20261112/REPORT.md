# 30% trailing stop on the US complementary sleeve (W10–12 × H2–4 × K1), variants A / B — REPORT

PROVISIONAL / research only, capital 0. 2026-11-12 19:00 Asia/Taipei. Official no-stop figures unchanged.

## 人话 headline
- 无止损基线完全复现：108 条路径加年冠 109/109 一致（W10H2 中位 45.24%，W12H4 中位 43.03%，年冠 60.5383%）。
- 加 30% 移动止损后，108 条路径**每条至少触发 1 次**（A 共 451 次，B 共 443 次），没有一次被迫转现金，也没有路径变 UNKNOWN。
- 收益：整片中位 CAGR 从 40.42% 降到 A 38.00%、B 36.99%。A 让 28/108 条变好，B 只有 20/108。
- 回撤：多数路径反而更深。整片中位日 MDD 从 −54.3% 变成 A −56.0%、B −57.1%；A 有 79 条变差、17 条改善。
  原因：触发多集中在 2018-10、2020-03、2022-06、2025-02/04 这几次急跌，卖在低点；换进去的名字（通常是同期强势股）也一起跌，之后又错过反弹。
- 两个一月锚点：W10H2P1 45.94% → A 44.86% / B 46.79%；W12H4P1 33.48% → A=B 32.35%（W12 时 A 和 B 规则相同）。
- 年冠（侧面对比，不替代 60.5383%）：60.54% → 57.67%，MDD −36.6% → −44.5%。
- 结论只描述，不选赢家：在这段历史上，这个止损没有改善风险收益，A 比 B 少伤一些。

## 1. Baseline (gate passed)
109/109 paths reproduced exactly from my loop (see README step 1).
| anchor | official | reproduced |
|---|---|---|
| W10H2K1 12-phase median CAGR | 45.24% | 45.2403% |
| W12H4K1 12-phase median CAGR | 43.03% | 43.0262% |
| W10H2K1P1 CAGR / MDD / final | 45.9408% / −49.2852% / 43.852 | identical |
| W12H4K1P1 CAGR / MDD / final | 33.4767% / −45.8236% / 17.957 | identical |
| W12H12K1P1 (annual) | 60.5383% / −36.6137% / 113.781 | identical |

Note: 45.24% / 43.03% are 12-phase cell medians; the January-phase paths themselves are 45.94% and 33.48%.

## 2. Band results (108 paths, primary = same-close)
| | no stop | A | B | A next-session | B next-session |
|---|---|---|---|---|---|
| median CAGR | 40.42% | 38.00% | 36.99% | 38.48% | 36.43% |
| mean CAGR | 40.59% | 38.14% | 37.13% | 38.65% | 37.14% |
| median Δ CAGR per path (pp) | – | −1.11 | −3.60 | −1.49 | −2.92 |
| paths better / worse CAGR | – | 28 / 80 | 20 / 88 | 39 / 69 | 19 / 89 |
| median daily MDD | −54.26% | −55.99% | −57.06% | −56.27% | −58.08% |
| worst MDD | −69.45% | −70.81% | −72.63% | −69.10% | −71.77% |
| median Δ MDD (pp, − = deeper) | – | −4.40 | −3.15 | −2.93 | −3.00 |
| paths with ≥1 stop | – | 108/108 | 108/108 | 108/108 | 108/108 |
| stop events | – | 451 | 443 | 448 | 443 |
| cash-flat events / UNKNOWN | – | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Δ CAGR quantiles (pp): A min −12.60, p25 −4.93, median −1.11, p75 +0.08, max +1.91; B min −12.60, p25 −5.63, median −3.60, p75 −0.33, max +1.91.
Per-path A vs B: A higher on 49, B higher on 23, equal on 36 (all W12 paths, where A ≡ B).

Per cell, 12-phase median CAGR (no stop → A / B): W10H2 45.24 → 40.30 / 41.40; W10H3 39.99 → 35.98 / 36.07;
W10H4 40.69 → 37.22 / 37.38; W11H2 40.42 → 40.18 / 35.90; W11H3 39.23 → 33.16 / 31.74; W11H4 39.75 → 38.30 / 35.02;
W12H2 39.00 → 35.30; W12H3 40.43 → 40.69; W12H4 43.03 → 38.96 (A = B for W12). Median MDD improves only in W12H4
(−52.0% → −49.2%) and is roughly flat in W10H4; all other cells deepen.

Stop frequency: stops per path 2–7 (A: 2×3, 3×37, 4×27, 5×24, 6×13, 7×4). Time from path start to first stop: median
337.5 days, mean 710 days. Time in position at trigger: median 70 days (min 22, max 580).
Only 22–23 distinct trigger events (date × stopped name) drive everything: AMD (2018-10-24/25, 2020-03-12), DVN 2022-06-24,
NFLX 2016-01-27 & 2018-10-29, UAL/AXON/TSLA 2025-02/04, ENPH 2023, CPRT 2020-03-12, ALGN 2018-10, NEM 2016-11, LUV 2015-07,
FCX 2021-09, NTAP 2018-12, MU 2018-09, NVDA 2022-04 & 2025-04, AMZN 2016-02 & 2018-12.
Switched into — A: OXY 121, NVDA 85, AMZN 59, ALGN 52, AMD 41, UAL 37, HWM 23, TRGP 15; B: MRO 72, AMZN 59, AMD 41, OXY 36,
LRCX 33, NFLX 32, HWM 28, RMD 26, NVDA 26. Top2 fallback (Top1 = stopped name) used in 340/451 A events and 275/443 B events.

Largest changes (A): worst W12H4P10 48.77 → 36.17 (−12.60 pp, MDD −15.5 pp, 6 stops), W12H4P6 −12.10, W11H4P9 −7.80,
W11H4P5 −7.56, W11H3P10 −7.45; best W12H3P2 +1.91 (MDD +7.8 pp better), W10H3P10 +1.75, W10H3P7 +1.71, W10H3P4 +1.65, W12H2P9 +0.81.
Largest changes (B): worst W12H4P10 −12.60, W11H4P9 −12.43, W12H4P6 −12.10, W11H4P5 −12.09, W11H3P10 −8.88; best W12H3P2
+1.91, W10H2P11/P9/P7/P1 ≈ +0.85–0.87.

## 3. January anchors (detail)
| path | variant | CAGR | daily MDD | final × | stops | sched. rebalances |
|---|---|---|---|---|---|---|
| W10H2K1P1 | no stop | 45.94% | −49.29% | 43.85 | 0 | 60 |
| | A | 44.86% | −54.91% | 40.72 | 3 | 60 |
| | B | 46.79% | −49.65% | 46.46 | 3 | 60 |
| | A next-session | 46.58% | −52.22% | 45.80 | 3 | 60 |
| | B next-session | 44.51% | −51.30% | 39.74 | 3 | 60 |
| W12H4K1P1 | no stop | 33.48% | −45.82% | 17.96 | 0 | 30 |
| | A = B | 32.35% | −49.18% | 16.50 | 3 | 30 |
| | A = B next-session | 33.48% | −48.41% | 17.96 | 3 | 30 |

W10H2K1P1 stops (both variants trigger on the same dates; replacements differ):
1. AMD entered 2018-09-04 @28.06, peak 2018-09-14 @32.72, trigger 2018-10-24 @22.79 (−30.35%). Table 2018-09-28: A W12 Top1 AMD → Top2 **ALGN**; B W10 Top1 AMD → Top2 **NFLX**. Back to schedule 2018-11-01. Equity 4.3278 → 4.3062.
2. AMD entered 2020-01-02 @49.10, peak 2020-02-19 @58.90, trigger 2020-03-12 @39.01 (−33.77%). Table 2020-02-28: A → **NVDA** (Top2); B → **RMD** (Top2). Back 2020-05-01.
3. DVN entered 2022-03-01 @59.25, peak 2022-06-07 @78.04, trigger 2022-06-24 @53.77 (−31.10%). Table 2022-05-31: A → **OXY**; B → **MRO**. Back 2022-07-01.
A is hurt mostly by ALGN: bought 2018-10-24 @290.83, closed 232.07 the next day (−20.2%), held to 2018-11-01.

W12H4K1P1 stops (A = B): NTAP peak = entry 2018-09-04 @87.92 → 2018-12-14 @61.43 (−30.13%) → **AMD** (Top1 2018-11-30);
AMD peak 2020-02-19 @58.90 → 2020-03-12 @39.01 → **NVDA** (Top2); DVN peak 2022-06-07 @78.04 → 2022-06-24 @53.77 → **OXY** (Top2).

Spot checks (`spot_checks_raw.txt`): for W10H2P1, W12H4P1, W12H4P10 (6 stops incl. a stop of a stop-replacement, UAL
2025-04-03 → HWM) and W12H12P1, peak and first close ≤70% recomputed from the pack's `execution_daily_closes` — all match;
every Top1/Top2 used (37 distinct signal×W tables) equals the official K=2 picks for that signal; every stop's equity ratio
= (1−c)/(1+c) = 0.9950125.

## 4. Annual champion side comparison (W12H12K1P1, stop + switch to then-visible W12 Top1) — NOT a replacement of 60.5383%
no stop 60.5383% / MDD −36.61% / 113.78× → with stop 57.67% / −44.48% / 94.99× (next-session 58.66% / −44.32%).
Stops: LUV 2015-07-01 (peak 46.99 on 2015-01-29 → 32.62) → MNST; AMD 2020-03-12 → NVDA (Top2); DVN 2022-06-24 → OXY (Top2).

## 5. Optional 50/50 books (phase 1, separate sleeves, buy-and-hold, overlap allowed)
| annual sleeve | comp sleeve | CAGR | MDD |
|---|---|---|---|
| no stop | W10H2P1 no stop | 54.75% | −36.01% |
| no stop | W10H2P1 A | 54.44% | −38.10% |
| no stop | W10H2P1 B | 55.01% | −38.38% |
| with stop | W10H2P1 A | 52.45% | −40.19% |
| with stop | W10H2P1 B | 53.09% | −41.16% |
| no stop | W12H4P1 no stop | 52.00% | −34.65% |
| no stop | W12H4P1 A=B | 51.83% | −36.16% |
| with stop | W12H4P1 A=B | 49.48% | −40.19% |

## 6. Conventions / caveats
Same-close execution primary (next-session as sensitivity, footnoted, not substituted); Top2 fallback only when Top1 is the
stopped name; cash if no eligible replacement (0 cases); peak = post-entry highest close, continues across no-trade
rebalances, resets on every newly opened position; signal table = latest month-end ≤ trigger date. ARNC/EQT and Claude's
nine corrections do not affect W9–W12 tables, so v18 inputs are used as-is. Full list of what was not checked: README.md.
