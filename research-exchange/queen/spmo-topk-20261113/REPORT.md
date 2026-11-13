# SPMO 规则只买前 1–3 名：回测报告（PROVISIONAL）

生成：2026-11-13（台北时间，UTC+8），Queen。仅研究，未上线，没动 Cat / Bee / MAS，没联系任何 bot，没推 GitHub。
规则全文和官方出处见 `SPEC.md`。

## 结论（人话）

1. **照 SPMO 的打分法只买前 1–3 名，2015–2025 一年只赚 11%–15%**（Top1 13.4%，Top2 11.0%，Top3 15.1%，仅价格），
   跟 SPY（11.0%，含股息 13.0%）和 SPMO 本身（2015-10 起约 15.5%，含股息 17.0%）差不多。
   但最大回撤 −47% 到 −74%，比 SPMO 的约 −31% 深得多。**风险多了一倍，收益没多。**
2. **离 Queen 年冠的 60.54% 和 W10H2 / W12H4 的 45.24% / 43.03% 很远。** 原因是 SPMO 的分数要除以波动：
   涨得猛但波动大的股票（我们年冠常押的那类）会被往下压，排前面的常是"稳稳涨"的股票（KR、HRL、BLL、NEE 这类）。
3. **还是靠 NVDA。** Top1 去掉 NVDA（让下一名补上），2015–2025 年化从 13.4% 变成 −0.6%；Top3 从 15.1% 变成 7.3%。
   Top1 里 NVDA 贡献了全部对数收益的约 106%（其余持仓合起来是亏的）。
4. **2005–2015（学术主结果，成分股和数据都是临时重建）更差：** Top1 −11.0%/年（回撤 −88%），Top2 −2.7%，Top3 +3.3%，SPY 5.5%。
5. **30% 移动止损（触发后拿现金到下次换仓）：** 只对 Top1 有帮助（2015–2025 13.4%→16.3%，回撤 −74%→−55%），
   对 Top2 / Top3 反而更差。2005–2015 也几乎没用。不能说"止损能救这个策略"。
6. **前 1–3 名几乎总是"并列"的：** 官方分数在 z=+3 封顶，前三名的分数几乎都是 4.0。所以"第 1 名是谁"其实是 Queen 用原始风险调整动量（不封顶）来分先后的，官方规则本身不区分。别人复现时必须用同一个破同分规则。
7. **Top100 近似版对得上 SPMO（检查工具没问题）：** 2015-10-12 → 2025-01-02，近似指数仅价格 15.15%/年 vs SPMO 15.55%（差 −0.40 个百分点/年），
   含股息 16.80% vs 17.05%（差 −0.25 个百分点/年；SPMO 还要扣 0.13% 管理费），日收益相关 0.92，回撤 −30.5% vs −31.3%。

## 跟 Queen 现有锚点比（2015-01-02 → 2025-01-02）

| 策略 | CAGR | MDD |
|---|---:|---:|
| Queen 年冠 W12H12K1P1（冻结锚点） | 60.5383% | −36.6137% |
| W10H2 12 相位中位数 / W12H4 12 相位中位数 | 45.2403% / 43.0262% | — |
| SPMO 规则 Top1 / Top2 / Top3（仅价格，无止损） | 13.39% / 10.99% / 15.13% | −74.4% / −63.3% / −46.6% |
| SPMO（2015-10-12 起，仅价格 / 含股息） | 15.55% / 17.05% | −31.3% / −30.9% |
| SPY（仅价格 / 含股息） | 11.02% / 12.97% | −34.1% / −33.7% |

同一日期（2015-10-12 → 2025-01-02，跟 SPMO 同起点）：Top1 14.61%、Top2 13.79%、Top3 18.48%（仅价格），SPMO 15.55%，SPY 12.24%。
含股息：Top1 15.86%、Top2 15.17%、Top3 19.88%，SPMO 17.05%，SPY 14.20%（`out/same_dates_as_spmo.csv`）。

## 全部结果（English tables; CAGR uses days/365.25; MDD = max daily drawdown; 25 bps per side included）

### Main 2015-01-02 → 2025-01-02

| Variant | CAGR % | Max DD % | $1 → | NVDA share of log growth % | stops |
|---|---:|---:|---:|---:|---:|
| Top1 price, no stop | 13.3917 | -74.438 | 3.5147 | 105.9 | 0 |
| Top1 price, 30% stop→cash | 16.2804 | -55.2887 | 4.5202 | 88.2 | 3 |
| Top1 price, no NVDA | -0.6065 | -75.6499 | 0.941 | -0.0 | 0 |
| Top1 price, no NVDA + stop | 1.9255 | -57.4084 | 1.2102 | 0.0 | 3 |
| Top1 total return, no stop | 14.6114 | -73.9971 | 3.9117 | 97.9 | 0 |
| Top2 price, no stop | 10.9875 | -63.3331 | 2.8366 | 92.4 | 0 |
| Top2 price, 30% stop→cash | 8.6915 | -67.602 | 2.3015 | 115.5 | 9 |
| Top2 price, no NVDA | 4.7221 | -64.3963 | 1.5864 | 0.0 | 0 |
| Top2 price, no NVDA + stop | 2.5557 | -68.5414 | 1.2871 | 0.0 | 9 |
| Top2 total return, no stop | 12.2914 | -60.1148 | 3.1881 | 83.5 | 0 |
| Top3 price, no stop | 15.1325 | -46.6448 | 4.0932 | 65.4 | 0 |
| Top3 price, 30% stop→cash | 11.4142 | -58.8494 | 2.9476 | 85.2 | 12 |
| Top3 price, no NVDA | 7.2893 | -49.6634 | 2.0212 | 0.0 | 0 |
| Top3 price, no NVDA + stop | 3.8243 | -61.1776 | 1.4555 | 0.0 | 12 |
| Top3 total return, no stop | 16.7676 | -44.0669 | 4.7132 | 59.6 | 0 |
| SPY price only (2015-01-02→) | 11.0238 | -34.1047 | 2.8459 | | |
| SPY total return (2015-01-02→) | 12.9737 | -33.7158 | 3.3873 | | |
| SPMO price only (2015-10-12→) | 15.5457 | -31.3136 | 3.7932 | | |
| SPMO total return (2015-10-12→) | 17.0485 | -30.9365 | 4.2735 | | |

### Academic main 2005-01-03 → 2015-01-02 (provisional membership & data)

| Variant | CAGR % | Max DD % | $1 → | NVDA share of log growth % | stops |
|---|---:|---:|---:|---:|---:|
| Top1 price, no stop | -10.9531 | -87.6753 | 0.3136 | -0.0 | 0 |
| Top1 price, 30% stop→cash | -10.3504 | -86.8156 | 0.3355 | -0.0 | 6 |
| Top1 price, no NVDA | -10.9531 | -87.6753 | 0.3136 | -0.0 | 0 |
| Top1 price, no NVDA + stop | -10.3504 | -86.8156 | 0.3355 | -0.0 | 6 |
| Top2 price, no stop | -2.704 | -67.2696 | 0.7603 | -0.0 | 0 |
| Top2 price, 30% stop→cash | -3.7409 | -67.2852 | 0.6831 | -0.0 | 11 |
| Top2 price, no NVDA | -2.704 | -67.2696 | 0.7603 | -0.0 | 0 |
| Top2 price, no NVDA + stop | -3.7409 | -67.2852 | 0.6831 | -0.0 | 11 |
| Top3 price, no stop | 3.2812 | -59.6636 | 1.3809 | 0.0 | 0 |
| Top3 price, 30% stop→cash | 2.0739 | -53.9542 | 1.2277 | 0.0 | 17 |
| Top3 price, no NVDA | 3.2812 | -59.6636 | 1.3809 | 0.0 | 0 |
| Top3 price, no NVDA + stop | 2.0739 | -53.9542 | 1.2277 | 0.0 | 17 |
| SPY price only (2005-01-03→) | 5.4992 | -56.4737 | 1.7076 | | |

### Extension 2015-01-02 → 2026-11-12

| Variant | CAGR % | Max DD % | $1 → | NVDA share of log growth % | stops |
|---|---:|---:|---:|---:|---:|
| Top1 price, no stop | 12.2621 | -74.438 | 3.9426 | 97.0 | 0 |
| Top1 price, 30% stop→cash | 18.575 | -55.2887 | 7.5437 | 65.9 | 7 |
| Top1 price, no NVDA | 0.4567 | -75.6499 | 1.0555 | 0.0 | 0 |
| Top1 price, no NVDA + stop | 6.1057 | -57.4084 | 2.0196 | 0.0 | 7 |
| Top1 total return, no stop | 13.3014 | -73.9971 | 4.398 | 90.1 | 0 |
| Top2 price, no stop | 14.406 | -63.3331 | 4.9343 | 60.3 | 0 |
| Top2 price, 30% stop→cash | 12.1691 | -67.602 | 3.9041 | 70.7 | 15 |
| Top2 price, no NVDA | 8.9352 | -64.3963 | 2.7595 | 0.0 | 0 |
| Top2 price, no NVDA + stop | 6.8054 | -68.5414 | 2.1834 | 0.0 | 15 |
| Top2 total return, no stop | 15.5973 | -60.155 | 5.5794 | 56.3 | 0 |
| Top3 price, no stop | 18.9927 | -57.6793 | 7.865 | 44.7 | 0 |
| Top3 price, 30% stop→cash | 15.2163 | -58.8494 | 5.3651 | 54.8 | 21 |
| Top3 price, no NVDA | 12.1196 | -57.6793 | 3.8837 | 0.0 | 0 |
| Top3 price, no NVDA + stop | 8.5614 | -61.1776 | 2.6492 | 0.0 | 21 |
| Top3 total return, no stop | 20.5 | -57.659 | 9.1314 | 41.8 | 0 |
| SPY price only (2015-01-02→) | 11.6257 | -34.1047 | 3.6855 | | |
| SPY total return (2015-01-02→) | 13.4628 | -33.7158 | 4.4729 | | |
| SPMO price only (2015-10-12→) | 16.3068 | -31.3136 | 5.3366 | | |
| SPMO total return (2015-10-12→) | 17.712 | -30.9365 | 6.0966 | | |

### Out-of-window 2025-01-02 → 2026-11-12

| Variant | CAGR % | Max DD % | $1 → | NVDA share of log growth % | stops |
|---|---:|---:|---:|---:|---:|
| Top1 price, no stop | 6.0896 | -73.5137 | 1.1162 | 0.0 | 0 |
| Top1 price, 30% stop→cash | 31.3661 | -50.4422 | 1.6606 | 0.0 | 4 |
| Top1 price, no NVDA | 6.0896 | -73.5137 | 1.1162 | 0.0 | 0 |
| Top1 price, no NVDA + stop | 31.3661 | -50.4422 | 1.6606 | 0.0 | 4 |
| Top1 total return, no stop | 6.2204 | -73.5052 | 1.1187 | 0.0 | 0 |
| Top2 price, no stop | 34.404 | -60.1678 | 1.7327 | 0.0 | 0 |
| Top2 price, 30% stop→cash | 32.6088 | -52.2899 | 1.6899 | 0.0 | 6 |
| Top2 price, no NVDA | 34.404 | -60.1678 | 1.7327 | 0.0 | 0 |
| Top2 price, no NVDA + stop | 32.6088 | -52.2899 | 1.6899 | 0.0 | 6 |
| Top2 total return, no stop | 34.8401 | -60.155 | 1.7431 | 0.0 | 0 |
| Top3 price, no stop | 41.4392 | -57.6793 | 1.9051 | 0.0 | 0 |
| Top3 price, 30% stop→cash | 37.3807 | -51.6293 | 1.8047 | 0.0 | 9 |
| Top3 price, no NVDA | 41.4392 | -57.6793 | 1.9051 | 0.0 | 0 |
| Top3 price, no NVDA + stop | 37.3807 | -51.6293 | 1.8047 | 0.0 | 9 |
| Top3 total return, no stop | 42.0683 | -57.659 | 1.9208 | 0.0 | 0 |
| SPY price only (2025-01-02→) | 14.9201 | -18.9989 | 1.295 | | |
| SPY total return (2025-01-02→) | 16.1308 | -18.7553 | 1.3205 | | |
| SPMO price only (2025-01-02→) | 20.1588 | -24.2546 | 1.4069 | | |
| SPMO total return (2025-01-02→) | 21.0613 | -24.1326 | 1.4266 | | |

### Calendar-year returns (%), W2015_2026x (price only; first/last year partial)

| Year | K1 | K2 | K3 | K1+stop30 | K1+exNVDA | SPY PX | SPMO PX |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 0.81 | -9.34 | -9.01 | 0.81 | 0.81 | -0.76 | 2.42 |
| 2016 | 31.16 | 31.69 | 13.98 | 31.16 | -31.32 | 9.64 | 5.27 |
| 2017 | 98.62 | 50.33 | 46.74 | 98.62 | 15.65 | 19.38 | 26.69 |
| 2018 | -4.01 | 21.71 | 17.71 | -7.54 | -4.01 | -6.35 | -1.89 |
| 2019 | -17.58 | -12.38 | 2.44 | -19.08 | -17.58 | 28.79 | 24.14 |
| 2020 | 37.53 | 18.72 | 29.09 | 37.53 | 37.53 | 16.16 | 26.25 |
| 2021 | -36.79 | -16.7 | -8.53 | -24.92 | -36.79 | 27.04 | 21.94 |
| 2022 | -13.35 | -10.8 | -14.76 | -0.78 | -13.35 | -19.48 | -12.0 |
| 2023 | -5.97 | -8.48 | 3.89 | -5.97 | -5.97 | 24.29 | 15.36 |
| 2024 | 121.11 | 76.14 | 107.01 | 121.11 | 94.15 | 23.3 | 45.05 |
| 2025 | 84.16 | 32.51 | 36.98 | 83.36 | 84.16 | 16.35 | 25.59 |
| 2026 | -34.04 | 37.15 | 44.89 | -1.44 | -34.04 | 11.03 | 12.66 |

### Calendar-year returns (%), W2005_2015 (price only; first/last year partial)

| Year | K1 | K2 | K3 | K1+stop30 | SPY PX |
|---|---:|---:|---:|---:|---:|
| 2005 | 23.68 | 12.15 | 48.54 | 23.68 | 3.5 |
| 2006 | -19.31 | -21.31 | -8.79 | -19.31 | 13.74 |
| 2007 | 10.37 | -2.08 | -0.61 | 10.37 | 3.24 |
| 2008 | -54.2 | -47.98 | -43.26 | -21.97 | -38.28 |
| 2009 | 24.76 | 34.88 | 34.84 | -9.19 | 23.49 |
| 2010 | -24.61 | -15.73 | -1.84 | -40.36 | 12.84 |
| 2011 | -18.32 | -6.31 | -10.51 | 7.81 | -0.2 |
| 2012 | -23.52 | 22.21 | 20.98 | -31.89 | 13.47 |
| 2013 | -15.3 | 14.18 | 5.15 | -18.12 | 29.69 |
| 2014 | 24.45 | 13.56 | 20.27 | 19.45 | 11.29 |
| 2015 | 0.62 | 0.49 | -0.01 | 0.62 | -0.05 |


"NVDA share of log growth" = NVDA's daily-weighted contribution to ln(ending value) ÷ ln(ending value). It can exceed 100% when the other holdings lost money in total.
2005–2015: NVDA never made the top 3, so the no-NVDA rows equal the base rows. 2025-01-02 → 2026-11-12: NVDA not held.

## 每次换仓买了什么（Top 3，K=1 买第一个，K=2 买前两个）
完整表在 `out/picks_top3_by_rebalance.csv`（含是否去掉 NVDA 两版、动量值、分数、数据标记）和 `out/holdings_top10_by_rebalance.csv`（前 10 名）。
2015 年代（执行日 → 前三）：
2014-09 LUV ARNC MU · 2015-03 KR LUV KDP · 2015-09 EA KR MNST · 2016-03 HRL ATVI RAI · 2016-09 NVDA NEM AWK · 2017-03 NVDA URI FCX · 2017-09 BA NVDA PGR · 2018-03 BA ALGN PYPL · 2018-09 HFC ABMD CPRT · 2019-03 NRG LLY KEYS · 2019-09 BLL SBUX HSY · 2020-03 CPRT NEE AAPL · 2020-09 DXCM NVDA WST · 2021-03 TSLA ENPH ETSY · 2021-09 MRNA TPR BBWI · 2022-03 DVN EXR COP · 2022-09 MCK DVN OXY · 2023-03 STLD FSLR MPC · 2023-09 GE RCL NVDA · 2024-03 NVDA META AVGO · 2024-09 VST GDDY KKR · 2025-03 PLTR TRGP VST · 2025-09 PLTR GEV TPR · 2026-03 WDC MU STX · 2026-09 SNDK MU WDC
2005 年代：2004-09 ADSK MWW VLO · 2005-03 AAPL TXU ADSK · 2005-09 KBH TXU HUM · 2006-03 ESRX AMD GLW · 2006-09 Q RAI ATI · 2007-03 DTV CMCSA BXP · 2007-09 GT AMZN EMC · 2008-03 ESRX MON CNX · 2008-09 CF MON SWN · 2009-03 FDO AMGN BF.B · 2009-09 AN ORLY FDO · 2010-03 F THC XL · 2010-09 LXK CRM AKAM · 2011-03 NFLX AZO CMI · 2011-09 COG BKNG CMG · 2012-03 DLTR BIIB ROST · 2012-09 ROST SHW ALXN · 2013-03 PHM S MPC · 2013-09 GME CI NFLX · 2014-03 MU NOC RTN · 2014-09 LUV ARNC MU

## Top100 近似版 vs SPMO（`out/top100_sanity_*.csv`）
| Basis | Approx CAGR | SPMO CAGR | Tracking diff (pp/yr) | Tracking error (ann.) | Daily corr | Approx MDD | SPMO MDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Price | 15.150% | 15.546% | −0.396 | 7.83% | 0.921 | −30.54% | −31.31% |
| Total return | 16.801% | 17.048% | −0.248 | 7.83% | 0.921 | −30.43% | −30.94% |

Approximations: full market cap (FMP), not float-adjusted; no buffer; security-level 9% / 3× cap; weights set at reference-date prices and held as fixed shares from the effective close; no index fees. CAGR is close, but daily tracking error is high (≈7.8%/yr), so treat it as a sanity check of the scoring, not a replica. Biggest yearly gaps: 2022 (−10.3 pp), 2020 (+6.5 pp), 2016 (−5.1 pp).

## 注意事项 / Caveats（必须一起看）
- **破同分是 Queen 的选择。** 官方分数封顶 4.0，前三名几乎全是 4.0；Top K 实际按不封顶的 RAMV（动量 ÷ 日波动）排。换别的破同分规则（比如市值）结果会不同。
- **成分股是公开重建**（sp500_hist_upd.csv + 冻结包的 4 处修正），不是 S&P 官方时点名单。2025-11-28 之后用同一文件的快照（最后一张 2026-08-18）。
- **2005–2015 数据缺口大：** 约 23% 的"股票×换仓日"用了月度代理波动（pack 月末价算出的月波动 × 当期日/月比例中位数）；首个表（2004-09，只持有到 2005-03-18）只有 309/495 只股票有数据。47 个关键名字（48 个文件）用 Massive 日线补齐（`data/massive_daily/`），前三名里只剩 BF.B（2009-03 第 3 名）用代理波动。
- **2015 年代** 约 2.8% 的"股票×换仓日"用代理波动；前三名里没有。2014-09 首表只有 444 只（2013-07 起点价缺口）。
- **数据修补：** FMP 的少数未复权拆股（如 COG 2012-01-26）按 pack 月末价自动修正（`Universe.repairs`，2005 年代 17 处、2015 年代 5 处）；COG 2007-04 的拆股 pack 本身也没复权，未修。
- **含股息版只做 2015 年代**（FMP dividend-adjusted）；2005 年代没有可靠的股息数据。持仓用含股息价，选股仍用价格。
- **止损版是 Queen 为这次测试选的"止损后拿现金到下次换仓"**，不是还没定的 A/B 规则。
- 被收购的持仓按最后收盘价扣费转现金（只有 2005 年代 S/Sprint 2013-07 一次）。
- 3 月第三个星期五休市（2008-03-21 耶稣受难日）时，Queen 用前一个交易日执行；官方文字没写。
- 风险调整公式和 9 个月备用公式在官方 PDF 里是图片，提取不出；Queen 用的是 RAMV = MV/σ 和 P(M−2)/P(M−11)−1（见 SPEC §1）。
- 本报告没有第三方数字；所有数字都由本目录代码从列出的输入算出。

## 文件
- `SPEC.md`（规则 + 官方原文引用）、`REPORT.md`（本文件）、`source/methodology_webfetch_2026-11-13.txt`（官方方法全文抓取）
- `code/spmo_topk.py`（引擎）、`code/run_all.py`（入口）、`code/top100_sanity.py`、`code/spmo_dates.py`、`code/make_report_tables.py`、`code/diag_missing.py`、`code/ingest_massive_daily.py`
- `out/summary.csv`、`out/yearly_returns.csv`、`out/picks_top3_by_rebalance.csv`、`out/holdings_top10_by_rebalance.csv`、`out/score_tables_full.csv.gz`（每次换仓全部股票的分数）、`out/nav_daily.csv.gz`（每个变体的日净值和持仓）、`out/events.csv`（每笔换仓、止损、公司行动、退市）、`out/same_dates_as_spmo.csv`、`out/top100_sanity_*.csv`
- `data/`：FMP 拉取脚本和原始 JSON（约 1.1 GB，可重拉）、Massive 原始导出和解析结果，以及引擎实际用到的日收盘价压缩包 `data/derived_daily_closes_*.csv.gz`（便于别人直接复现）
- `SHA256SUMS.txt`
