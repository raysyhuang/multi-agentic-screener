# 覆盖率附录 / Coverage appendix (PROVISIONAL, 2005–2015)

## 1. 成分股与资格 / Membership & eligibility
- 来源：公开重建的 S&P 500 历史成分股 `sp500_hist_upd.csv`，2004–2015 共 1,317 个快照，每个快照 494–504 只。**这是公开重建，不是 S&P 官方的时点成分股。**
- 130 个信号日（2004-12-31..2015-09-30）的合格股票数为 456–485 只，用 365 天连续成员资格筛选。全部 717 个标签都在宇宙里，包括 FRX、ROH、UST、MMI、Q、TIE、XL 等后来退市或被收购的公司。
- 没有只保留幸存者，没有限制在冻结包的名单里，也没有用 150 只的子集。
- 已知的成分股缺陷（未修复，只记录）：
  - Tyco 缺失。
  - ACE 和旧 Chubb 合并成 CB，Tyco 和 JCI 合并成 JCI。
  - AT&T Corp、Sears 等少数历史成员缺失或被并到别的代码。
  - 部分快照不足 500 只。
- 每个信号日 W10 和 W12 的不可排名数都是 0（`coverage_by_signal.csv`）。
- 不需要改名桥接：hist 文件已经用后来的代码给整条谱系打了标签。`build/swap_pairs.txt` 里逐对检查过，没有一对是同一证券改名。

## 2. 排名价格 / Rank prices
- 共 70,336 个 (代码, 月末) 价格：
  - 70,291 个来自 Massive grouped 月末收盘。逐个拆股做实证测试，判断 grouped 是否已经调整过；还原成原始价后，再乘以公司自身的拆股因子。
  - 43 个用 FMP 分段修复，2 个用 FMP 单点修复。
- 供应商数据修复见 `pack/repairs.csv`：
  - 剔除虚假拆股：A 2014-11-03、EBAY 2015-07-20、VLO 2013-05-02、ASH 2005-07-01、AIG 2011-01-20，以及 MNST/Hansen 的全部记录。
  - 手工补拆股 18 条，包括 BK、RF、PLD、DISCA、GOOGL、CINF、FTI、STJ、TPR、ANTM、CTX、SYMC、NE、WFT、CBS。
  - 错误月末价：GOOG 2008-09-30、HUM 2015-05-29。
  - 分段替换：HES ≤2006-04、TAP ≤2005-01。
  - 代码复用造成的比例错误：ET ×4、GLK ×0.1。
- UST 标签带有 2012-05-11 的 2:1 拆股，那是后来复用 UST 代码的 ProShares ETF 的拆股。它对 UST（2009 年前）的排名价格只是一个常数因子，收益率里会抵消；执行侧对 UST 用原始价格。

## 3. 公司行为 / Corporate actions
- 排名侧 84 条：spin 71、special_div 2、not_applicable 10、unresolved_BLOCKER 1（WIN，从冻结包复制）。证据等级 A 9 / B 21 / C 43。
- 精确 F 值（同日母公司与子公司的首个收盘价）：EXPE/TRIP、BTU/PCX、WEN/THI、FDC/WU、CCE、SUN/SXC、VIA/CBS、MOT/MMI、FO/FBHS、MRO/MPC、CVC/AMCX、FIS/LPS、AIG/WS（C）、SWY/BHN（C）。其余用月末估计，等级 C。
- 重要性：只有 1 个 (S, W) 的第一名会因 CA 改变，即 2012-07-31 W12：不调整时 ROST 第一，调整后 EXPE 第一（EXPE +79.13% 对 ROST +75.37%）。这里用的是精确 F 0.50198，等级 B，单位已核对：先做 1:2 并股，再按每股新 EXPE 分 1 股 TRIP。其余 12 个 CA 碰到前三名的情形，第一名都不变（`ca_materiality.csv`）。
- 持仓侧事件（`pack/holding_events.csv`，附 SEC 8-K 编号）：MMI、UST、ROH 现金收购；FRX→ACT 换股（标准/未选择对价）；Sprint/SoftBank（未选择即视同选现金：$5.647658 + 0.261744048 股新 S）；AIG 认股权证（C）。

## 4. 执行价格 / Execution prices
- 25 条路径上共有 61 个入选标签、71 个持有段。每段的数据源见 `execution_series_selection.csv`：
  - Massive 原始日线 × 公司拆股因子：AA、AAPL 2005、ACT、AIG 2011、AMZN、FDO、FRX、HAR、HSP、LSI、MMI、MON、NFLX 2013–14、ORLY、Q、ROH、S、SNDK、TIE、UST、XL。
  - 其余用 FMP 日线。
  - 每段都要求月末价与排名价之比的波动 ≤0.5%，且交易日没有缺失。
- 早期拆股调整后价格很低的 FMP 序列有 0.01 美元的舍入误差，所以 AAPL 2005、AMZN 2007–08、NFLX 2013–14、ORLY 2012 改用 Massive 原始日线。仍用 FMP 的低价序列（MNST、AVGO、NFLX 2015 段，最低约 9.5–10.9 美元）舍入误差 ≤0.05%。
- 抽查：每个持有段的引擎收益与月末排名价收益（各差 1 个交易日）做了对比（`segment_return_crosscheck.csv`）。差异最大的几段（APOL 2009-04-01 −15%、MU 2009-10-01、F 2009-11-02、CNX 2008-07/08）用 Massive 原始日线逐日核对，与 FMP 完全一致，是真实的单日大幅波动。
- 开盘价未收集。引擎按收盘价成交，与冻结包相同。

## 5. 与冻结 2015–2025 包的重叠验证 / Frozen-overlap validation
- 信号 2014-12-31..2015-09-30，W10 与 W12 共 20 个 (S, W)。冻结包的 `rank()` 以只读方式运行（python -B，不写任何文件）。
- **第一名 20/20 一致，合格数完全一致，第一名的回看收益差 ≤ 5e-16**（`frozen_overlap_validation.csv`）。
- 其他非第一名的差异是有意为之：本包补上了 IP/NE/VNO/SWY 的 spin，GOOGL/DISCA 的 C 类股息调整，以及 HUM 错误价格的修复。这些都不影响第一名。

## 6. UNKNOWN 与缺失数据清单 / UNKNOWN & missing-data list
- W12H4K1 的 P3、P7、P11 为 UNKNOWN，原因是 pick_not_tradable FRX@2014-07-01：信号日 2014-06-30 W12 第一名是 FRX，而 Actavis 对 FRX 的收购在执行日 2014-07-01 当天完成。按冻结规则这是规则性 UNKNOWN，不是缺数据，这三条路径的日程完全相同。失败前（2014-06-30）的净值仅供参考，分别为 0.6205 / 0.7603 / 0.6868。
- 没有因缺数据造成的 UNKNOWN。
- 仍然是假设或 C 级的项目：
  - AIG 认股权证的子公司首个收盘价 15.77，比例 0.533933（C）。
  - Sprint 按"未选择"对价处理（B）。
  - 43 个月末估计的 F 值（C），但第一名不受影响。
  - 成分股为公开重建。
