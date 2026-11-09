# W10H2K1 / W12H4K1 逐笔对账表（v17 strict 引擎导出）
生成时间：2026-11-09（北京时间）。仅供研究对账，不涉及实盘。引擎是冻结的 v17 strict4，运行时只读，没有改任何引擎逻辑，导出时只挂了一个只看不改的快照钩子。

## 规则（和 v17 一致）
- **信号**：每个月最后一个交易日出信号，看过去 W 个自然月的价格涨幅（不含普通股息；分拆按证据因子 F 只调整一次）。
- **成交**：信号后第一个交易日，按收盘价成交。
- **持有**：H 个月；K=1，即只买第一名。
- **入场相位**：phase m 表示 2015 年 m 月第一个交易日入场，2025 年同月第一个交易日用收盘价清仓。
- **资格**：截至信号日，已连续在 S&P 500 里满 365 天。核实过的纯更名可以延续计算。
- **费用**：每边 25bps，按实际成交金额收。
  - 首次买入：E/(1+c)。
  - 续持且份额不变：不收费。
  - 期末清仓：只对股票收费。
  - 分拆子公司股票卖出：收 25bps。
  - 发行人支付的现金对价、换股：不收费。
- **换仓**：换仓日把所有入选股票恢复等权，两次换仓之间权重自然漂移。K=1 时等权就是 100%。
- **缺数据**：缺价格或公司行动未核实时，整条路径标 UNKNOWN，不由下一名递补，也不沿用前一天价格。本批 24 条路径全部完成，没有 UNKNOWN。
- **公司行动**：现金收购按发行人公布的现金价；换股收购转成收购方股票继续持有；分拆子股按上市首个正常交易日收盘价卖出，所得现金放到下一个换仓日再用。

## 文件和字段
**summary.csv**（每个相位一行）
- `config`, `phase`, `entry`, `end`
- `status`：DATA_COMPLETE_PROVISIONAL_C 表示数据完整，但触及 C 级证据事件；UNKNOWN 表示无法计算。
- `unknown_reason` / `unknown_ticker` / `unknown_date`：本批都为空。
- `final_multiple`：期末倍数，起始净值为 1。
- `CAGR`：用 (end − entry) 天数 ÷ 365.25 年化。
- `daily_MDD`：按每日收盘净值计算的最大回撤。
- `n_rebalance_dates`
- `grade_C_events_touched`：排名或持仓触及的 C 级事件。这里的 PFE/VTRS 和 PPL/TLN 只出现在排名窗口里，两条路径都没有持有过这两只。
- `v17_frozen_*`：v17 已封存的结果。
- `v12_sealed_*`：揭盲前封存的 v1.2 结果。
- `abs_diff_vs_v17_*`：本次导出和 v17 封存值的差。

**rankings.csv**（每个实际执行的信号日一组）
- `signal_date`, `trade_date`
- `rank`：1–10，按 lookback_return 从高到低；并列时按代码字母排序。
- `ticker`：成员代码，按当时的代码，已处理更名。
- `lookback_return`：B→S 的价格涨幅，B 为 S 往前 W 个月的月末，0.5 表示 +50%。
- `eligible`：是否满足 365 天连续成员资格。
- `ranked`：是否进入排名。
- `selected`：是否被买入或续持。
- `exclusion_reason`：排除原因。
- 每个信号日都至少列出合格股票的前 10 名。
- 合格但无法排名的股票（本批 0 只）单独列出，并写明原因。
- 当时是指数成员但不满 365 天的股票：只列出涨幅足以进入前 10 名、或涨幅算不出来的那些，`eligible=False`。这部分方便 GPT 对照"按买入日判断资格"的差别。
- 由于各相位分别输出，同一个信号日会在多个相位里重复出现。

**trades.csv**
- `trade_date`
- `event`：rebalance / terminal_liquidation / cash_deal / spin / stock_deal。
- `signal_date`, `ticker`
- `action`：
  - buy、sell、hold（续持，份额不变，不收费）
  - rebalance_buy / rebalance_sell（K=1 时不会出现）
  - corporate_action_cash_out：现金收购，不收费
  - spin_child_sold：分拆子股按首日收盘价卖出，收 25bps
  - corporate_action_exchange_out / _in：换股或更名
- `weight_after`：交易后这只股票占净值的比例。
- `price`：成交收盘价，和引擎使用的价格序列同一单位（已按拆股调整）。
- `shares_traded`：起始净值 1 对应的股数，正数为买入。
- `shares_after`
- `trade_value`：shares_traded × price。
- `fee`：0.0025 × |trade_value|。
- `equity_before` / `equity_after`：交易前后的净值。
- `ca_id`：对应 corporate_actions_v11.csv 里的事件编号。
- `note`：包括证据链接。
- 同一换仓日所有行的 fee 加起来，和引擎台账的费用一致（误差在 1e-9 以内，已断言检查）。

**daily_equity.csv**
- `date`：美国交易日。
- `equity`：起始为 1。
- `cash`
- `holdings`：格式为 `代码:股数@收盘价`。期末当天写 LIQUIDATED，equity 是扣费后的清仓净值。

**pick_diff_v12_vs_v17.csv**：每个换仓日 v1.2 和 v17 的选股对照。

**medians_v12_vs_v17.csv**：两个版本 12 个相位的中位数等汇总。

## 和此前数字的关系（必读）
- 本次导出和 v17 封存结果（`full_coverage/paths_status.csv`）在 24 条路径上逐条一致，CAGR 和 MDD 的差都是 0。
- 此前报过的 W10H2 中位 41.96%（MDD −51.1%）和 W12H4 中位 37.43%，是揭盲前封存的 **v1.2** 结果。用 `/workspace/momentum432-engine-20261108/out_strict4/paths_status.csv` 可以原样复现这两个数，但它们**不是** v17 的数字。
- v17 的中位数：
  - W10H2：CAGR 45.24%（33.85–50.29%），MDD 中位 −51.78%。
  - W12H4：CAGR 43.03%（30.74–48.77%），MDD 中位 −52.00%。
- 差异全部来自 v14 修正的一个输入错误：HPE 在 2017 年分拆 DXC 时，旧数据把比例错写成 1.0（那是 CSC 持有人的比例）。HPE 持有人实际每股只获得 0.085904 股 DXC，证据是 SEC 8-K 0001193125-17-113607。这个错误让 HPE 的回看涨幅被虚高，2017 年末的 12 个月涨幅从错误的 286.50% 改正为 5.80%。
- v1.2 和 v17 之间有 96 个换仓日的选股不同（W10H2 有 60 个，W12H4 有 36 个）。这 96 个在 v1.2 里全部是 HPE，时间都在 2017-05 到 2018-04 之间。其余换仓日的选股完全相同。
- v13 的费用会计修正对中位数的影响小于 0.02 个百分点。

## 复现
`python3 export_ledgers.py && python3 compare_v12_picks.py`
需要 v17 目录（`/workspace/second-sleeve-v17-targeted-ca-20261108`）和它依赖的本地数据，全部只读。
