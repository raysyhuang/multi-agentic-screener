# 432 全网格盲测输入包（2026-11-09，Asia/Shanghai）——放 shared/

## 这是什么
这个包让 GPT 不用 Queen 的引擎，就能重算 **432 个配置的全部 5,184 条路径**：W1–12 × H1–12 × Top1/2/3 × 12 个入场月。
- 引擎是 v18：冻结的 v17 严格引擎加上 KORS/CPRI 去重。
- 这里**只有输入，没有答案**；答案在 queen/ 下的 answer-key 文件夹。
- 研究用途，资金 0，不涉及实盘、Cat、Bee。之前所有文件夹都没改。

规则全文在 `SPEC.md`，中英对照。第 8 节是 Top2/Top3 的补充：
- 每次调仓重置为等权；
- 续持股票只对调整量收费；
- 两次调仓之间权重漂移；
- 持有中被现金收购或换股收购怎么处理；
- UNKNOWN 怎么传递；
- UA/UAA、GOOG/GOOGL 按两只独立股票处理。

## 怎么用
```
python3 rebuild_full.py 12 12 1 1          # 任意一条路径：W H K 入场月；只用本文件夹的 CSV，只用 Python 标准库
python3 verify_full.py <answer-key文件夹> all     # 5,184 条全部重算，并和答案逐条比较（8 进程约 15 秒）
python3 verify_full.py <answer-key文件夹> sample  # 抽样：W10H2/W12H4/W12H12 的全部 K2、K3，加随机 200 条
```

## 已做的验证（Queen 端，只用本文件夹重算）
- **全部 5,184 条：5,184 条一致**。
  - 状态完全一致：DATA_COMPLETE_PROVISIONAL_C 4,341 条，CONDITIONAL 377 条，UNKNOWN 466 条。
  - UNKNOWN 的失败原因、失败股票、失败日期也全部一致。
  - 136,198 次调仓的选股全部一致；已知路径的成交笔数全部一致。
- 最大差异：
  - 终值相对差 1.8e-15；
  - CAGR 2.2e-16；
  - MDD 3.3e-16；
  - 每日净值相对差 5.0e-12（答案文件只存 12 位有效数字）。
- 抽样模式（272 条：W10H2/W12H4/W12H12 的全部 K2、K3 共 72 条，加随机 200 条，随机种子 20261109）：272/272 一致。
- 锚点（K1 中位数）：W10H2 45.24%，W12H4 43.03%，W12H6 38.95%，W12H12 36.39%，W9H2 16.74%。年冠 60.5383%。

## 文件
| 文件 | 内容 |
|---|---|
| `SPEC.md` | 规则（中英），含第 8 节全网格补充 |
| `membership_intervals.csv`、`eligibility_by_signal.csv.gz` | 成分股区间和每个信号日是否合格（含 v18 去重），与上一版输入包相同 |
| `rank_prices_month_end.csv.gz`、`rank_price_overrides.csv`、`corporate_actions_applied.csv` | 排名价格和公司行动调整，与上一版相同 |
| `trading_sessions.csv` | 交易日历 2013-11 至 2025-12 |
| `execution_availability.csv.gz` | 19,289 次新买入时查执行序列的结果（ok / not_tradable / no_verified_series），附收盘价和开盘价 |
| `execution_daily_closes.csv.gz` | 166,144 行，所有持有窗口的每日收盘价 |
| `series_info.csv` | 每个价格序列的起止日期 |
| `holding_events.csv` | 125 行，被持有股票的公司行动和引擎处理方式 |
| `untradable_deal_dates.csv` | K=1 路径跑完后补充检查用的交易日期 |
| `rebuild_full.py`、`verify_full.py` | 参考重算和比对 |
| `build/` | 生成本包的脚本（依赖 Queen 盒子上的路径，只为留档） |

## UNKNOWN、未包含和需要知道的
1. **执行数据会透露一部分答案**：`execution_availability` 列出了引擎在每个日期实际查过哪些股票，相当于暴露了部分选股名单。这是"能复现每条路径"的必要代价。真正的盲测是 GPT 先独立排名、选股，再用这些数据算收益。
2. 开盘价：19,253 条可成交记录中有 1,565 条写 UNKNOWN（没有同一口径的开盘价）。执行查询结果：ok 19,253，not_tradable 36，no_verified_series 0。引擎按收盘价成交，开盘价不影响任何结果。
3. 466 条 UNKNOWN 路径（pick_not_tradable 417 条，corporate_action_unresolved 49 条）沿用 v17/v18 的结论，本包没有试图解决。
4. 2025-11-28 信号日因为 DD 未解决的分拆无法排名。用到这个信号日的路径都已经因为别的原因提前 UNKNOWN 了。
5. 引擎小问题：K=1 路径跑完后的补充检查对 K≥2 实际不生效（见 SPEC 8.1 第 5 条）。已核对，对现有结果影响 0 条路径。
6. 成分股生效日、公司行动证据仍是 PROVISIONAL；价格只看涨跌，不含股息。
7. 不含引擎代码、原始厂商文件和 GPT 的原始文件。
