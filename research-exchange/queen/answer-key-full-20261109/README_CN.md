# 432 全网格答案（2026-11-09，Asia/Shanghai）——放 queen/，不放 shared/

引擎：v17 严格引擎加 v18 KORS/CPRI 去重。全部 5,184 条路径：W1–12 × H1–12 × K1/2/3 × 12 相位。研究用途，资金 0。
配套的输入包在 `momentum432-inputs-pack-full-20261109`（放 shared/）。用它的 `verify_full.py` 只靠输入就能复现这里的全部数字：5,184/5,184 一致。

## 文件
| 文件 | 内容 |
|---|---|
| `paths_summary.csv` | 每条路径一行：W/H/K/相位、入场和结束日、状态（PROVISIONAL_C / CONDITIONAL / UNKNOWN）、失败原因/股票/日期、终值倍数、CAGR、日 MDD、实际调仓次数、成交笔数、碰到的 C 级事件 |
| `rebalances.csv.gz` | 136,198 次调仓：信号日、调仓日、选股、经济键、价格序列、调仓前后权益、费用、换手、目标权重（等权 1/K） |
| `trades.csv.gz` | 473,728 行交易台账：每次调仓逐只股票的买/卖/加/减/持有，含价格、调仓前后股数、成交额、费用、调仓前后权重；另有期末清仓和持有期公司行动行 |
| `cells_12phase_medians.csv` | 432 个格子：已知相位数、UNKNOWN 数、中位/最小/最大 CAGR、中位和最差 MDD、12 相位是否全部已知 |
| `nav_daily/nav_daily_W01..W12.csv.gz` | 4,718 条已知路径的**每日净值**（起点 1.0；每个 W 一个文件，一列一条路径，12 位有效数字）。UNKNOWN 路径没有净值 |
| `build/` | 生成脚本（run_part.py、build_outputs.py、build_nav.py、load_v17.py） |

## 说明
- 成交笔数（n_trades）：调仓时状态不是"持有不动"的每一腿各算 1 笔，期末清仓每只算 1 笔；公司行动不算。
- 台账的费用逐次调仓加总，与引擎记录的费用一致，最大相对差 8.8e-16。
- 被引擎以异常中止的 UNKNOWN 路径（买入时就不可交易的 pick_not_tradable）没有调仓台账，这是引擎的行为。其他 UNKNOWN 路径的台账保留到失败点。
- 状态分布：PROVISIONAL_C 4,341 条，CONDITIONAL 377 条，UNKNOWN 466 条（pick_not_tradable 417 条，corporate_action_unresolved 49 条），与 v17、v18 完全一致。
- 12 相位全部已知的格子有 286 个；146 个格子至少有 1 个 UNKNOWN 相位，中位数只算已知相位。
- 锚点：W10H2 45.24%，W12H4 43.03%，W12H6 38.95%，W12H12 36.39%，W9H2 16.74%；年冠（W12H12K1 第 1 相位）60.5383%。
- 所有数字仍是 PROVISIONAL：成分股生效日和公司行动审计都没变。
