# GPT ↔ Queen 逐笔选股核对（2026-11-12）

本报告只涉及研究分支 `research/momentum-exchange`，与 MAS `main` 无关。

## 复算输入与方法

- 使用 `shared/inputs-pack-full-20261109/` 里冻结的 `eligibility_by_signal.csv.gz`、`rank_prices_month_end.csv.gz`、`rank_price_overrides.csv`、`corporate_actions_applied.csv`、`trading_sessions.csv`。
- GPT 自行实现信号日回看与累计收益排序（见 `gpt/independent_ranker.py`），不调用 Queen 的 `rebuild_full.py` 或 `verify_full.py`。
- 排名使用信号日合格股票、W 个月前的月末价格、分拆/特别分红 F 因子以及 AGN/CB 指定覆盖价格；公司行动缺失会将信号置为 UNKNOWN。并列时股票代码升序。
- 所有结果都是**同一套冻结输入的不同代码计算**，并非第二来源独立校对历史成分股和供应商价格；不代表已验证真实可交易表现。

## GPT 独立计算

| 项目 | GPT 结果 |
|---|---:|
| 信号日 | 132 |
| W = 1..12 的排名表 | 1,584 |
| 因 DD 分拆 UNKNOWN 的信号排名表 | 12（均为 2025-11-28） |
| 完成排名的信号表 | 1,572 |
| W10/H2 K1：12相位 × 60笔 | 720 |
| W12/H4 K1：12相位 × 30笔 | 360 |
| W12/H12 K1：12相位 × 10笔 | 120 |
| K1 选股条目合计 | 1,200 |

**年冠 2015–2024**：LUV → NFLX → NVDA → NRG → AMD → AMD → NVDA → DVN → OXY → NVDA。

## 第一轮对账（GPT 结果生成并入库后，再读 Queen v18）

对照 Queen `queen/v18-dedupe-kors-cpri-20261109/rankings_W10H2_W12H4_v18.csv`：

- Queen 被选中的行 1,080 条；GPT 对应 1,080 条。
- `(config, phase, signal_date)` 逐一关联；**Top1 股票代码 1,080/1,080 一致，0 个差异、0 个缺失**。
- 对两边共同覆盖的 258 张独立 `(signal_date, W)` 信号表，**Top1/Top2/Top3 名称全部一致，0 个差异**。
- 其余不在 Queen 该份 v18 专题文件里的信号表没有标成「已核对」，不推测结论。
- W12/H12 的120条选股已独立输出，但本轮未与全量排名答案做同级别逐笔比对。

## 输出

- `gpt/independent_ranker.py`：独立排名程序（Python 3 标准库，无 API）。
- `gpt/independent_rankings_1584.csv`：每个信号 × W 的 Top3、涨幅及 UNKNOWN。
- `gpt/independent_k1_monthly_picks.csv`：三种 K1 策略所有相位的1,200条选股。
- `gpt/STATUS_20261112.md`：工作状态。

## 还没有完成的事情

1. 独立模拟所有 5,184 条路径的成交、手续费、分拆/收购事件、每日净值和最大回撤。**不能把 Queen `answer-key-full` 里已有的数字当作 GPT 的结果。**
2. 从独立市场数据源审计成分生效日、公司行动、退市和代码映射；当前输入源仍 PROVISIONAL。
3. 466 条 UNKNOWN 路径的触发条件与现实可交易性，仍需专项审查。
4. 补齐 GPT 与 Queen 的逐笔买卖数量、权重、费用及净值逐日差异报告。

结论：**选股计算已在 W10H2、W12H4 的1,080次执行信号层面独立代码对齐；全网格交易级别尚未核准。**
