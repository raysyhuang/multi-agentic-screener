# Momentum-432 Top-K 选股与执行规则 / Ranking & Execution Spec (Queen v17 strict engine + v18 dedupe)

Status: PROVISIONAL research spec, 2026-11-09 (Asia/Shanghai). Research only, capital 0.
The pack's own inputs reproduce every rule below exactly (see README_CN.md and the reproduction results).

## 0. 符号 / Notation
- `ME`：每个月最后一个交易日（来自 trading_sessions.csv）。/ last trading session of each month.
- `W`：回看月数，1–12。/ lookback in months.
- `H`：持有月数，即每 H 个月调仓一次，1–12。/ holding months; rebalance every H months.
- `K`：持股数，1、2、3。/ number of names held.
- `phase` ph：入场月份，1=一月 … 12=十二月。/ entry month.
- `c = 0.0025`：单边费用（25bp），按成交额收取。/ one-way fee on traded value.

## 1. 调仓日和信号日 / Rebalance and signal dates
- 调仓日：从 2015 年第 ph 月起，每 H 个月的第一个交易日：`d_j = first session of month (ph-1)+j*H after 2015-01`。
  Rebalance dates are the first session of every H-th month starting at month ph of 2015.
- 只要 `d_j < END` 就调仓。`END = 2025 年第 ph 月的第一个交易日`，大约 10 年。
  Rebalancing continues while d_j < END, where END is the first session of month ph in 2025.
- 信号日 `S = d_j 的前一个交易日`，也就是上个月的最后一个交易日（总是一个 ME）。
  The signal date S is the session before d_j, i.e. the previous month-end.
- 年冠就是 W12 H12 K1 phase 1：2014-12-31 出信号，2015-01-02 收盘买入，持有一年。
  The annual champion is W12H12K1 phase 1.

## 2. 合格股票池 / Eligible universe (`eligibility_by_signal.csv.gz`, column `eligible_final`)
- 成分股来源：`membership_intervals.csv`，即公开重建名单 sp500_hist_upd.csv 加上 4 处官方生效日修正（v14）。在 S 当天或之前的最后一个名单快照里出现，就算 S 日成员。
  Source: the public reconstruction plus 4 official effective-date patches. A name is a member at S if it is in the last snapshot dated on or before S.
- 365 天连续条件：从 `S-365 天`（含这天之前的最后一个快照）到 S 的每一个快照里都在。
  365-day rule: the name must be present in every snapshot from S−365 calendar days (including the last snapshot on or before that day) through S.
- 改名桥接：15 个已验证的纯改名（同一 CIK），新旧代码合起来算连续，只给母公司，不给分拆子公司（`eligible_via_rename_bridge`）。
  Rename bridge: for 15 verified pure renames (same CIK), the old and new tickers count as one continuous membership. Parents only, never spin children.
- v18 去重：如果 KORS 和 CPRI 同时合格（2017-01-31 至 2018-08-31），删掉 KORS，这家公司以引擎代码 CPRI 排一次，当时的真实代码是 KORS（`ticker_at_signal`）。
  v18 dedupe: when KORS and CPRI are both eligible, KORS is removed. The company ranks once as CPRI; its ticker at the time was KORS.
- 不补位、不事后加入。/ No substitution, no look-ahead additions.

## 3. 回看涨幅 / Lookback return
- `B = ME[index(S) - W]`，即 S 往前数第 W 个月末。/ the W-th month-end before S.
- 价格 `P(t, d, S)`：如果 `rank_price_overrides.csv` 里有 `(t, d)` 而且 `signal_from <= S <= signal_to`，用覆盖值，否则用 `rank_prices_month_end.csv.gz` 的值。覆盖只有 17 行：AGN 发行人切换、CB 代码复用。
  Price lookup: use the override when (t,d) is listed and S is inside its signal range; otherwise use the base table. There are only 17 override rows (AGN issuer routing, CB ticker reuse).
- 分拆和特别分红调整：对 `corporate_actions_applied.csv` 里 kind 为 `spin` 或 `special_div`、并且 `B < ex_date <= S` 的每个事件，把起点价乘以 `factor_F`。
  Spin and special-dividend adjustment: for every such event with B < ex_date ≤ S, multiply the start price by factor_F.
  - `F = P_ex / (P_ex + ratio × child_first_close + cash_per_share)`。
  - 例：HPE→DXC：ratio 0.085904（SEC 8-K 0001193125-17-113607），F=0.750654。
- 涨幅 `ret = P(t,S,S) / (P(t,B,S) × ΠF) - 1`。只看价格涨跌，不含股息。/ Price return only, no dividends.
- 价格口径：Massive 分组日收盘价（厂商做了拆股调整，不做股息调整）。Massive 自己对部分分拆做的调整已经按 q 因子去掉（`vendor_spin_deadjust_q_ALREADY_IN_PRICES` 行，已含在价格里，不要再乘），所以分拆只调整一次，就是上面的 F。
  Price basis: Massive grouped closes, split-adjusted, not dividend-adjusted. Massive's own spin adjustments were removed with q (already in the prices; do not apply again), so each spin is adjusted exactly once, through F.
- 已退市的标的在交易完成后，排名价是文件写明的交易对价（现金，或换股比例 × 收购方收盘价），已经含在价格里。
  After a deal closes, the rank price of the delisted target is the documented deal value; this is already in the prices.

## 4. 缺数据 / UNKNOWN
- 合格股票出现下面任何一种情况，就是"无法排名"：
  - 起点或终点没有价格；
  - 有 F 但 F 为空；
  - `B < ex_date <= S` 内有 `unresolved_BLOCKER`；
  - S 落在 `legacy_data_error_BLOCKER` 区间里。
  An eligible name is unrankable if any of these holds: no start or end price, a blank F, an unresolved blocker with B < ex_date ≤ S, or S inside a legacy data-error interval.
- 只要有一只无法排名，**整个信号日就是 UNKNOWN**，用到它的路径整条变 UNKNOWN。不跳过、不递补、不取上下界。
  If any eligible name is unrankable, the whole signal is UNKNOWN and every path that uses it is UNKNOWN. No skipping, substitution or bounds.
- 在本包范围内（2014-12 至 2025-11），只有 2025-11-28 一个信号日（12 个 W）因 DD 未解决的分拆变成 UNKNOWN。
  Within the pack, only the 2025-11-28 signal (12 W) is UNKNOWN, because of DD's unresolved spin.
- 执行侧也会让路径变 UNKNOWN（本包只附了 36 条 K1 路径的执行数据）：
  Execution-side causes of UNKNOWN (execution data in this pack covers only the 36 K1 paths):
  - 选中的股票没有经过核验的执行价格序列；
    the pick has no verified execution series;
  - 选中的股票在执行日前 60 天到执行日之间（不含 60 天前那天，含执行日）有现金或换股交易生效，买不到；
    the pick's cash or stock deal took effect in (execution − 60 days, execution], so it cannot be bought;
  - 持有期间缺日线，或者遇到没有证据的公司行动。
    a daily close is missing during the holding, or a corporate action without evidence is hit.

## 5. 排序和并列 / Ranking and ties
- 按 ret 从高到低排；ret 相同时按引擎代码字母顺序。取前 K 只。
  Sort by return descending, then engine ticker ascending. Take the first K.

## 6. 执行和费用 / Execution and fees
- 在调仓日 d 的收盘价成交（`execution_prices_picks.csv`）。开盘价只作参考，引擎不用。
  Trades execute at the close of d. Opens are informational only.
- 调仓前的权益 `E0 = 现金 + Σ 股数×当日收盘`。目标每只 `T = Ep/K`。费用用不动点解：`Ep = E0 - c × Σ|T_i - h_i|`，其中 h 是调仓前各持仓市值。
  Pre-trade equity is cash plus marked holdings. Each target is Ep/K, solved by the fixed point Ep = E0 − c·Σ|T_i − h_i|.
- K=1 的写法：
  K=1 forms:
  - 换股：`Ep = (E0 - c·h)/(1+c)`。/ switch.
  - 首次买入：`Ep = E0/(1+c)`。/ first buy.
  - **续持 Top1**（同一只，而且没有现金）：Ep=E0，**不收费**。/ retained Top1 with no cash: no fee.
  - 续持但有分拆现金：`Ep = (E0 + c·h)/(1+c)`，只对补仓部分收费。/ retained with spin cash: only the top-up pays a fee.
- K=2 或 3：每次调仓都重新等权；续持的股票只对调整的部分收费。两次调仓之间权重自然漂移，不做中途再平衡。
  K=2 or 3: reset to equal weight at each rebalance; retained names pay fees only on the adjustment; weights drift between rebalances.
- END 当天收盘全部卖出，收卖出费：`E_end = 现金 + 市值 × (1 - c)`。/ terminal liquidation with fee.
- 每个交易日按收盘价计权益。MDD 是日权益的最大回撤（包含 END 当天扣费后的值）。CAGR 是 `E_end^(1/年数) - 1`，年数 = (END - 首个调仓日) 天数 / 365.25。
  Equity is marked daily at closes. MDD is the maximum drawdown of daily equity, including the after-fee END value. CAGR uses (END − D0) days / 365.25.

## 7. 持有期间的公司行动 / Corporate actions while held (`holding_side_events.csv`)
- 分拆：子公司股份按首个正常交易日收盘价卖成现金，并对子公司市值收 c。母公司继续按自己的序列持有。现金留到下一次调仓。
  Spin: child shares are sold at the child's first regular close, paying c on the child value. The parent stays on its series; the cash waits for the next rebalance.
- 特别分红 / 现金收购：拿到现金（现金收购后持仓结束）。/ special dividend or cash deal: cash (a cash deal ends the holding).
- 换股收购：按比例拿到收购方股份（加现金），之后跟收购方的序列（`acq:` 前缀）。
  Stock deal: receive ratio × acquirer shares (+ cash), then follow the acquirer series.
- 改名不算交易：持有的序列直接延续。/ Renames are not deals; the held series continues.
- 执行序列必须和排名价同一只证券：每个月末误差 ≤0.5%（引擎内部检查；本包的执行收盘价已通过这项检查）。
  The execution series must be the same security as the rank prices (≤0.5% at every month-end; the pack's execution closes already passed this check).

---
## 8. 全网格补充（Top2 / Top3 与执行数据）/ Full-grid addendum (Top2/Top3 and execution inputs)
本节和 `rebuild_full.py` 的实现一一对应。用本文件夹的数据，可以把 5,184 条路径全部复现，误差在 1e-9 以内。
This section matches `rebuild_full.py` line by line. With this folder's data, all 5,184 paths reproduce to within 1e-9.

### 8.1 每天的处理顺序 / Daily order of operations (each session d from D0 to END)
1. **持有期间的公司行动**：对每个持仓（按经济键、价格序列），找 `holding_events.csv` 里 `前一交易日 < ex_date <= d` 的事件。
   **Corporate actions while held**: for each holding (by economic key and price series), take events in `holding_events.csv` with previous session < ex_date ≤ d.
   - 只要其中有 `unresolved`，整条路径 UNKNOWN（`corporate_action_unresolved`，fail_ticker 是事件 id）。
     Any `unresolved` makes the path UNKNOWN.
   - 然后按 `action` 处理：`apply` 照下面的规则做；`skip` 表示持有的序列已经是承继或改名后的序列，交易不另算一腿；`ignore` 不处理；`fail` 让路径 UNKNOWN（`corporate_action_unresolved`）。
     Then handle each by `action`: apply, skip (the held series already continues as the successor or renamed series, so no extra leg), ignore, or fail (path UNKNOWN).
   - 分拆：子公司价值 = n×ratio×child_first_close，扣 c，再加上 n×cash，记入现金。母公司股数不变。
     Spin: child value n×ratio×child_first_close minus fee c, plus n×cash, goes to cash. Parent shares are unchanged.
   - 特别分红：n×cash 记入现金。/ Special dividend: n×cash to cash.
   - **持有期间被现金收购**：n×cash_per_share 记入现金，持仓结束。这笔现金不再投资，**留到下一个调仓日**才重新分配。
     **Cash buyout while held**: n×cash_per_share to cash and the holding ends. The cash stays uninvested until the next rebalance.
   - **持有期间被换股收购**：拿到 n×ratio 股收购方（序列 `acq:<收购方>`），加上 n×cash 的现金。如果组合里已经持有收购方，先把已有市值按新序列的价格折成股数再合并。
     **Stock deal while held**: receive n×ratio acquirer shares (series `acq:<acquirer>`) plus n×cash in cash. If the acquirer is already held, its existing value is converted into the new series' units and merged.
2. **按收盘价计价**：每个持仓取 `execution_daily_closes.csv.gz` 中 (price_series, execution_key, d) 的收盘价。缺价时：
   **Mark to market** at each holding's close. If the close is missing:
   - d 晚于 `series_info.csv` 的 series_last_date（序列已结束，没有记录退市事件）：`corporate_action_unresolved`；
     d is after the series' last date (series ended, no documented exit): corporate_action_unresolved;
   - 否则：`daily_gap`。两种都让路径 UNKNOWN。
     otherwise: daily_gap. Both make the path UNKNOWN.
3. **调仓**（d 是调仓日时）：
   **Rebalance** (when d is a rebalance date):
   - 排名有任何无法排名的股票：`rank_missing_unbounded`。/ any unrankable name.
   - 取前 K 只。经济键：AGN 在 2015-03-17 前记作 AGN_LEGACY，之后记作 AGN_PLC；ACT 记作 AGN_PLC；其他股票就用代码本身。如果前 K 只里有两只经济键相同：`duplicate_economic_execution_key`。
     Take the top K. Economic keys: AGN before 2015-03-17 → AGN_LEGACY, from then on → AGN_PLC; ACT → AGN_PLC; others = ticker. Two picks sharing a key → UNKNOWN.
   - 新买入的股票（经济键不在持仓中），查 `execution_availability.csv.gz`，键是 (trade_date, engine_ticker, window_end)，window_end 是下一个调仓日，最后一次是 END：
     New buys (key not held) are looked up in `execution_availability.csv.gz` by (trade_date, engine_ticker, window_end), where window_end is the next rebalance date or END:
     - `ok`：用给出的 price_series；
     - `not_tradable`：`pick_not_tradable`。detail 是 `ticker@交易生效日`，fail_date 是交易生效日。规则是交易生效日落在 (d−60 天, d] 内；
       not_tradable: the deal took effect in (d − 60 days, d]; fail_date is the deal date;
     - `no_verified_series`：`exec_price_missing`。
   - 续持的股票沿用原来的序列。/ Retained names keep their series.
   - **Top2/Top3 每次调仓都重置为等权**：目标 T = Ep/K。费用按不动点 `Ep = E0 − c·Σ|T_i − h_i|` 求解（迭代到 |ΔEp| ≤ 1e-12·E0）。h 是调仓前市值。**续持股票只对调整量收费**，新买按全额收费，卖出也按全额收费。调仓后现金为 0。
     **Top2/Top3 reset to equal weight at every rebalance**: target Ep/K, fee by the fixed point (iterate until |ΔEp| ≤ 1e-12·E0). **Retained names pay fees only on the adjustment**; buys and sells pay on their full value. Cash is 0 after the rebalance.
   - **两次调仓之间权重自然漂移**，不做中途再平衡。期间收到的现金（分拆、分红、收购）留到下一次调仓。
     **Weights drift between rebalances**, with no interim rebalancing. Cash received in between waits for the next rebalance.
4. END 当天收盘清仓，按市值收 c。/ Liquidate at the END close, paying c on market value.
5. 路径跑完后，K=1 才有一项补充检查：如果某次调仓日 d 的入选股票有 `untradable_deal_dates.csv` 里的交易日期落在 (S, d] 内，路径记为 `pick_not_tradable`，fail_date 是 d。
   After the run, for K=1 only: if a pick at rebalance d has a deal date in (S, d], the path is pick_not_tradable with fail_date d.
   - **引擎的小问题**：v17 把 K≥2 的选股串按空格拆开，而股票之间用"|"连接，所以这项检查实际上只对 K=1 生效。
     **Engine quirk**: v17 splits the K≥2 pick string on whitespace while picks are joined with "|", so the check only bites for K=1.
   - 已核对：对全部 K=2/3 路径打开这项检查，**没有任何路径会受影响**（0 条）。这里如实记录引擎的行为。
     Checked: enabling it for all K=2/3 paths affects 0 paths. The engine's behaviour is documented as is.
6. 路径状态：路径中实际用到的公司行动有任何证据等级为 C（包括持有期事件，以及每次调仓的信号日上排名一侧用到的分拆 F），记为 DATA_COMPLETE_PROVISIONAL_C，否则记为 DATA_COMPLETE_CONDITIONAL。
   Status: PROVISIONAL_C if any corporate action the path touched has evidence grade C (holding-side events, plus rank-side spin F events in every frame it rebalanced on); otherwise CONDITIONAL.

### 8.2 UNKNOWN 传递 / UNKNOWN propagation
- 从第一个失败点起，整条路径都是 UNKNOWN，不报告任何收益数字。不跳过、不递补。
  From the first failure on, the whole path is UNKNOWN and no return is reported. No skipping or substitution.
- 格子的 12 相位中位数只算已知的相位，并同时列出已知相位数。
  A cell's 12-phase median uses known phases only, and the number of known phases is reported with it.

### 8.3 同一公司的不同股票类别 / Share classes
- UA/UAA（Under Armour C 类和 A 类）、GOOG/GOOGL（Alphabet C 类和 A 类）是真实存在的两只成分股，价格不同，**按两只独立股票处理**，可以同时入选。
  UA/UAA and GOOG/GOOGL are genuinely separate index members with different prices. **They are treated as separate stocks** and can both be picked.
- 只有经核实为同一证券的重复行才合并（v18：KORS/CPRI）。
  Only rows verified to be the same security are merged (v18: KORS/CPRI).

### 8.4 执行数据文件 / Execution input files
- `execution_availability.csv.gz`：引擎在每个调仓日对每只新买股票查执行序列的结果，附调仓日收盘价和开盘价。开盘价只在和收盘价同一份 FMP 文件、同一口径时才给出，其余写 UNKNOWN，引擎不用。
  Result of every purchase-time series lookup, with the trade-date close and open. The open is given only when it comes from the same FMP file and scale as the close; otherwise UNKNOWN. The engine doesn't use opens.
- `execution_daily_closes.csv.gz`：每个价格序列在所有持有窗口内的每日收盘价。不同窗口的序列做过不同的去调整时，用 `#hash` 区分。
  Daily closes for every price series over all holding windows. Series de-adjusted differently for different windows are told apart by `#hash`.
- `series_info.csv`：每个序列的起止日期。/ first and last date of each series.
- `holding_events.csv`：每个被持有过的（序列, 经济键）的全部公司行动，以及引擎对它的处理方式（apply / skip / ignore / fail）。
  All corporate actions for every held (series, economic key), with the engine's action (apply / skip / ignore / fail).
- `untradable_deal_dates.csv`：补充检查用的交易日期。/ deal dates for the post-run check.
