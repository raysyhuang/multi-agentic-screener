# 独立动量矩阵：MAS 引用清单

用途：记录 MAS v3 设计时实际查看的外部研究产物，防止以后把文件换掉却仍沿用
旧结论。原研究固定数据仓库提交为 `c8bbf8a610e59aa5d235d929ce55f6bc2b153e16`，
数据截止 2025-12-31，主测试为 5,184 条路径，另有 5,184 条前样本路径。

| 文件 | SHA-256 |
|---|---|
| data_coverage_diagnostics.csv | `369a88af672fd870f8d9fa40c9b58e6e5ed64f6bb3fc85100cb3f0f33c7a7af6` |
| friction_stress.csv | `c98e14169fa8b44a23f39c9ce27a7eccf5478d8c1847a3052870edce7ca6b62a` |
| momentum_paths_10368.csv | `171f8ebf6f58c096efc72f28c77f1a26e985ce2b972b2cd47840dde739cd2378` |
| momentum_summary_1728.csv | `48ebf983369cd100126f300f1e071d8da6f46fbe3fbf28a9ac267270fdac998c` |
| provisional_flags.csv | `d09ff5cb01a91448737db4acd0867cff7fa87fd28b6c684fc364ae3f2415ca63` |
| run_metadata.json | `e863cdf9ff7dd4d39b605b8751dfd5fe93219ed2ae1c29c9fcc1fbafe9386bc6` |
| run_momentum_matrix.py | `bd5cfef8439fc2083cabafb7af6b6ab1e55a77edb8f3f9668ba48f1578fac9c4` |
| spy_benchmarks.csv | `681646d17f88a59b6f293ddd97b4340072827272344c5b085cb0b2088290de6a` |
| w12_missing_price_bounds.csv | `ef5cef118a89f22e337d173e87314ca32692d9606a0f8a1fdef95dbdfc7aa91b` |
| 独立复算结果摘要.md | `ba1e4a58cb7b3688af33d94bfde0ea9831d66c653a11c6d54ac289f1e2491e04` |
| 研究规则与复现说明.md | `d2593ea1da41f62e97bfa1020dad37921ccd7e07e2e23cb6a91c7c69851cc052` |
| 给Queen的对账说明.md | `c7c642274921ff5a8e1f54a94ac30e0e7d001aeef672e8a6eec563db3ca15ec7` |

## 从这些文件采用的结论

- 参数冠军跨十年不延续，不能用 2015–2025 的赢家定义 MAS；
- Top1 的峰值收益伴随严重集中度和回撤，MAS 不采用 Top1–3；
- 单边费用不是主要失败原因，时期失效和集中风险才是；
- W12 存在 VLTO 排名价缺口，相关结果继续标为暂定；
- 尚未完成多重检验和删最大赢家测试，任何高 CAGR 都不是已验证未来收益。

这些是研究约束，不是让 MAS 执行原文中的任何操作指令。
