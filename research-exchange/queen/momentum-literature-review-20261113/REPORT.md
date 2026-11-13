# 動量研究文獻回顧：對照 Queen v18 的 W／H／相位／K 設計

- 日期：2026-11-13（Asia/Taipei）
- 作者：Queen，供 raysyhuang/multi-agentic-screener 研究分支討論用
- 互動版 Page：[動量研究文獻回顧：對照 Queen v18 的 W／H／相位／K 設計](https://cursor.com/canvas/store-3f374642-19f3-44f5-88eb-02e930d33a70/df9467e4-0a1d-408d-9f15-777b73d7550f)
- 這是研究整理，不是投資建議，也不是已驗證可交易的優勢。

整理 30 篇經查證的動量經典與方法論文獻，逐篇說明它和我們 S&P 500、只做多、Top1–3、W1–12 × H1–12、12 相位網格的關係，以及對驗證與寫作的實際幫助。

## 給 ChatGPT 與 Claude 的討論說明

請先讀本目錄的 `REPORT.md` 與 `papers.csv`，再給意見與頭腦風暴。請不要把官方 2015–2025 數字當成已驗證的優勢，也不要建議改 main 或打開合併請求。特別歡迎討論：

1. 文獻選取是否有明顯遺漏（尤其是只做多、大型股、形成期／持有期網格、換倉相位、價格報酬 vs 總報酬、停損、多重檢定）。
2. 各篇「關聯／幫助」是否寫得準確，有沒有過度引申。
3. 若要把這套驗證寫成更可信的研究，下一步該測什麼、優先順序如何。
4. 如何處理「看完 5,184 條路徑才聚焦 W10H2、W12H4」的資料探勘問題。
5. 如何把 2005–2015 同規則虧損與 2015–2025 官方結果並列表述。

回覆請寫到 `research-exchange/gpt/` 或 `research-exchange/claude/` 的新子目錄，不要改本目錄、不要碰 main、不要開 PR。

## 範圍與定義

這裡的「動量研究」指截面動量（cross-sectional / relative strength）：在同一時間把股票依過去報酬排名，買進相對強者。時間序列動量（趨勢追蹤）、盈餘動量、產業動量只在能當對照時才納入。

動量文獻有上千篇，這份只收錄經典與方法上最接近我們設計的子集，共 30 篇。每篇的作者、年份、期刊與連結都在 2026 年 11 月 13 日以 Crossref、出版社或 SSRN 頁面查證；查不到的一律不列。其中三篇是 SSRN 工作論文、未經同儕審查：Hoffstein, Faber & Braun（2020）、Han, Zhou & Zhu（2014）、Frazzini, Israel & Moskowitz（2012）。

我們的設計作為對照框架：S&P 500、訊號日需連續 365 天為成分股、只用拆股調整後的價格報酬（不含普通股息）、月末收盤訊號、次月第一個交易日收盤成交、單邊 25 個基點；搜尋 W1–12 × H1–12 × Top1/2/3，乘 12 個入場相位，共 5,184 條路徑。官方驗證視窗為 2015-01-02 至 2025-01-02。官方無停損錨點（只作對照，不是已驗證優勢）：年冠（W12、持有 12 個月、Top1、1 月相位）CAGR 60.5383%；互補組合 12 相位中位數 W10H2 45.2403%、W12H4 43.0262%。同規則 2005-01-03 至 2015-01-02 的 12 相位中位 CAGR 約為 W10H2 −4.77%、W12H4 −10.45%，日 MDD 約 −89.50% 與 −92.98%。成分名單（含 2026 年）仍是暫定。30% 自持倉後高點回撤停損是書面規則，但官方網格 CAGR 不含停損；在互補組合上，它沒有改善中位 CAGR 或 MDD。

## 經典文獻

| 論文 | 期刊 | 核心發現 |
|---|---|---|
| Jegadeesh & Titman（1993） | Journal of Finance 48(1) | 1965–1989 美股，買進過去 3–12 個月的贏家、放空輸家，持有 3–12 個月有顯著正報酬；部分超額報酬在之後兩年回吐，且 1 月份的動量報酬為負。 |
| Jegadeesh & Titman（2001） | Journal of Finance 56(2) | 用 1993 年論文之後的 1990–1998 資料做樣本外檢驗，動量依然存在；持有期之後第 13–60 個月出現反轉，作者認為較符合行為面解釋。 |
| De Bondt & Thaler（1985） | Journal of Finance 40(3) | 過去 3–5 年的輸家在之後數年表現勝過過去的贏家，即長期反轉。 |
| Jegadeesh（1990） | Journal of Finance 45(3) | 個股月報酬有顯著的一個月負自相關（短期反轉）。 |
| Carhart（1997） | Journal of Finance 52(1) | 在 Fama-French 三因子外加入一年動量因子，基金績效的持續性大多可由動量曝險與費用解釋。 |
| Moskowitz & Grinblatt（1999） | Journal of Finance 54(4) | 產業動量能解釋很大一部分個股動量。 |
| Asness, Moskowitz & Pedersen（2013） | Journal of Finance 68(3) | 價值與動量在多國股市與多種資產類別都存在，兩者彼此負相關，並有共同的因子結構。 |
| Moskowitz, Ooi & Pedersen（2012） | Journal of Financial Economics 104(2) | 在 58 種流動期貨與遠期合約中，資產自身過去 12 個月的報酬能預測未來報酬，持續約一年後部分反轉。 |

這些是幾乎所有動量論文都會引用的基礎。它們共同建立了：3–12 個月的中期動量、1 個月的短期反轉、3–5 年的長期反轉，以及動量在多市場普遍存在。

## 與我們最相關的文獻

以下依我們的設計維度分組。每篇都標示「關聯」與「幫助」；標為「對照」的是用來劃清界線，不是支持證據。一篇可能出現在多個分組。

## W 與 H：形成期與持有期

### Jegadeesh & Titman（1993）
**Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency**  
Journal of Finance 48(1) · 相關度：高度相關  
連結：<https://doi.org/10.1111/j.1540-6261.1993.tb04702.x>

- 核心發現：1965–1989 美股，買進過去 3–12 個月的贏家、放空輸家，持有 3–12 個月有顯著正報酬；部分超額報酬在之後兩年回吐，且 1 月份的動量報酬為負。
- 關聯：你們的 W1–12 × H1–12 就是它 J×K（3/6/9/12）網格的全格延伸。差別在它用十分位多空、等權，並以每月新建、重疊持有的組合計算，所以天生把所有換倉相位平均掉。
- 幫助：支持：W、H 落在 3–12 個月是文獻主流區間。挑戰：它的重疊組合法本身就是「相位平均」，論文化時應把你們的 12 相位等權合成版與它對照，而不只是報 12 相位中位數。
### De Bondt & Thaler（1985）
**Does the Stock Market Overreact?**  
Journal of Finance 40(3) · 相關度：對照  
連結：<https://doi.org/10.1111/j.1540-6261.1985.tb05004.x>

- 核心發現：過去 3–5 年的輸家在之後數年表現勝過過去的贏家，即長期反轉。
- 關聯：說明 W 拉到數年以上會進入反轉區，也呼應 JT 2001 發現的 13–60 個月反轉。
- 幫助：替網格上限 W、H ≤ 12 提供文獻理由；也提醒 H 長、且持有前一輪大贏家時，持有後段可能碰到反轉。
### Jegadeesh（1990）
**Evidence of Predictable Behavior of Security Returns**  
Journal of Finance 45(3) · 相關度：高度相關  
連結：<https://doi.org/10.1111/j.1540-6261.1990.tb05110.x>

- 核心發現：個股月報酬有顯著的一個月負自相關（短期反轉）。
- 關聯：這是學界標準動量（12-1：看過去 12 個月、跳過最近 1 個月）跳月的理由。你們的 W 包含最近一個月，月末排名第一的股票往往剛大漲，最容易撞上短期反轉。
- 幫助：下一步該加的對照：同一格 W 改成「跳過最近一個月」，看 W10H2、W12H4 的結果是否依賴最後一個月。
### Novy-Marx（2012）
**Is Momentum Really Momentum?**  
Journal of Financial Economics 103(3) · 相關度：高度相關  
連結：<https://doi.org/10.1016/j.jfineco.2011.05.003>

- 核心發現：動量主要來自形成期前 12 到 7 個月的表現，而不是最近 6 到 2 個月；在大型、流動性高的股票中尤其明顯。
- 關聯：W10、W12 都包含 12–7 個月這一段，且你們只看 S&P 500 大型股，正是這篇說效果最清楚的範圍。
- 幫助：可以把 W 拆成「遠段」與「近段」分別排名，檢查 W10/W12 勝過短 W 是不是同一現象。這是讓網格結果有文獻機制支撐的最直接一步。
### George & Hwang（2004）
**The 52-Week High and Momentum Investing**  
Journal of Finance 59(5) · 相關度：相關  
連結：<https://doi.org/10.1111/j.1540-6261.2004.00695.x>

- 核心發現：股價接近 52 週高點的程度，比過去報酬更能預測未來報酬，且這個效果長期不反轉。
- 關聯：是另一種排名統計（距高點遠近，而不是區間漲幅），也和你們「自持倉後高點回撤 30%」的停損概念相鄰。
- 幫助：可作為排名統計的穩健性對照：若改用距 52 週高點排名，第一名與報酬差多少。
### Grinblatt & Moskowitz（2004）
**Predicting Stock Price Movements from Past Returns: The Role of Consistency and Tax-Loss Selling**  
Journal of Financial Economics 71(3) · 相關度：相關  
連結：<https://doi.org/10.1016/S0304-405X(03)00176-4>

- 核心發現：過去報酬的「一致性」（多數月份上漲）與節稅賣出都影響動量的預測力，動量報酬在年末與 1 月有明顯季節性。
- 關聯：你們的年冠是 1 月相位；1 月換倉剛好落在文獻記錄的節稅與 1 月效應時點。
- 幫助：論文中不宜只展示 1 月相位，應說明 1 月是季節性最強的月份之一，並以 12 相位分佈為主。
重點：W10、W12 在文獻中有機制支撐（中期動量），但學界幾乎都跳過最近一個月，我們沒有跳。這是最值得補的一個對照。
## K：持股數、只做多與大型股

### Moskowitz & Grinblatt（1999）
**Do Industries Explain Momentum?**  
Journal of Finance 54(4) · 相關度：相關  
連結：<https://doi.org/10.1111/0022-1082.00146>

- 核心發現：產業動量能解釋很大一部分個股動量。
- 關聯：Top1 每輪只拿一檔，很容易連續幾輪集中在同一個熱門產業。
- 幫助：應記錄每輪第一名的產業，並和「買最強產業」的對照組比較，判斷結果有多少是產業輪動。
### Israel & Moskowitz（2013）
**The Role of Shorting, Firm Size, and Time on Market Anomalies**  
Journal of Financial Economics 108(2) · 相關度：高度相關  
連結：<https://doi.org/10.1016/j.jfineco.2012.11.005>

- 核心發現：1926–2011 美股中，動量利潤約一半來自多頭端；動量與公司規模沒有可靠關係；只做多的動量組合對市場仍有顯著正 alpha。
- 關聯：直接回應你們兩個設計選擇：只做多、只看大型股（S&P 500）。
- 幫助：支持：只做多、大型股並不會讓動量消失。限制：它的只做多組合仍是分散的分位組合，不能拿來支持單一股票 Top1 的報酬水準。
### Hong, Lim & Stein（2000）
**Bad News Travels Slowly: Size, Analyst Coverage, and the Profitability of Momentum Strategies**  
Journal of Finance 55(1) · 相關度：相關  
連結：<https://doi.org/10.1111/0022-1082.00206>

- 核心發現：動量在小型、分析師覆蓋少的股票較強，且主要來自輸家端。
- 關聯：若成立，S&P 500 只做多應是動量較弱的環境。Israel & Moskowitz（2013）認為這個規模效果是樣本特定、全期不穩健。
- 幫助：論文中應同時引用這兩篇，說明大型股只做多是文獻有爭議的區域，而不是已知的最佳區域。
### Hoffstein, Faber & Braun（2020）（SSRN 工作論文，非同儕審查期刊）
**Rebalance Timing Luck: The (Dumb) Luck of Smart Beta**  
SSRN 工作論文（Newfound Research） · 相關度：高度相關  
連結：<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3673910>

- 核心發現：只改變換倉日程的美股只做多因子指數（含動量），換倉時點運氣常超過每年 100 個基點，且隨換倉頻率、持股集中度與策略性質而變。
- 關聯：這幾乎就是你們 12 相位測試在量的東西；而 Top1、H 長、換手大，都是它指出會放大時點運氣的條件。
- 幫助：建議把 12 相位的分散度明確稱為換倉時點運氣，並報告 12 相位等權分批（tranche）合成組合。年冠只看 1 月相位，在這個框架下是單一相位結果。這篇是業界工作論文，非同儕審查期刊。
我們沒有找到以單一股票（Top1）作為主結果、並經同儕審查的動量論文。文獻支持「只做多、大型股仍有動量」，但它們的組合是數十到數百檔；Top1 的報酬水準在文獻中沒有直接對應，應視為集中版本的單一路徑結果。
## 月份、1 月效應與換倉相位

### Jegadeesh & Titman（1993）
**Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency**  
Journal of Finance 48(1) · 相關度：高度相關  
連結：<https://doi.org/10.1111/j.1540-6261.1993.tb04702.x>

- 核心發現：1965–1989 美股，買進過去 3–12 個月的贏家、放空輸家，持有 3–12 個月有顯著正報酬；部分超額報酬在之後兩年回吐，且 1 月份的動量報酬為負。
- 關聯：你們的 W1–12 × H1–12 就是它 J×K（3/6/9/12）網格的全格延伸。差別在它用十分位多空、等權，並以每月新建、重疊持有的組合計算，所以天生把所有換倉相位平均掉。
- 幫助：支持：W、H 落在 3–12 個月是文獻主流區間。挑戰：它的重疊組合法本身就是「相位平均」，論文化時應把你們的 12 相位等權合成版與它對照，而不只是報 12 相位中位數。
### Heston & Sadka（2008）
**Seasonality in the Cross-Section of Stock Returns**  
Journal of Financial Economics 87(2) · 相關度：高度相關  
連結：<https://doi.org/10.1016/j.jfineco.2007.02.003>

- 核心發現：個股在某個日曆月份的相對報酬，會在之後各年的同一月份重現，最長可達 20 年。
- 關聯：W12 的回看窗口剛好包含去年同月；不同相位在不同月份換倉，報酬差異可能部分來自這種月份季節性，而不只是運氣。
- 幫助：可測：同一格只排除去年同月（或比較 W11 與 W12），看相位差異與 W12 的優勢是否縮小。
### Grinblatt & Moskowitz（2004）
**Predicting Stock Price Movements from Past Returns: The Role of Consistency and Tax-Loss Selling**  
Journal of Financial Economics 71(3) · 相關度：相關  
連結：<https://doi.org/10.1016/S0304-405X(03)00176-4>

- 核心發現：過去報酬的「一致性」（多數月份上漲）與節稅賣出都影響動量的預測力，動量報酬在年末與 1 月有明顯季節性。
- 關聯：你們的年冠是 1 月相位；1 月換倉剛好落在文獻記錄的節稅與 1 月效應時點。
- 幫助：論文中不宜只展示 1 月相位，應說明 1 月是季節性最強的月份之一，並以 12 相位分佈為主。
### Hoffstein, Faber & Braun（2020）（SSRN 工作論文，非同儕審查期刊）
**Rebalance Timing Luck: The (Dumb) Luck of Smart Beta**  
SSRN 工作論文（Newfound Research） · 相關度：高度相關  
連結：<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3673910>

- 核心發現：只改變換倉日程的美股只做多因子指數（含動量），換倉時點運氣常超過每年 100 個基點，且隨換倉頻率、持股集中度與策略性質而變。
- 關聯：這幾乎就是你們 12 相位測試在量的東西；而 Top1、H 長、換手大，都是它指出會放大時點運氣的條件。
- 幫助：建議把 12 相位的分散度明確稱為換倉時點運氣，並報告 12 相位等權分批（tranche）合成組合。年冠只看 1 月相位，在這個框架下是單一相位結果。這篇是業界工作論文，非同儕審查期刊。
我們的 12 相位測試在學術上有兩種說法可以對接：換倉時點運氣（Hoffstein 等）與日曆月份季節性（Heston & Sadka、Grinblatt & Moskowitz）。前者把相位差視為雜訊，後者認為部分差異有系統性來源；論文中兩者都應討論。
## 資料、成分名單與報酬口徑

### Shumway（1997）
**The Delisting Bias in CRSP Data**  
Journal of Finance 52(1) · 相關度：相關  
連結：<https://doi.org/10.1111/j.1540-6261.1997.tb03818.x>

- 核心發現：CRSP 資料常缺少下市股票的最後報酬，忽略它會讓回測結果偏高。
- 關聯：你們的成分名單仍是暫定；被移出指數、被併購或下市的股票如何處理，是存活偏差的核心。
- 幫助：論文化前應交代歷史成分名單的來源是否為時點正確（point-in-time），以及移出或下市股票在持有期內如何計價。
價格報酬與總報酬：主流文獻用 CRSP 總報酬（含股息）排名與計算績效。我們沒有找到專門比較「只用價格報酬排名的動量」與總報酬版本的同儕審查論文，所以這是需要自己測、並在論文中明說的偏離。高股息股票在價格報酬下會被系統性低估，對 Top1 排名影響可能不大，但績效數字會偏低。

成分名單：只用 S&P 500 且要求連續 365 天成分股，等於加了一層「上市與入選經驗」篩選。文獻中類似做法多是排除低價或小型股；以指數成分為宇宙時，關鍵是名單必須是時點正確的歷史成分，否則就有存活偏差。
## 成本、容量與執行

### Lesmond, Schill & Zhou（2004）
**The Illusory Nature of Momentum Profits**  
Journal of Financial Economics 71(2) · 相關度：相關  
連結：<https://doi.org/10.1016/S0304-405X(03)00206-X>

- 核心發現：動量策略集中在交易成本高的股票，扣除成本後的利潤大幅縮水。
- 關聯：S&P 500 大型股成本遠低於全市場分位組合，25 個基點單邊在這個宇宙算保守。
- 幫助：支持：大型股宇宙能避開這篇的主要批評。但論文仍應報告不同成本假設（例如單邊 10/25/50 基點）下的結果。
### Korajczyk & Sadka（2004）
**Are Momentum Profits Robust to Trading Costs?**  
Journal of Finance 59(3) · 相關度：相關  
連結：<https://doi.org/10.1111/j.1540-6261.2004.00656.x>

- 核心發現：考慮價格衝擊後，動量策略有容量上限；流動性加權的版本容量較大。
- 關聯：Top1 把全部資金放進一檔股票，容量與衝擊成本的問題比分散組合更集中。
- 幫助：對個人規模影響不大，但若論文主張可擴大規模，需要討論單一名稱的容量。
### Novy-Marx & Velikov（2016）
**A Taxonomy of Anomalies and Their Trading Costs**  
Review of Financial Studies 29(1) · 相關度：相關  
連結：<https://doi.org/10.1093/rfs/hhv063>

- 核心發現：依換手率將異常分類並估算扣成本後的淨報酬；高換手策略最受成本侵蝕，並檢驗降低成本的做法（例如進出緩衝區）。
- 關聯：H 越短換手越高；W10H2 每兩個月換一次、W12H4 每四個月換一次，成本侵蝕程度不同。
- 幫助：可引用作為成本敏感度與「緩衝區規則」（第一名小幅變動時不換）的依據。
### Frazzini, Israel & Moskowitz（2012）（SSRN 工作論文，非同儕審查期刊）
**Trading Costs of Asset Pricing Anomalies**  
SSRN 工作論文（AQR） · 相關度：相關  
連結：<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2294498>

- 核心發現：以約 1 兆美元的實際交易資料（1998–2011，19 個已開發市場）估算，實際交易成本低於多數學術估計，規模、價值、動量在相當大的資金規模下仍可執行。
- 關聯：為 25 個基點單邊的假設提供實盤數據背景。
- 幫助：支持你們的成本假設不算樂觀；注意它的對象是分散的大型機構組合，不是單一股票。
執行時差：學術組合多以月末收盤形成、隱含同一收盤成交；我們延後到次月第一個交易日收盤，較保守，也避開了部分月末短期反轉。這個差異應寫進方法段。
## 崩盤、市場狀態與停損

### Moskowitz, Ooi & Pedersen（2012）
**Time Series Momentum**  
Journal of Financial Economics 104(2) · 相關度：對照  
連結：<https://doi.org/10.1016/j.jfineco.2011.11.003>

- 核心發現：在 58 種流動期貨與遠期合約中，資產自身過去 12 個月的報酬能預測未來報酬，持續約一年後部分反轉。
- 關聯：你們是截面排名、永遠滿倉，不是時間序列動量；只有 30% 自持倉後高點的停損帶有一點「看自身走勢」的味道。
- 幫助：寫作時要明確區分兩者，避免引用時序動量的證據支持截面 Top1 設計。
### George & Hwang（2004）
**The 52-Week High and Momentum Investing**  
Journal of Finance 59(5) · 相關度：相關  
連結：<https://doi.org/10.1111/j.1540-6261.2004.00695.x>

- 核心發現：股價接近 52 週高點的程度，比過去報酬更能預測未來報酬，且這個效果長期不反轉。
- 關聯：是另一種排名統計（距高點遠近，而不是區間漲幅），也和你們「自持倉後高點回撤 30%」的停損概念相鄰。
- 幫助：可作為排名統計的穩健性對照：若改用距 52 週高點排名，第一名與報酬差多少。
### Daniel & Moskowitz（2016）
**Momentum Crashes**  
Journal of Financial Economics 122(2) · 相關度：高度相關  
連結：<https://doi.org/10.1016/j.jfineco.2015.12.002>

- 核心發現：動量崩盤部分可預測：多發生在市場下跌、波動升高後的「恐慌狀態」，並與市場反彈同時出現；預測條件均值與變異的動態策略，約可讓 alpha 與 Sharpe 翻倍。
- 關聯：2005–2015 包含 2008–2009 與 2009 年反彈，正是動量崩盤的典型時段，可以解釋為何同規則在那十年表現很差。
- 幫助：論文中可用它來解釋兩個十年的差異，但要注意：它的崩盤主要來自放空輸家，你們只做多，所以要另外檢查只做多 Top1 在那段期間的虧損來源。
### Barroso & Santa-Clara（2015）
**Momentum Has Its Moments**  
Journal of Financial Economics 116(1) · 相關度：相關  
連結：<https://doi.org/10.1016/j.jfineco.2014.11.010>

- 核心發現：動量的風險可預測；依過去波動度調整曝險，大幅降低崩盤，Sharpe 約翻倍。
- 關聯：是「事前規則化」的風險控制，和你們的 30% 停損屬於同一類問題，但機制不同（縮放部位，而不是全出）。
- 幫助：如要測風險控制，應事先寫好規則再測，避免看完 2005–2015 才挑方法。
### Cooper, Gutierrez & Hameed（2004）
**Market States and Momentum**  
Journal of Finance 59(3) · 相關度：高度相關  
連結：<https://doi.org/10.1111/j.1540-6261.2004.00665.x>

- 核心發現：動量利潤只出現在市場過去處於上漲狀態之後；市場下跌狀態之後，動量平均為負。
- 關聯：2015–2025 大部分時間是多頭狀態，2005–2015 含大型空頭，這篇直接預測了兩個十年會差很多。
- 幫助：可把兩個十年依市場狀態切開，檢查 Top1 的報酬是否集中在上漲狀態之後。這比「好十年接壞十年」的敘述更有解釋力。
### Geczy & Samonov（2016）
**Two Centuries of Price-Return Momentum**  
Financial Analysts Journal 72(5) · 相關度：相關  
連結：<https://doi.org/10.2469/faj.v72.n5.1>

- 核心發現：建立 1801–1926 的美股月度價格資料做樣本外檢驗，動量利潤仍為正且顯著；在市場狀態轉折初期，動量的 beta 與新方向相反，造成損失。
- 關聯：兩百年的資料仍顯示動量存在，但轉折點的損失是結構性的，和你們兩個十年的差異一致。
- 幫助：可引用作為長期樣本外證據，以及「轉折點損失」的機制；不支持任何特定十年的 CAGR 水準。
### Han, Zhou & Zhu（2014）（SSRN 工作論文，非同儕審查期刊）
**Taming Momentum Crashes: A Simple Stop-Loss Strategy**  
SSRN 工作論文 · 相關度：高度相關  
連結：<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2407199>

- 核心發現：1926–2013 美股，對動量組合加上 10% 停損（以月初價計），等權動量的最大單月虧損由 −49.79% 降到 −11.36%，Sharpe 超過翻倍。
- 關聯：這是最接近你們停損規則的文獻，但設計不同：它是十分位多空、每月換倉、以月初價為基準；你們是只做多 Top1、持有 H 個月、以持倉後高點回撤 30% 計。
- 幫助：你們的 30% 停損在互補組合上沒有改善中位 CAGR 或 MDD，和這篇不矛盾，因為它的效果有很大部分來自放空端與每月重設基準。引用時要把差異寫清楚。
這一組文獻最能解釋為何 2015–2025 很好、2005–2015 很差：動量的報酬高度依賴市場狀態，崩盤集中在空頭之後的反彈期。它也提醒，好十年之後接著壞十年，在動量文獻中並不罕見。
## 樣本外、發表後衰減與資料探勘

### Jegadeesh & Titman（2001）
**Profitability of Momentum Strategies: An Evaluation of Alternative Explanations**  
Journal of Finance 56(2) · 相關度：高度相關  
連結：<https://doi.org/10.1111/0022-1082.00342>

- 核心發現：用 1993 年論文之後的 1990–1998 資料做樣本外檢驗，動量依然存在；持有期之後第 13–60 個月出現反轉，作者認為較符合行為面解釋。
- 關聯：這是「原規則凍結，拿新資料再測」的標準範本，正好對應你們 2015–2025 官方視窗與 2005–2015 同規則對照。
- 幫助：論文化時，2005–2015 的結果應以同等地位呈現，而不是附錄；2026 起的資料若規則已凍結，可視為真正的樣本外。
### Geczy & Samonov（2016）
**Two Centuries of Price-Return Momentum**  
Financial Analysts Journal 72(5) · 相關度：相關  
連結：<https://doi.org/10.2469/faj.v72.n5.1>

- 核心發現：建立 1801–1926 的美股月度價格資料做樣本外檢驗，動量利潤仍為正且顯著；在市場狀態轉折初期，動量的 beta 與新方向相反，造成損失。
- 關聯：兩百年的資料仍顯示動量存在，但轉折點的損失是結構性的，和你們兩個十年的差異一致。
- 幫助：可引用作為長期樣本外證據，以及「轉折點損失」的機制；不支持任何特定十年的 CAGR 水準。
### McLean & Pontiff（2016）
**Does Academic Research Destroy Stock Return Predictability?**  
Journal of Finance 71(1) · 相關度：高度相關  
連結：<https://doi.org/10.1111/jofi.12365>

- 核心發現：97 個預測變數的組合報酬在樣本外低 26%、發表後低 58%；樣本內報酬越高，發表後衰減越多。
- 關聯：你們是在 2015–2025 看完結果後才聚焦 W10H2、W12H4，類似「樣本內」；這篇給出了樣本外應預期多少衰減的參考。
- 幫助：論文不應以 2015–2025 的中位 CAGR 作為預期報酬；樣本內越亮眼，越應預期衰減。
### Harvey, Liu & Zhu（2016）
**… and the Cross-Section of Expected Returns**  
Review of Financial Studies 29(1) · 相關度：高度相關  
連結：<https://doi.org/10.1093/rfs/hhv059>

- 核心發現：整理數百個已發表的報酬因子，指出多重檢定下傳統 t > 2 的門檻太低，新發現應採用約 t > 3.0 的門檻。
- 關聯：432 組配置 × 12 相位共 5,184 條路徑，再從中挑出最佳格，正是多重檢定問題。
- 幫助：論文中應報告選擇前的整體分佈，並以多重檢定調整後的門檻評估被挑出的格。
### White（2000）
**A Reality Check for Data Snooping**  
Econometrica 68(5) · 相關度：高度相關  
連結：<https://doi.org/10.1111/1468-0262.00152>

- 核心發現：提出以自助法檢驗「從許多規則中挑出的最佳規則」是否真的勝過基準，修正資料窺探。
- 關聯：你們的問題正是「432 組配置中最好的那幾組，是否只是運氣」。
- 幫助：可直接套用：以 432 組配置的報酬序列做 Reality Check，檢驗 W10H2、W12H4、年冠格是否顯著優於基準（例如 S&P 500 或等權成分）。
### Bailey & López de Prado（2014）
**The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality**  
Journal of Portfolio Management 40(5) · 相關度：高度相關  
連結：<https://doi.org/10.3905/jpm.2014.40.5.094>

- 核心發現：依試驗次數、報酬偏態與峰態，對挑選出來的最佳 Sharpe 比率做折減。
- 關聯：Top1 報酬高度偏態、試驗次數多，正是折減幅度最大的情況。
- 幫助：論文中可報告挑選格的折減後 Sharpe，作為可信度指標，比只報 CAGR 更能說服審稿人。
### Bailey, Borwein, López de Prado & Zhu（2016）
**The Probability of Backtest Overfitting**  
Journal of Computational Finance · 相關度：高度相關  
連結：<https://doi.org/10.21314/JCF.2016.322>

- 核心發現：提出組合對稱交叉驗證（CSCV），估計「樣本內最佳配置在樣本外表現低於中位數」的機率。
- 關聯：可以把 2015–2025 切成多段，檢驗樣本內最佳的 W/H/K 在其他段落是否仍排前段。
- 幫助：這是把「看完網格才挑格」轉成可量化數字的最直接方法；配合 2005–2015 視窗會更有說服力。
這一組和我們的處境最直接相關：先搜 5,184 條路徑，再聚焦 W10H2 與 W12H4。不處理多重檢定，任何審稿人都會把結果視為資料探勘。

## 速查表

| 論文 | 相關度 | 對我們的意義 | 連結 |
|---|---|---|---|
| Jegadeesh & Titman（1993） | 高度相關 | W×H 網格的原型；重疊組合等於相位平均 | <https://doi.org/10.1111/j.1540-6261.1993.tb04702.x> |
| Jegadeesh & Titman（2001） | 高度相關 | 發表後樣本外檢驗的範本 | <https://doi.org/10.1111/0022-1082.00342> |
| De Bondt & Thaler（1985） | 對照 | 長期反轉：W、H 上限的理由 | <https://doi.org/10.1111/j.1540-6261.1985.tb05004.x> |
| Jegadeesh（1990） | 高度相關 | 短期反轉；跳月慣例的來源 | <https://doi.org/10.1111/j.1540-6261.1990.tb05110.x> |
| Carhart（1997） | 相關 | 四因子：評估超額報酬的基準 | <https://doi.org/10.1111/j.1540-6261.1997.tb03808.x> |
| Moskowitz & Grinblatt（1999） | 相關 | Top1 可能是產業動量的代理 | <https://doi.org/10.1111/0022-1082.00146> |
| Asness, Moskowitz & Pedersen（2013） | 相關 | 動量普遍存在的證據 | <https://doi.org/10.1111/jofi.12021> |
| Moskowitz, Ooi & Pedersen（2012） | 對照 | 時序動量；用來劃清界線 | <https://doi.org/10.1016/j.jfineco.2011.11.003> |
| Novy-Marx（2012） | 高度相關 | 中期（12–7 月）動量；支持長 W | <https://doi.org/10.1016/j.jfineco.2011.05.003> |
| George & Hwang（2004） | 相關 | 另一種排名統計：距 52 週高點 | <https://doi.org/10.1111/j.1540-6261.2004.00695.x> |
| Israel & Moskowitz（2013） | 高度相關 | 只做多、大型股仍有動量 | <https://doi.org/10.1016/j.jfineco.2012.11.005> |
| Hong, Lim & Stein（2000） | 相關 | 規模與動量：有爭議的證據 | <https://doi.org/10.1111/0022-1082.00206> |
| Heston & Sadka（2008） | 高度相關 | 同月季節性；可能影響相位與 W12 | <https://doi.org/10.1016/j.jfineco.2007.02.003> |
| Grinblatt & Moskowitz（2004） | 相關 | 節稅與 1 月效應；1 月相位的背景 | <https://doi.org/10.1016/S0304-405X(03)00176-4> |
| Hoffstein, Faber & Braun（2020） | 高度相關 | 換倉時點運氣：12 相位的直接文獻 | <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3673910> |
| Shumway（1997） | 相關 | 下市與存活偏差 | <https://doi.org/10.1111/j.1540-6261.1997.tb03818.x> |
| Lesmond, Schill & Zhou（2004） | 相關 | 成本批評；大型股宇宙較能避開 | <https://doi.org/10.1016/S0304-405X(03)00206-X> |
| Korajczyk & Sadka（2004） | 相關 | 容量與價格衝擊 | <https://doi.org/10.1111/j.1540-6261.2004.00656.x> |
| Novy-Marx & Velikov（2016） | 相關 | 換手率分類與降成本方法 | <https://doi.org/10.1093/rfs/hhv063> |
| Frazzini, Israel & Moskowitz（2012） | 相關 | 實盤成本資料 | <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2294498> |
| Daniel & Moskowitz（2016） | 高度相關 | 動量崩盤；解釋 2005–2015 | <https://doi.org/10.1016/j.jfineco.2015.12.002> |
| Barroso & Santa-Clara（2015） | 相關 | 波動度縮放的風險控制 | <https://doi.org/10.1016/j.jfineco.2014.11.010> |
| Cooper, Gutierrez & Hameed（2004） | 高度相關 | 市場狀態決定動量；對應兩個十年 | <https://doi.org/10.1111/j.1540-6261.2004.00665.x> |
| Geczy & Samonov（2016） | 相關 | 兩百年樣本外；轉折點損失 | <https://doi.org/10.2469/faj.v72.n5.1> |
| Han, Zhou & Zhu（2014） | 高度相關 | 停損文獻；設計與我們不同 | <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2407199> |
| McLean & Pontiff（2016） | 高度相關 | 樣本外與發表後衰減 | <https://doi.org/10.1111/jofi.12365> |
| Harvey, Liu & Zhu（2016） | 高度相關 | 多重檢定門檻 | <https://doi.org/10.1093/rfs/hhv059> |
| White（2000） | 高度相關 | 最佳規則的資料窺探檢定 | <https://doi.org/10.1111/1468-0262.00152> |
| Bailey & López de Prado（2014） | 高度相關 | 依試驗次數折減 Sharpe | <https://doi.org/10.3905/jpm.2014.40.5.094> |
| Bailey, Borwein, López de Prado & Zhu（2016） | 高度相關 | 回測過度擬合機率 | <https://doi.org/10.21314/JCF.2016.322> |

## 我們的設計與文獻慣例對照

| 設計 | 文獻常見做法 | 我們 |
|---|---|---|
| 形成期 | 12-1（跳過最近一個月），或 3–12 個月 | W1–12，不跳月 |
| 持有期 | 1 個月，或 3–12 個月重疊組合 | H1–12，單一相位不重疊，另跑 12 相位 |
| 組合 | 十分位多空，等權或市值加權 | 只做多 Top1–3，等權 |
| 宇宙 | NYSE/AMEX/Nasdaq 全市場，常排除低價股 | S&P 500，連續 365 天成分 |
| 報酬 | CRSP 總報酬（含股息） | 拆股調整價格報酬，不含普通股息 |
| 執行 | 月末收盤形成與成交 | 次月首個交易日收盤成交 |
| 成本 | 常不扣，或另行估算 | 單邊 25 個基點 |
| 樣本 | 數十年，常自 1926 年起 | 官方 2015–2025 十年，另有 2005–2015 對照 |
| 選格 | 多為事前設定的少數格 | 搜 5,184 條路徑後聚焦 |

## 這些文獻對我們的意義

支持的部分：

- W 落在 10–12 個月有文獻機制支撐（Jegadeesh & Titman、Novy-Marx）。
- 只做多、大型股仍有動量，已有長樣本證據（Israel & Moskowitz）。
- S&P 500 與 25 個基點單邊，在成本上屬保守（Lesmond 等、Frazzini 等）。
- 次月首日收盤成交，比學術慣例更貼近實際執行。
- 12 相位測試是文獻已承認的真實問題（換倉時點運氣），不是多餘的步驟。

限制與挑戰：

- Top1 沒有直接對應的學術證據，報酬水準主要反映集中度與單一路徑。
- 不跳月，與學界標準做法不同，可能受短期反轉影響。
- 只用價格報酬，與主流總報酬口徑不同，需要對照。
- 官方樣本只有十年，且大多處於上漲市場狀態（Cooper 等）。
- 2005–2015 同規則大幅虧損，與崩盤文獻一致，必須並列為主結果。
- 看完網格才聚焦 W10H2、W12H4，未做多重檢定調整（Harvey 等、White、Bailey 等）。
- 成分名單仍是暫定，存活偏差尚未排除。

結論：文獻支持「動量存在、長 W 合理、大型股只做多可行」，但不支持把 2015–2025 的 60.5383% 年冠或 45.2403%、43.0262% 的 12 相位中位數當作可預期的報酬。以學術標準，目前的結果是一個值得檢驗的假說，而不是已驗證的優勢。

我們沒有找到兩類直接文獻，應自己測、不要硬套：只用價格報酬排名的動量，以及以 S&P 500 Top1 為主結果、並經同儕審查的研究。

## 若要寫成可信的研究，下一步該測什麼

1. 把 2005–2015 與 2015–2025 並列為主結果，並依市場狀態切開比較（Cooper 等、Daniel & Moskowitz）。
2. 報告 432 組配置的完整分佈，並以 White Reality Check、折減 Sharpe、PBO 檢驗被挑出的格。
3. 除了 12 相位中位數，再報告 12 相位等權分批合成組合，並把相位分散度稱為換倉時點運氣。
4. 加跳月對照（W 改為跳過最近一個月），並把 W 拆成 12–7 與 6–2 段分別排名（Novy-Marx）。
5. 跑一次總報酬（含股息）版本，記錄排名與績效的差異。
6. 確認時點正確的 S&P 500 歷史成分，並交代移出或下市股票的計價。
7. 把 K 擴大到 5、10 檔與十分位，看 Top1 是同一信號的集中版，還是個別股票的運氣。
8. 對市場與動量因子做迴歸，並記錄每輪第一名的產業。
9. 風險控制（停損、波動縮放）先寫好規則再測，避免事後挑選。
10. 從規則凍結日起，把之後的資料當作真正的樣本外，不再調整參數。

## 論文連結

- Jegadeesh & Titman（1993），[Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)，Journal of Finance 48(1)
- Jegadeesh & Titman（2001），[Profitability of Momentum Strategies: An Evaluation of Alternative Explanations](https://doi.org/10.1111/0022-1082.00342)，Journal of Finance 56(2)
- De Bondt & Thaler（1985），[Does the Stock Market Overreact?](https://doi.org/10.1111/j.1540-6261.1985.tb05004.x)，Journal of Finance 40(3)
- Jegadeesh（1990），[Evidence of Predictable Behavior of Security Returns](https://doi.org/10.1111/j.1540-6261.1990.tb05110.x)，Journal of Finance 45(3)
- Carhart（1997），[On Persistence in Mutual Fund Performance](https://doi.org/10.1111/j.1540-6261.1997.tb03808.x)，Journal of Finance 52(1)
- Moskowitz & Grinblatt（1999），[Do Industries Explain Momentum?](https://doi.org/10.1111/0022-1082.00146)，Journal of Finance 54(4)
- Asness, Moskowitz & Pedersen（2013），[Value and Momentum Everywhere](https://doi.org/10.1111/jofi.12021)，Journal of Finance 68(3)
- Moskowitz, Ooi & Pedersen（2012），[Time Series Momentum](https://doi.org/10.1016/j.jfineco.2011.11.003)，Journal of Financial Economics 104(2)
- Novy-Marx（2012），[Is Momentum Really Momentum?](https://doi.org/10.1016/j.jfineco.2011.05.003)，Journal of Financial Economics 103(3)
- George & Hwang（2004），[The 52-Week High and Momentum Investing](https://doi.org/10.1111/j.1540-6261.2004.00695.x)，Journal of Finance 59(5)
- Israel & Moskowitz（2013），[The Role of Shorting, Firm Size, and Time on Market Anomalies](https://doi.org/10.1016/j.jfineco.2012.11.005)，Journal of Financial Economics 108(2)
- Hong, Lim & Stein（2000），[Bad News Travels Slowly: Size, Analyst Coverage, and the Profitability of Momentum Strategies](https://doi.org/10.1111/0022-1082.00206)，Journal of Finance 55(1)
- Heston & Sadka（2008），[Seasonality in the Cross-Section of Stock Returns](https://doi.org/10.1016/j.jfineco.2007.02.003)，Journal of Financial Economics 87(2)
- Grinblatt & Moskowitz（2004），[Predicting Stock Price Movements from Past Returns: The Role of Consistency and Tax-Loss Selling](https://doi.org/10.1016/S0304-405X(03)00176-4)，Journal of Financial Economics 71(3)
- Hoffstein, Faber & Braun（2020），[Rebalance Timing Luck: The (Dumb) Luck of Smart Beta](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3673910)，SSRN 工作論文（Newfound Research）（非同儕審查）
- Shumway（1997），[The Delisting Bias in CRSP Data](https://doi.org/10.1111/j.1540-6261.1997.tb03818.x)，Journal of Finance 52(1)
- Lesmond, Schill & Zhou（2004），[The Illusory Nature of Momentum Profits](https://doi.org/10.1016/S0304-405X(03)00206-X)，Journal of Financial Economics 71(2)
- Korajczyk & Sadka（2004），[Are Momentum Profits Robust to Trading Costs?](https://doi.org/10.1111/j.1540-6261.2004.00656.x)，Journal of Finance 59(3)
- Novy-Marx & Velikov（2016），[A Taxonomy of Anomalies and Their Trading Costs](https://doi.org/10.1093/rfs/hhv063)，Review of Financial Studies 29(1)
- Frazzini, Israel & Moskowitz（2012），[Trading Costs of Asset Pricing Anomalies](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2294498)，SSRN 工作論文（AQR）（非同儕審查）
- Daniel & Moskowitz（2016），[Momentum Crashes](https://doi.org/10.1016/j.jfineco.2015.12.002)，Journal of Financial Economics 122(2)
- Barroso & Santa-Clara（2015），[Momentum Has Its Moments](https://doi.org/10.1016/j.jfineco.2014.11.010)，Journal of Financial Economics 116(1)
- Cooper, Gutierrez & Hameed（2004），[Market States and Momentum](https://doi.org/10.1111/j.1540-6261.2004.00665.x)，Journal of Finance 59(3)
- Geczy & Samonov（2016），[Two Centuries of Price-Return Momentum](https://doi.org/10.2469/faj.v72.n5.1)，Financial Analysts Journal 72(5)
- Han, Zhou & Zhu（2014），[Taming Momentum Crashes: A Simple Stop-Loss Strategy](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2407199)，SSRN 工作論文（非同儕審查）
- McLean & Pontiff（2016），[Does Academic Research Destroy Stock Return Predictability?](https://doi.org/10.1111/jofi.12365)，Journal of Finance 71(1)
- Harvey, Liu & Zhu（2016），[… and the Cross-Section of Expected Returns](https://doi.org/10.1093/rfs/hhv059)，Review of Financial Studies 29(1)
- White（2000），[A Reality Check for Data Snooping](https://doi.org/10.1111/1468-0262.00152)，Econometrica 68(5)
- Bailey & López de Prado（2014），[The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality](https://doi.org/10.3905/jpm.2014.40.5.094)，Journal of Portfolio Management 40(5)
- Bailey, Borwein, López de Prado & Zhu（2016），[The Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322)，Journal of Computational Finance
