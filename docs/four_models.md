
https://github.com/raysyhuang/multi-agentic-screener
'/Users/rayhuang/Documents/Python Project/Multi-Agentic Screener'

https://github.com/raysyhuang/KooCore-D
'/Users/rayhuang/Documents/Python Project/KooCore-D'

https://github.com/raysyhuang/gemini_STST
'/Users/rayhuang/Documents/Python Project/Gemini STST'

https://github.com/raysyhuang/Top3-7D-engine
'/Users/rayhuang/Documents/Python Project/Top3-7D Engine'

---

# Signal Models & Feature Engine

## Feature Engine (`src/features/technical.py`)

### Moving Averages & Trend
| Feature | Formula | Purpose |
|---------|---------|---------|
| `sma_10` | SMA(close, 10) | Short-term trend |
| `sma_20` | SMA(close, 20) | Swing trend baseline |
| `sma_50` | SMA(close, 50) | Intermediate trend |
| `sma_200` | SMA(close, 200) | Long-term trend reference |
| `ema_9` | EMA(close, 9) | Fast momentum |
| `ema_21` | EMA(close, 21) | Slow momentum |
| `pct_above_sma20` | (close - sma_20) / sma_20 * 100 | Distance from 20 SMA |
| `pct_above_sma50` | (close - sma_50) / sma_50 * 100 | Distance from 50 SMA |
| `pct_above_sma200` | (close - sma_200) / sma_200 * 100 | Distance from 200 SMA |
| `sma_20_slope` | (sma_20 - sma_20[10 bars ago]) / sma_20[10 bars ago] * 100 | Trend direction (>0 rising, <0 falling) |

### Momentum
| Feature | Formula | Purpose |
|---------|---------|---------|
| `rsi_14` | RSI(close, 14) | Standard overbought/oversold |
| `rsi_2` | RSI(close, 2) | Mean-reversion trigger |
| `MACD*` | MACD(12, 26, 9) | Trend momentum |
| `roc_5` | ROC(close, 5) | 5-day rate of change |
| `roc_10` | ROC(close, 10) | 10-day rate of change |

### Volatility
| Feature | Formula | Purpose |
|---------|---------|---------|
| `atr_14` | ATR(14) | Position sizing |
| `atr_pct` | atr_14 / close * 100 | Normalized volatility |
| `BBands` | Bollinger(20, 2) | Volatility envelope |

### Volume
| Feature | Formula | Purpose |
|---------|---------|---------|
| `vol_sma_20` | SMA(volume, 20) | Average volume baseline |
| `rvol` | volume / vol_sma_20 | Relative volume |
| `volume_surge` | volume > vol_sma_20 * 2.0 | Binary surge flag |
| `obv` | OBV(close, volume) | Cumulative volume flow |

### Breakout & Structure
| Feature | Formula | Purpose |
|---------|---------|---------|
| `high_20d` | 20-day rolling high | Resistance level |
| `low_20d` | 20-day rolling low | Support level |
| `near_20d_high` | close >= high_20d * 0.98 | Near breakout flag |
| `near_20d_low` | close <= low_20d * 1.02 | Near breakdown flag |
| `range_10d` | 10-day high - 10-day low | Consolidation width |
| `is_consolidating` | range_10d < atr_14 * 1.5 | Tight range flag |

### Gap Detection
| Feature | Formula | Purpose |
|---------|---------|---------|
| `gap_pct` | (open - prev_close) / prev_close * 100 | Gap size (%) |
| `is_gap_up` | gap_pct >= 3.0 | Binary gap-up flag |
| `is_gap_down` | gap_pct <= -3.0 | Binary gap-down flag |

---

## Signal Model 1: Momentum Breakout (`src/signals/breakout.py`)

Ported from **KooCore-D**. Identifies stocks breaking out of consolidation on high volume.

**Weights:**
| Factor | Weight | Key Inputs |
|--------|--------|------------|
| Technical Momentum | 30% | RSI(14) 55-75 sweet spot, ROC(5), ROC(10) |
| Volume Confirmation | 25% | RVOL >= 1.5, volume surge, gap+volume bonus |
| Consolidation Breakout | 20% | Near 20d high + was consolidating, gap-up bonus |
| Trend Alignment | 15% | Above SMA(20) + SMA(50) + SMA(20) slope > 0.5 |
| Volatility Context | 10% | ATR% 2-5% sweet spot |

**Swing trading enhancements:**
- Gap-up breakout bonus (+25 in consolidation bucket) — gaps above 3% with volume confirmation are high-conviction breakout entries (Humbled Trader methodology)
- Gap + high volume bonus (+15 in volume bucket) — gap breakouts with RVOL >= 2.0 get additional conviction
- SMA(20) slope confirmation (+30 in trend bucket) — rising 20 SMA (slope > 0.5%) confirms momentum direction rather than just price level (Emmanuel methodology)

**Execution:** Fire on day T close, enter at T+1 open. Stop = 2x ATR below entry. Targets = 2:1 and 3:1 R:R. Hold up to 10 days.

---

## Signal Model 2: RSI(2) Mean Reversion (`src/signals/mean_reversion.py`)

Ported from **gemini_STST**. Catches oversold bounces in stocks with intact uptrends.

**Weights:**
| Factor | Weight | Key Inputs |
|--------|--------|------------|
| RSI(2) Oversold | 40% | RSI(2) <= 10 (extreme), <= 20 (moderate) |
| Long-term Trend Intact | 25% | pct_above_sma200 > 0 (pre-computed), fallback to sma50 |
| Consecutive Down Days | 15% | Streak <= -3 preferred |
| Distance from 5d Low | 10% | Close within 1% of 5-day low |
| Volume Liquidity | 10% | RVOL >= 0.5 (still liquid) |

**Swing trading enhancements:**
- Pre-computed SMA(200) — uses `pct_above_sma200` from the feature engine instead of inline rolling computation. Falls back to `pct_above_sma50` for tickers with < 200 bars of history.

**Execution:** Fire on day T close, enter at T+1 open. Stop = 1.5x ATR below entry. Target = reversion to 5-day and 10-day SMA. Hold up to 5 days.

---

## Regime Gate

Market regime (bull/bear/neutral) determines which models are allowed to fire:
- **Bull**: breakout + mean_reversion
- **Neutral**: breakout + mean_reversion (with tighter thresholds)
- **Bear**: mean_reversion only (breakout signals blocked)

---

## Confluence Detection

When a ticker is flagged by multiple signal models (e.g., both breakout and mean_reversion), it receives a +10% confluence bonus. This cross-model agreement increases conviction.
