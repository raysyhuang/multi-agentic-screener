# Multi-Strategy Quantitative Screener

A quant-first stock screening system that scans NYSE/NASDAQ daily to surface short-term trade candidates using complementary deterministic signal models.

The system runs autonomously: morning pipeline at 6:00 AM ET and afternoon position checks at 4:30 PM ET.

> **Fully deterministic — zero LLM.** The LLM agent stack was removed (2026-07-19); the app is lean quant-only. The repo name is historical.

## Architecture

```
L1  Data Ingestion     Async Python — Polygon (daily + minute) + FMP + yfinance + FRED
L2  Feature Engine     pandas, pandas-ta, numpy — 30+ technical indicators
L3  Signal Models      Mean Reversion + Sniper (+ PEAD, paper trial) — regime-gated, score-ranked
L4  Validation         8-check NoSilentPass gate + Deflated Sharpe Ratio
L5  Output             PostgreSQL, Telegram alerts (with model scorecard), FastAPI dashboard

    research/          Alpha probes — IC analysis, signal backtests, drift checks
    governance/        Threshold management, retrain policy, performance monitor, divergence ledger
    portfolio/         Capital Guardian — position sizing, drawdown defense, regime scaling
```

Scheduling runs on GitHub Actions cron (`.github/workflows/scheduled-pipelines.yml`, DST-guarded); alerts from those runs are tagged `[MAS-GH]`. APScheduler remains available for local/worker use.

## Signal Models

### Mean Reversion (Primary — All Regimes)

Deeply oversold stocks with intact long-term uptrends. Defensive compounder.

- **Entry**: RSI(2) <= 10, above 200d SMA
- **Stop**: Score-tiered — score >= 85: 1.25x ATR, 70-84: 0.85x ATR, < 70: 0.50x ATR
- **Target**: 1.5x ATR above entry
- **Hold**: 3 days max
- **Trailing stop**: Activates at +0.5%, distance 0.3%
- **Entry filters**: Gap < 0.2x ATR, volume slope (3-bar), choppy gate (min score 75), ATR floor (10th pctile), earnings blackout (2d)
- **Works in all regimes**: bull Sharpe 0.46, bear 0.40, choppy 0.15
- **Backtest baseline** (1Y): 7,549 trades, 69.5% WR, Sharpe 2.56, PF 2.78, MaxDD 29.2%

Five scoring factors: RSI(2) oversold level, long-term trend intact, consecutive down days, distance from recent low, volume liquidity.

### Sniper (Offensive — Bull/Choppy Only)

BB squeeze + volume compression setups on high-ATR stocks.

- **Entry**: BB squeeze + volume compression→expansion + RS vs SPY + trend alignment + momentum base
- **Hard gates**: ATR% < 5.0, avg volume < 500K, bear regime block, score < 70
- **Stop**: 1.5x ATR below entry · **Target**: 3.0x ATR above entry
- **Hold**: 7 days max, 1-day time stop (exit if flat/negative after day 1)
- **Trailing stop**: Activates at +1.0%, distance 0.5%
- **Max positions**: 3, enforced as a concurrent-position cap

> ⚠️ **Retracted backtest.** The previously reported 84-trade, 85.7% WR / Sharpe 10.5 / PF 5.19 result was a **fill artifact**, not a tradeable edge (see the sniper truth matrix). Do not cite those figures. Sniper remains live but unproven; treat its expectancy as unestablished pending honest re-measurement.

| Component | Weight | Logic |
|-----------|--------|-------|
| BB squeeze | 30% | BB width percentile over 60 bars |
| Volume compression→expansion | 25% | 5-day vol declining + today > 1.5x avg |
| RS vs SPY | 20% | Relative strength (10-day ROC vs SPY) |
| Trend alignment | 15% | Price > SMA50 > SMA200 |
| Momentum base | 10% | RSI(14) 40-65 (coiled, not overbought) |

### PEAD — Post-Earnings Announcement Drift (Paper Trial, default OFF)

`signals/post_earnings_drift.py`. The **first alpha candidate to survive backtesting** in the research program below. Wired into the pipeline as a paper-trial signal and **disabled by default** — it must prove out on paper before promotion.

### Ticker Blacklist

24 tickers with win rate < 35% over 50+ trades are permanently excluded. See `mean_reversion_blacklist` in config.

## Alpha Research Program

Systematic probe → backtest → refute-or-promote workflow. Scripts live in `scripts/`, analysis tooling in `src/research/`, and outputs in `outputs/research/`. Results to date:

| Probe | Script | Verdict |
|---|---|---|
| **PEAD (post-earnings drift)** | `pead_probe.py`, `pead_backtest.py` | ✅ **Survives** → promoted to paper trial |
| Gap continuation | `gap_continuation_research.py`, `gap_intraday_probe.py` | ❌ No tradeable daily edge |
| Intraday mean reversion | `intraday_mr_probe.py` | ❌ Reversion refuted (surfaced a VWAP-momentum lead) |
| VWAP momentum | `intraday_vwap_backtest.py` | ❌ Refuted on backtest |
| Short volume ratio | (untried edge #1) | ❌ No coherent edge |
| Days to cover | `days_to_cover_probe.py` | ❌ No coherent edge |

Supporting infrastructure: Polygon minute-bar fetch with a windowed intraday cache, an intraday same-bar tie resolver in the exit engine, and hardened Polygon retries.

## Regime Detection

All signals are gated by market regime. Sniper is hard-blocked in bear markets.

Regime is classified using five inputs:
- SPY/QQQ trend (above/below 20d SMA)
- SPY price slope (20-day normalized)
- VIX level (complacent / neutral / fear)
- Yield curve (normal / flat / inverted)
- Market breadth (% of universe above 20d SMA)

| Regime | Allowed Models | Max Position Size |
|---|---|---|
| BULL | Mean Reversion, Sniper | 10% |
| BEAR | Mean Reversion only | 5% |
| CHOPPY | Mean Reversion, Sniper | 5% |

## Validation Gate (NoSilentPass)

Every pipeline run passes through 8 validation checks. A single failure blocks all picks:

1. **Timestamp integrity** — signals only use data available at as-of date
2. **Next-bar execution** — execution must be T+1 or later
3. **Future data guard** — reject columns tagged as future-known
4. **Slippage sensitivity** — signal must survive +50% slippage increase
5. **Threshold sensitivity** — score threshold +/-10% must not flip >30% of signals (requires 30+ trades)
6. **Confidence calibration** — win rate must exceed 45% minimum bar (requires 30+ trades)
7. **Regime survival** — positive expectancy in at least 2 of 3 regime types (requires 30+ trades)
8. **Deflated Sharpe Ratio** — penalizes for selection bias across strategy variants

## Telegram Alerts

Daily alerts include:
- Regime status and pick details (entry, stop, target, R:R, holding period)
- Validation gate results (pass/fail with specific check failures)
- **Model Scorecard (30d)**: per-model trades, WR, avg P&L, PF, and open positions (`--days` configurable)
- The scorecard appears in all alert paths (normal picks, no picks, validation failure)

## Removed Subsystems

Two subsystems were deleted on 2026-07-19 and are documented here only so old references make sense:

- **LLM agent stack** — agents, memory, skills, and the Sunday meta-analyst review. The app no longer calls any LLM.
- **Cross-engine observer (former L6)** — collector, credibility tracker, outcome resolver, deterministic synthesizer and agreement analysis. It ingested three external engines (`koocore-dashboard`, `geministst`, `top3-7d-engine`), all of which were scaled to zero on 2026-03-09 after 90 days of zero convergence and no actionable alpha. The code is now gone; those engines are archive-only and have no runtime relationship to this project.

## Project Structure

```
src/
├── config.py                   # Environment + typed settings
├── main.py                     # Pipeline orchestration, scheduling, fail-closed wrapper
├── contracts.py                # StageEnvelope + typed payload models
├── worker.py                   # Worker process
├── data/                       # Data ingestion
│   ├── aggregator.py           # Unified interface with fallback chains
│   ├── polygon_client.py       # OHLCV, news, minute bars (windowed intraday cache)
│   ├── fmp_client.py           # Fundamentals, earnings, insider transactions (/stable endpoints)
│   ├── yfinance_client.py      # Fallback price data
│   └── fred_client.py          # VIX, yield curve, macro indicators
├── features/                   # technical.py, fundamental.py, sentiment.py, regime.py
├── signals/                    # Signal generation
│   ├── filter.py               # Universe filtering + funnel counters
│   ├── mean_reversion.py       # RSI(2) mean reversion model (active)
│   ├── sniper.py               # BB squeeze + vol compression model (active)
│   ├── post_earnings_drift.py  # PEAD (paper trial, default OFF)
│   ├── breakout.py             # Momentum breakout model (disabled — zero edge)
│   ├── catalyst.py             # Event-driven catalyst model (disabled)
│   └── ranker.py               # Ranking + correlation + confluence + cooldown
├── backtest/
│   ├── walk_forward.py         # Walk-forward backtesting engine
│   ├── exit_engine.py          # Exit simulation + intraday same-bar tie resolver
│   ├── metrics.py              # Sharpe, Sortino, Calmar, DSR, profit factor
│   ├── validation_card.py      # 8-check NoSilentPass validation gate
│   └── runner.py
├── research/                   # Alpha research tooling
│   ├── ic_analysis.py          # Information coefficient analysis
│   ├── ic_report.py
│   ├── signal_backtest.py      # Standalone signal model backtester
│   ├── drift_check.py          # Model drift detection (afternoon schedule)
│   └── sp500_tickers.py
├── governance/                 # Model governance
│   ├── threshold_manager.py    # Threshold lifecycle
│   ├── retrain_policy.py       # Retrain triggers
│   ├── performance_monitor.py
│   ├── divergence_ledger.py
│   └── artifacts.py
├── portfolio/
│   ├── capital_guardian.py     # Position sizing, drawdown defense, regime scaling
│   └── construct.py            # Portfolio construction
├── validation/stage_validator.py
├── output/                     # telegram.py, report.py, health.py, performance.py
└── db/                         # models.py, session.py (async PostgreSQL)

scripts/                        # Backtesters, probes, evaluators, backfills
api/app.py                      # FastAPI dashboard — reports, signals, outcomes
tests/                          # 913+ tests across all modules
```

## Daily Orchestration

```
  6:00 AM ET   Morning pipeline:
    Steps 1-4    Macro regime → Universe → OHLCV → Features
    Step 5       Signal generation (MR + Sniper, regime-gated; PEAD paper trial if enabled)
    Steps 6-7    Ranking + Validation gate
    Steps 8-9    DB save + Telegram alert (with model scorecard)

  4:30 PM ET   Afternoon check:
    - Position health assessment
    - Outcome resolution
    - Model drift detection (alerts if drift detected)
```

## Setup

### Prerequisites

- Python 3.12
- PostgreSQL database
- API keys: Polygon, FMP, FRED, Finnhub

### Installation

```bash
git clone https://github.com/raysyhuang/multi-agentic-screener.git
cd multi-agentic-screener

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```
# API Keys (required)
POLYGON_API_KEY            # Market data — OHLCV, news, minute bars
FMP_API_KEY                # Fundamentals, earnings, insider transactions
FRED_API_KEY               # Macro indicators (VIX, yield curve)
FINNHUB_API_KEY            # Earnings calendar

# Output
TELEGRAM_BOT_TOKEN         # Telegram alerts
TELEGRAM_CHAT_ID           # Target chat for alerts
DATABASE_URL               # PostgreSQL connection string

# API Security
API_SECRET_KEY             # Bearer token for /api/* routes
ALLOWED_ORIGINS            # Comma-separated CORS origins

# Trading
TRADING_MODE               # PAPER (default) or LIVE
```

> FMP's legacy `/api/v3` endpoints were retired (2025-08-31). This project uses the `/stable` endpoints.

### Running

```bash
# Run the full pipeline once
python -m src.main --run-now

# Run afternoon position check
python -m src.main --check-now

# Start scheduler + API server
python -m src.main

# Run tests
pytest tests/ -v
```

### Heroku Deployment

```
web:    gunicorn api.app:app --worker-class uvicorn.workers.UvicornWorker
worker: python -m src.worker  (APScheduler with SIGTERM handling)
```

```bash
git push heroku main
heroku run --size standard-2x "python -m src.main --run-now" --app multi-agentic-screener
```

## Cost

**Zero LLM cost** — the pipeline is fully deterministic and the LLM stack has been removed.

Data API costs: Polygon + FMP + FRED + Finnhub (see provider plans).

## Experiment Log

All experiments are tracked in `configs/experiment_log.yaml` with hypothesis, config, results, and verdict. Key results:

| Experiment | Verdict | Key Result |
|---|---|---|
| Two-leg partial TP | Fail | Dilutes winners, trailing stop already captures MFE |
| Entry refinement filters | **Pass** | Sharpe +6.5%, Sortino +27%, MaxDD -20% |
| Confirmation proxy (T+1 close) | Shadow | +8pp WR but lower expectancy |
| Score-tiered stops v3 | **Pass** | Sharpe +6.7%, PF +9.4%, no trade count impact |
| Sniper BB squeeze model | **Retracted** | Headline 85.7% WR was a fill artifact — not a real edge |
| Adaptive MFE exit | Fail | WR=50%, MaxDD=97% without trailing stop |
| PEAD | **Pass** | Survived backtest → paper trial (default OFF) |
| Gap continuation / intraday MR / VWAP momentum / short-volume / days-to-cover | Fail | No coherent tradeable edge |

## License

Private project. All rights reserved.
