# Multi-Strategy Quantitative Screener

A quant-first stock screening system that scans NYSE/NASDAQ daily to surface 1-3 short-term trade candidates using two complementary signal models: **Mean Reversion** (defensive compounder, all regimes) and **Sniper** (concentrated high-velocity, bull/choppy only).

The system runs autonomously on a daily schedule: morning pipeline at 6 AM ET, afternoon position checks at 4:30 PM ET, and weekly meta-reviews on Sundays.

## Architecture

```
L0  Scheduler          APScheduler (cron), triggers daily at 6:00 AM ET
L1  Data Ingestion     Async Python — Polygon + FMP + yfinance + FRED (semaphore-controlled)
L2  Feature Engine     pandas, pandas-ta, numpy — 30+ technical indicators
L3  Signal Models      Mean Reversion + Sniper — regime-gated, score-ranked
L4  Validation         8-check NoSilentPass gate + Deflated Sharpe Ratio
L5  Output             PostgreSQL, Telegram alerts (with model scorecard), FastAPI dashboard
L6  Engine Observer    Collect external engine results → deterministic synthesis → Capital Guardian
```

Layers 1-5 produce picks with zero LLM cost. Layer 6 collects signals from external engines as observer/telemetry — currently inactive (all engine apps scaled to zero). The only LLM usage is the optional weekly meta-analyst review (Sundays).

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

BB squeeze + volume compression setups on high-ATR stocks. Concentrated high-velocity.

- **Entry**: BB squeeze + volume compression→expansion + RS vs SPY + trend alignment + momentum base
- **Hard gates**: ATR% < 5.0 (must be volatile enough), avg volume < 500K, bear regime block, score < 70
- **Stop**: 1.5x ATR below entry
- **Target**: 3.0x ATR above entry
- **Hold**: 7 days max, 1-day time stop (exit if flat/negative after day 1)
- **Trailing stop**: Activates at +1.0%, distance 0.5%
- **Max positions**: 3 (16.7% each)
- **Backtest baseline** (1Y): 84 trades, 85.7% WR, Sharpe 10.5, PF 5.19, sleeve DD 3.8%

| Component | Weight | Logic |
|-----------|--------|-------|
| BB squeeze | 30% | BB width percentile over 60 bars |
| Volume compression→expansion | 25% | 5-day vol declining + today > 1.5x avg |
| RS vs SPY | 20% | Relative strength (10-day ROC vs SPY) |
| Trend alignment | 15% | Price > SMA50 > SMA200 |
| Momentum base | 10% | RSI(14) 40-65 (coiled, not overbought) |

### Ticker Blacklist

24 tickers with win rate < 35% over 50+ trades are permanently excluded. See `mean_reversion_blacklist` in config.

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
- **Model Scorecard (30d)**: Per-model breakdown of trades, WR, avg P&L, PF, and open positions
- The scorecard appears in all alert paths (normal picks, no picks, validation failure)

## Engine Observer Layer (Currently Inactive)

Three external engines previously fed into a cross-engine synthesis layer. After 90 days of zero convergence and no actionable alpha, all engine Heroku apps were scaled to zero (2026-03-09). The pipeline handles engine unavailability gracefully — Steps 10-14 are non-fatal.

| Engine | Heroku App | Status |
|---|---|---|
| KooCore-D | `koocore-dashboard` | Scaled to zero |
| Gemini STST | `geministst` | Scaled to zero |
| Top3-7D | `top3-7d-engine` | Scaled to zero |

Steps 1-9 (signal generation, validation, DB, Telegram) are fully independent of engine availability.

## Project Structure

```
src/
├── config.py                   # Environment + typed settings
├── main.py                     # Pipeline orchestration, scheduling, fail-closed wrapper
├── contracts.py                # StageEnvelope + typed payload models + engine contracts
├── worker.py                   # Heroku worker process
├── data/                       # Data ingestion
│   ├── aggregator.py           # Unified interface with fallback chains
│   ├── polygon_client.py       # OHLCV, news
│   ├── fmp_client.py           # Fundamentals, earnings, insider transactions
│   ├── yfinance_client.py      # Fallback price data
│   └── fred_client.py          # VIX, yield curve, macro indicators
├── features/                   # Feature engineering
│   ├── technical.py            # 30+ indicators (RSI, ATR, VWAP, RVOL, MAs, BB width pctile, vol slope)
│   ├── fundamental.py          # Earnings surprise, insider activity scoring
│   ├── sentiment.py            # News headline sentiment scoring
│   └── regime.py               # Market regime classifier + breadth + allowed model gate
├── signals/                    # Signal generation
│   ├── filter.py               # Universe filtering + funnel counters
│   ├── mean_reversion.py       # RSI(2) mean reversion model (active)
│   ├── sniper.py               # BB squeeze + vol compression model (active)
│   ├── breakout.py             # Momentum breakout model (disabled — zero edge)
│   ├── catalyst.py             # Event-driven catalyst model (disabled)
│   └── ranker.py               # Ranking + correlation + confluence + cooldown
├── engines/                    # Cross-engine system (observer layer)
│   ├── collector.py            # Async HTTP engine fetcher (aiohttp, fail-open)
│   ├── credibility.py          # Dynamic weight tracker (hit rate, Brier score, convergence)
│   ├── outcome_resolver.py     # Resolve past picks, update credibility
│   ├── deterministic_synthesizer.py  # Deterministic cross-engine synthesis
│   └── agreement_analysis.py   # Engine convergence/agreement analysis
├── portfolio/                  # Position sizing + risk management
│   └── capital_guardian.py     # Position sizing, drawdown defense, regime scaling
├── backtest/                   # Validation
│   ├── walk_forward.py         # Walk-forward backtesting engine
│   ├── metrics.py              # Sharpe, Sortino, Calmar, DSR, profit factor
│   └── validation_card.py      # 8-check NoSilentPass validation gate
├── research/                   # Research tools
│   ├── signal_backtest.py      # Standalone signal model backtester
│   └── drift_check.py          # Model drift detection (runs on afternoon schedule)
├── output/                     # Output
│   ├── telegram.py             # Telegram alerts (daily picks, scorecard, cross-engine)
│   ├── report.py               # Jinja2 HTML report generation
│   ├── health.py               # Position health card engine
│   └── performance.py          # Outcome tracking, calibration, risk metrics
└── db/
    ├── models.py               # SQLAlchemy models
    └── session.py              # Async PostgreSQL connection

scripts/
├── run_v12_backtest.py         # Standardized MR backtester
├── run_sniper_backtest.py      # Sniper signal backtester
├── run_phase2_backtest.py      # A/B experiment matrix runner
├── simulate_sniper_sleeve.py   # Sleeve portfolio simulator (Max 2 vs Max 3)
├── evaluate_sniper_campaign.py # Campaign evaluator with hybrid promotion gates
└── analyze_weekday.py          # Weekday effect analysis

api/
└── app.py                      # FastAPI dashboard — reports, signals, outcomes, cross-engine

tests/                          # 913+ tests across all modules
```

## Daily Orchestration

```
  6:00 AM ET   Morning pipeline:
    Steps 1-4    Macro regime → Universe → OHLCV → Features
    Step 5       Signal generation (MR + Sniper, regime-gated)
    Steps 6-7    Ranking + Validation gate
    Steps 8-9    DB save + Telegram alert (with model scorecard)
    Steps 10-14  Engine observer (non-fatal, currently all engines offline)

  4:30 PM ET   Afternoon check:
    - Position health assessment
    - Outcome resolution (MR + Sniper)
    - Model drift detection (alerts if drift detected)

  Sunday 7 PM  Weekly meta-review (only LLM usage — Claude Opus 4.6)
```

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL database
- API keys: Polygon, FMP, FRED, Finnhub (required); Anthropic (optional — weekly meta-review only)

### Installation

```bash
git clone https://github.com/raysyhuang/multi-agentic-screener.git
cd multi-agentic-screener

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```
# API Keys (required)
POLYGON_API_KEY            # Market data — OHLCV, news
FMP_API_KEY                # Fundamentals, earnings, insider transactions
FRED_API_KEY               # Macro indicators (VIX, yield curve)
FINNHUB_API_KEY            # Earnings calendar

# API Keys (optional)
ANTHROPIC_API_KEY          # Weekly meta-analyst review only

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

### Running

```bash
# Run the full pipeline once
python -m src.main --run-now

# Run afternoon position check
python -m src.main --check-now

# Run weekly meta-review
python -m src.main --meta-now

# Start scheduler + API server (production)
python -m src.main

# Run tests
pytest tests/ -v
```

### Heroku Deployment

The app runs on Heroku with two Standard-2X dynos:

```
web:    gunicorn api.app:app --worker-class uvicorn.workers.UvicornWorker
worker: python -m src.worker  (APScheduler with SIGTERM handling)
```

```bash
git push heroku main
# Production test run:
heroku run --size standard-2x "python -m src.main --run-now" --app multi-agentic-screener
```

## Cost

Near-zero daily LLM cost. The pipeline is fully deterministic (quant_only mode). The only LLM usage is the optional weekly Sunday meta-analyst review (~$0.15/run via Claude Opus 4.6).

Data API costs: Polygon + FMP + FRED + Finnhub (see provider plans).

## Experiment Log

All experiments are tracked in `configs/experiment_log.yaml` with hypothesis, config, results, and verdict. Key results:

| Experiment | Verdict | Key Result |
|---|---|---|
| Two-leg partial TP | Fail | Dilutes winners, trailing stop already captures MFE |
| Entry refinement filters | **Pass** | Sharpe +6.5%, Sortino +27%, MaxDD -20% |
| Confirmation proxy (T+1 close) | Shadow | +8pp WR but lower expectancy |
| Score-tiered stops v3 | **Pass** | Sharpe +6.7%, PF +9.4%, no trade count impact |
| Sniper BB squeeze model | **Pass** | 85.7% WR, Sharpe 10.5, PF 5.19 (backtest) |
| Adaptive MFE exit | Fail | WR=50%, MaxDD=97% without trailing stop |

## License

Private project. All rights reserved.
