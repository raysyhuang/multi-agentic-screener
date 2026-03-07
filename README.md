# Multi-Agentic Short-Term Stock Screener

A quant-first stock screening system that scans NYSE/NASDAQ daily to surface 1-2 short-term trade candidates (3-day holds, mean reversion). Three independent engines feed into a deterministic cross-engine synthesizer that weights picks by proven track record.

The system runs autonomously on a daily schedule: morning pipeline at 6 AM ET, afternoon position checks at 4:30 PM ET, evening engine collection at 9:30 PM ET, and weekly meta-reviews on Sundays.

## Architecture

```
L0  Scheduler          APScheduler (cron), triggers daily at 6:00 AM ET
L1  Data Ingestion     Async Python — Polygon + FMP + yfinance + FRED (semaphore-controlled)
L2  Feature Engine     pandas, pandas-ta, numpy — 30+ technical indicators
L3  Signal & Filter    Mean reversion model + regime gate + confluence + cooldown + correlation
L4  Validation         8-check NoSilentPass gate + Deflated Sharpe Ratio
L5  Output             PostgreSQL, Telegram alerts, HTML reports, FastAPI dashboard
L6  Cross-Engine       Collect → Deterministic Verify → Deterministic Synthesize → Capital Guardian
```

Layers 1-4 produce picks with zero LLM cost. Layer 6 aggregates signals from 3 external engines using deterministic logic — no LLM calls. The only LLM usage is the optional weekly meta-analyst review (Sundays).

## Cross-Engine Synthesis (Layer 6)

The system acts as the **hub** in a multi-engine architecture, collecting and synthesizing results from independent engines alongside its own pipeline output. Each engine approaches the market differently — convergence detection and dynamic credibility weighting produce higher-conviction picks.

```
                       +------------------------------------------+
                       |     Multi-Agentic Screener (THE HUB)     |
                       |                                          |
  +-----------+  REST  |  +----------+  +-------------------+     |
  | KooCore-D |------->|  | Engine   |  | Credibility       |     |
  +-----------+        |  | Collector|->| Tracker (DB)      |     |
  +-----------+  REST  |  |          |  | - hit rates       |     |
  |Gemini     |------->|  +----------+  | - dynamic weights |     |
  |STST       |        |       |        +--------+----------+     |
  +-----------+        |       v                 v                |
  +-----------+  REST  |  +------------------------+              |
  | Top3-7D   |------->|  | Deterministic Verifier |              |
  +-----------+        |  | - Regime weight adjust |              |
                       |  | - Strategy floor       |              |
                       |  | - Confidence recalib   |              |
                       |  +----------+-------------+              |
                       |             v                            |
                       |  +------------------------+              |
                       |  | Deterministic Synth    |              |
                       |  | - Convergent first     |              |
                       |  | - Score-ranked unique   |              |
                       |  | - Max 5 positions      |              |
                       |  +----------+-------------+              |
                       |             v                            |
                       |  +------------------------+              |
                       |  | Capital Guardian       |              |
                       |  | - Position sizing      |              |
                       |  | - Drawdown defense     |              |
                       |  +------------------------+              |
                       |             v                            |
                       |  DB + Telegram + Dashboard               |
                       +------------------------------------------+
```

### Current Integrated Engines

| Engine | Strategy | Holds | Collection |
|---|---|---|---|
| **Multi-Agentic Screener** | RSI(2) Mean Reversion, trailing stop | 3d | Local pipeline |
| **KooCore-D** | Weekly/Pro30/Swing momentum, regime-gated | 7-30d | HTTP from Heroku (`koocore-dashboard`) |
| **Gemini STST** | Momentum (RVOL>2, ATR>6.5%) + Reversion (RSI2<10) | 5-7d | Local (in-process runner) |
| **Top3-7D** | Energy/Release/Amplification 3-gate, top 3 picks | 7d | HTTP from Heroku (`sleepy-everglades-94250`) |

Collection mode is **hybrid** (`engine_run_mode=hybrid`): KooCore-D and Top3-7D via HTTP, Gemini STST runs locally in-process. All engines normalize to the `EngineResultPayload` contract.

### Pipeline Steps (10-14)

After the core pipeline (Steps 1-9) completes, five cross-engine steps run:

1. **Step 10 — Collect**: Fetches engine results in parallel (hybrid mode: KooCore-D + Top3-7D via HTTP, Gemini STST locally). Fail-open per engine (one engine down doesn't block others). Results stored in `external_engine_results` table with payload hashing and revision tracking.

2. **Step 11 — Resolve Outcomes**: Resolves previous-day engine pick outcomes against actual market prices via yfinance. Updates `engine_pick_outcomes` table and recomputes credibility weights.

3. **Step 12 — Deterministic Verify**: Applies regime-based weight adjustments (bear boosts mean-reversion, penalizes momentum; bull does the reverse), strategy-level hit-rate floors, and confidence recalibration. Zero LLM cost.

4. **Step 13 — Deterministic Synthesize**: Builds final portfolio: convergent picks (2+ engines agree) prioritized over unique picks, sorted by combined score, max 5 positions with equal 10% weights. Capital Guardian adjusts sizing downstream. Zero LLM cost.

5. **Step 14 — Save + Alert**: Results saved to `cross_engine_synthesis` table and sent via Telegram.

### Dynamic Credibility Weighting

Engines **earn their weight** through proven track record — not equal weighting:

```
engine_weight = base_weight * hit_rate_multiplier * calibration_bonus
```

- `hit_rate_multiplier` = engine hit rate / average hit rate across all engines
- `calibration_bonus` = 1.2x if the engine's Brier score < 0.15 (well-calibrated confidence)
- Weights clamped to [0.3, 3.0] range
- Requires 10+ resolved picks before differentiation begins
- Strategy-level floor: strategies below 15% hit rate (with 5+ picks) are filtered out
- Confidence recalibration: scales engine confidence by historical accuracy ratio

### Convergence Multipliers

When multiple engines independently identify the same ticker, conviction increases:

| Engines Agreeing | Multiplier |
|---|---|
| 1 engine | 0.9x (penalty) |
| 2 engines | 1.3x |
| 3+ engines | 1.0x (base) |
| Same sector (2+ engines) | 1.15x |

## Signal Model

**RSI(2) Mean Reversion** — Deeply oversold stocks with intact long-term uptrends.

- **Entry**: RSI(2) <= 10, above 200d SMA
- **Stop**: 0.75x ATR below entry
- **Target**: 1.5x ATR above entry
- **Hold**: 3 days max
- **Trailing stop**: Activates at +0.5%, distance 0.3%
- **Works in all regimes**: bull Sharpe 0.46, bear 0.40, choppy 0.15

Five scoring factors: RSI(2) oversold level, long-term trend intact, consecutive down days, distance from recent low, volume liquidity.

Previously tested models (breakout, catalyst) were disabled after backtesting showed zero edge.

### Ticker Blacklist

24 tickers with win rate < 35% over 50+ trades are permanently excluded. See `mean_reversion_blacklist` in config.

## Regime Detection

All signals are gated by market regime.

Regime is classified using five inputs:
- SPY/QQQ trend (above/below 20d SMA)
- SPY price slope (20-day normalized)
- VIX level (complacent / neutral / fear)
- Yield curve (normal / flat / inverted)
- Market breadth (% of universe above 20d SMA)

| Regime | Allowed Models | Max Position Size |
|---|---|---|
| BULL | Mean Reversion | 10% |
| BEAR | Mean Reversion | 5% |
| CHOPPY | Mean Reversion | 5% |

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
│   ├── technical.py            # 30+ indicators (RSI, ATR, VWAP, RVOL, MAs, etc.)
│   ├── fundamental.py          # Earnings surprise, insider activity scoring
│   ├── sentiment.py            # News headline sentiment scoring
│   └── regime.py               # Market regime classifier + breadth
├── signals/                    # Signal generation
│   ├── filter.py               # Universe filtering + funnel counters
│   ├── mean_reversion.py       # RSI(2) mean reversion model (active)
│   ├── breakout.py             # Momentum breakout model (disabled)
│   ├── catalyst.py             # Event-driven catalyst model (disabled)
│   └── ranker.py               # Ranking + correlation + confluence + cooldown
├── engines/                    # Cross-engine system
│   ├── collector.py            # Async HTTP engine fetcher (aiohttp, fail-open)
│   ├── credibility.py          # Dynamic weight tracker (hit rate, Brier score, convergence)
│   ├── outcome_resolver.py     # Resolve past picks, update credibility, generate feedback
│   ├── deterministic_synthesizer.py  # Deterministic cross-engine synthesis (replaces LLM)
│   └── agreement_analysis.py   # Engine convergence/agreement analysis
├── agents/                     # LLM agents (mostly inactive in quant_only mode)
│   ├── meta_analyst.py         # Weekly self-review (only active LLM agent)
│   ├── cross_engine_synthesizer.py # Data models (SynthesizerOutput, PortfolioPosition, etc.)
│   └── ...                     # Other agents preserved but unused in quant_only mode
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
│   ├── telegram.py             # Telegram alerts (daily picks, health, cross-engine)
│   ├── report.py               # Jinja2 HTML report generation
│   ├── health.py               # Position health card engine
│   └── performance.py          # Outcome tracking, calibration, risk metrics
└── db/
    ├── models.py               # SQLAlchemy models
    └── session.py              # Async PostgreSQL connection

api/
└── app.py                      # FastAPI dashboard — reports, signals, outcomes, cross-engine

tests/                          # 877+ tests across all modules
```

## Daily Orchestration

```
  6:00 AM ET   Morning pipeline:
    Steps 1-9    Data -> features -> signals -> validation -> DB + Telegram
    Step 10      Collect engine results via HTTP (fail-open per engine)
    Step 11      Resolve previous-day engine pick outcomes -> update credibility
    Step 12      Deterministic regime weight adjustment
    Step 13      Deterministic cross-engine synthesis -> portfolio
    Step 14      Capital Guardian sizing -> DB + Telegram

  4:30 PM ET   Afternoon check:
    - Position health assessment
    - Outcome resolution
    - Model drift detection (alerts if drift detected)

  9:30 PM ET   Evening engine collection:
    - Re-collect engine results (catches engines that run after market close)
    - Process any new picks through Steps 11-14

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

# Cross-Engine System
KOOCORE_API_URL            # KooCore-D Heroku URL
GEMINI_API_URL             # Gemini STST Heroku URL
TOP3_API_URL               # Top3-7D Heroku URL
ENGINE_API_KEY             # Shared API key for engine-to-engine auth
CROSS_ENGINE_ENABLED       # true (default) or false
```

Note: The API runtime requires PostgreSQL. The ORM models use `JSONB`, so running
`api.app` with a SQLite `DATABASE_URL` will fail at startup.

### Running

```bash
# Run the full pipeline once (includes cross-engine synthesis)
python -m src.main --run-now

# Run with cross-engine debug mode
python -m src.main --run-now --debug-engines

# Run afternoon position check
python -m src.main --check-now

# Run weekly meta-review
python -m src.main --meta-now

# Run agreement analysis report
python -m src.main --agreement-report

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

## License

Private project. All rights reserved.
