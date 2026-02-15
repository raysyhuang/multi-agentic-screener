# Multi-Agentic Short-Term Stock Screener

A fully automated stock screening system that scans NYSE/NASDAQ daily to surface 1-2 high-conviction short-term trade candidates (5-15 day holds, targeting 5-10% per trade). It combines a deterministic quantitative core with an LLM-powered reasoning layer for signal validation, adversarial debate, and risk management.

The system runs autonomously on a daily schedule: morning pipeline at 6 AM ET, afternoon position checks at 4:30 PM ET, and weekly meta-reviews on Sundays.

## Architecture

```
L0  Scheduler        APScheduler (cron), triggers daily at 6:00 AM ET
L1  Data Ingestion   Async Python — Polygon + FMP + yfinance + FRED (semaphore-controlled, batch+GC)
L2  Feature Engine   pandas, pandas-ta, numpy — 30+ technical indicators
L3  Signal & Filter  Rule-based models + regime gate + confluence + cooldown + correlation filter
L4  LLM Agents       Signal Interpreter → Adversarial Debate → Risk Gate
L5  Validation       7-check NoSilentPass gate + Deflated Sharpe Ratio
L6  Output           PostgreSQL, Telegram alerts, HTML reports, FastAPI
L7  Governance       Audit trail, decay detection, retrain policy, portfolio construction
```

**Key principle**: Layers 1-3 and 5 produce useful candidates with zero LLM cost. Layer 4 adds reasoning quality. Layer 6 adds self-improvement via weekly meta-review. Layer 7 provides model lifecycle management and position sizing.

## LLM Agent Pipeline

Four agents with intentional model diversity for cross-validation:

| Agent | Model | Role |
|---|---|---|
| Signal Interpreter | Claude Sonnet 4.5 | Reads structured features, produces thesis + confidence 0-100 + risk flags |
| Adversarial Validator | GPT-4.1 | Attacks thesis from a different model's perspective (bull/bear debate) |
| Risk Gatekeeper | Claude Opus 4.6 | Final approve/veto/adjust — highest-stakes decision, fail-safe to VETO |
| Meta-Analyst (weekly) | Claude Opus 4.6 | Reviews 30-day performance, detects biases, suggests threshold adjustments |

The pipeline narrows a universe of ~200 stocks down to 1-2 picks:

```
~3000 NYSE/NASDAQ  →  ~200 filtered  →  ~50 scored  →  10 interpreted
→  5 debated  →  1-2 risk-gated  →  7-check validation  →  Final picks
```

## Signal Models

Three deterministic signal models, each with multi-factor composite scoring:

**Momentum Breakout** — Stocks breaking out of consolidation on high volume. Five factors: technical momentum (RSI, MACD, price vs MAs), volume confirmation (RVOL, surge), consolidation breakout (tight range expansion), trend alignment (above key MAs), and volatility context (ATR). 10-day hold. Regime-gated: blocked in BEAR markets.

**RSI(2) Mean Reversion** — Deeply oversold stocks with intact long-term uptrends. Five factors: RSI(2) oversold level, long-term trend intact (above 200d SMA), consecutive down days, distance from recent low, and volume liquidity. 5-day hold targeting reversion to 5-day SMA. Works in all regimes.

**Catalyst/Event-Driven** — Pre-earnings positioning in stocks with strong beat history. Five factors: earnings timing (5-15 day sweet spot), beat streak, earnings momentum, news sentiment, and insider activity. Hold period adapts to earnings date. Blocked in BEAR markets.

## Regime Detection

All signals are gated by market regime — a momentum breakout in a bear market will destroy returns.

Regime is classified using five inputs:
- SPY/QQQ trend (above/below 20d SMA)
- SPY price slope (20-day normalized)
- VIX level (complacent / neutral / fear)
- Yield curve (normal / flat / inverted)
- **Market breadth** (% of universe above 20d SMA)

| Regime | Allowed Models | Max Position Size |
|---|---|---|
| BULL | Breakout, Mean Reversion, Catalyst | 10% |
| BEAR | Mean Reversion only | 5% |
| CHOPPY | Mean Reversion, Catalyst | 5% |

## Validation Gate (NoSilentPass)

Every pipeline run passes through 7 validation checks. A single failure blocks all picks:

1. **Timestamp integrity** — signals only use data available at as-of date
2. **Next-bar execution** — execution must be T+1 or later
3. **Future data guard** — reject columns tagged as future-known
4. **Slippage sensitivity** — signal must survive +50% slippage increase
5. **Threshold sensitivity** — score threshold +/-10% must not flip >30% of signals
6. **Confidence calibration** — win rate must exceed 45% minimum bar
7. **Regime survival** — positive expectancy in at least 2 of 3 regime types

The system also computes a **Deflated Sharpe Ratio** (Bailey & Lopez de Prado) to penalize for selection bias across strategy variants.

## Data Contracts

Every pipeline stage communicates through typed `StageEnvelope` wrappers with `extra="forbid"` enforcement. Seven stage payloads are defined in `src/contracts.py`:

```
DataIngest → Feature → SignalPrefilter → Regime → AgentReview → Validation → FinalOutput
```

All payloads inherit from `StrictModel` (Pydantic `extra="forbid"`) — unknown fields cause immediate rejection.

## Project Structure

```
src/
├── config.py                   # Environment + typed settings
├── main.py                     # Pipeline orchestration, scheduling, fail-closed wrapper
├── contracts.py                # StageEnvelope + 16 typed payload models
├── worker.py                   # Heroku worker process
├── data/                       # L1 — Data ingestion
│   ├── aggregator.py           # Unified interface with fallback chains
│   ├── polygon_client.py       # OHLCV, news
│   ├── fmp_client.py           # Fundamentals, earnings, insider transactions
│   ├── yfinance_client.py      # Fallback price data
│   └── fred_client.py          # VIX, yield curve, macro indicators
├── features/                   # L2 — Feature engineering
│   ├── technical.py            # 30+ indicators (RSI, ATR, VWAP, RVOL, MAs, etc.)
│   ├── fundamental.py          # Earnings surprise, insider activity scoring
│   ├── sentiment.py            # News headline sentiment scoring
│   └── regime.py               # Market regime classifier + breadth
├── signals/                    # L3 — Signal generation
│   ├── filter.py               # Universe filtering + funnel counters
│   ├── breakout.py             # Momentum breakout model
│   ├── mean_reversion.py       # RSI(2) mean reversion model
│   ├── catalyst.py             # Event-driven catalyst model
│   └── ranker.py               # Ranking + correlation + confluence + cooldown
├── agents/                     # L4 — LLM agents
│   ├── base.py                 # Base agent + Pydantic output schemas
│   ├── llm_router.py           # Routes to Claude/GPT, tracks cost
│   ├── signal_interpreter.py   # Claude Sonnet — thesis generation
│   ├── adversarial.py          # GPT — bull/bear debate (2 rounds)
│   ├── risk_gate.py            # Claude Opus — approve/veto/adjust
│   ├── meta_analyst.py         # Claude Opus — weekly self-review
│   └── orchestrator.py         # Runs the 3-agent flow sequentially
├── governance/                 # Model lifecycle management
│   ├── artifacts.py            # Audit trail, GovernanceContext per run
│   ├── performance_monitor.py  # Rolling metrics, decay detection
│   └── retrain_policy.py       # When to retrain, model versioning
├── portfolio/                  # Position sizing + risk management
│   └── construct.py            # Kelly, volatility-scaled, equal-weight sizing
├── backtest/                   # L5 — Validation
│   ├── walk_forward.py         # Walk-forward backtesting engine
│   ├── metrics.py              # Sharpe, Sortino, Calmar, DSR, profit factor
│   └── validation_card.py      # Fragility report + 7-check validation gate
├── output/                     # L6 — Output
│   ├── telegram.py             # Telegram alerts with validation details
│   ├── report.py               # Jinja2 HTML report generation
│   └── performance.py          # Outcome tracking, calibration, risk metrics
└── db/
    ├── models.py               # 6 SQLAlchemy models (DailyRun, Signal, Outcome, etc.)
    └── session.py              # Async PostgreSQL connection

api/
└── app.py                      # FastAPI — reports, signals, outcomes, costs, artifacts

tests/                          # 179 tests across all modules
```

## Safety Mechanisms

- **Fail-closed pipeline** — any unhandled exception guarantees a NoTrade DB record and Telegram alert, never a silent abort
- **30-day paper trading gate** — LIVE mode is blocked until 30+ days of paper trading with positive profit factor
- **Correlation filter** — drops picks with >0.75 return correlation to prevent concentrated risk
- **Signal cooldown** — suppresses re-triggering same ticker for 5 calendar days after a signal fires
- **Confluence detection** — multi-model agreement boosts confidence (2+ models = 10%+ score bonus)
- **Decay detection** — rolling metrics monitor for hit rate collapse, MAE expansion, or negative expectancy
- **Governance audit trail** — every run captures regime, model versions, decay status, config hash, and git commit
- **Portfolio construction** — Kelly/volatility-scaled sizing with regime exposure multipliers and liquidity caps
- **Risk gate fail-safe** — LLM errors default to VETO, not APPROVE
- **No look-ahead** — signals fire on day T close, execute at T+1 open
- **Realistic costs** — 10 bps slippage + commissions in all validation
- **Memory-safe batching** — semaphore-controlled concurrency (20 max) + batch+GC for Heroku 512 MB
- **Funnel counters** — every filter stage tracks elimination counts for debugging

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Index page listing all daily reports |
| `GET /report/{date}` | Daily HTML report with signal cards |
| `GET /performance` | 30-day performance summary |
| `GET /api/signals/{date}` | Signals JSON for a given date |
| `GET /api/runs` | Pipeline run history |
| `GET /api/outcomes` | Outcomes with filters (date, regime, model, confidence) |
| `GET /api/outcomes/{ticker}` | All outcomes for a specific ticker |
| `GET /api/costs` | Daily and per-agent cost breakdown |
| `GET /api/artifacts/{run_id}` | Pipeline stage artifacts for full traceability |
| `GET /api/meta-reviews` | Recent weekly meta-analyst reviews |
| `GET /health` | Health check |

## Setup

### Prerequisites

- Python 3.12+
- PostgreSQL database
- API keys: Anthropic, OpenAI, Polygon, FMP

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
ANTHROPIC_API_KEY          # Claude API (signal interpreter, risk gate, meta-analyst)
OPENAI_API_KEY             # GPT API (adversarial validator)
POLYGON_API_KEY            # Market data — OHLCV, news
FMP_API_KEY                # Fundamentals, earnings, insider transactions
FINANCIAL_DATASETS_API_KEY # Additional data source
FRED_API_KEY               # Macro indicators (optional — yfinance fallback)
TELEGRAM_BOT_TOKEN         # Telegram alerts
TELEGRAM_CHAT_ID           # Target chat for alerts
DATABASE_URL               # PostgreSQL connection string
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

The app runs on Heroku with two dynos:

```
web:    gunicorn api.app:app --worker-class uvicorn.workers.UvicornWorker
worker: python -m src.worker  (APScheduler with SIGTERM handling)
```

```bash
heroku create your-app-name
heroku addons:create heroku-postgresql:essential-0
heroku config:set ANTHROPIC_API_KEY=... OPENAI_API_KEY=... # etc.
git push heroku main
heroku ps:scale worker=1
```

## Database Schema

Six tables in PostgreSQL:

| Table | Purpose |
|---|---|
| `daily_runs` | One row per pipeline execution (regime, universe size, duration) |
| `candidates` | All tickers that passed scoring (composite score, features) |
| `signals` | Final approved picks (entry, stop, target, thesis, debate summary) |
| `outcomes` | Tracks actual P&L vs predictions (entry date, exit, max favorable/adverse) |
| `pipeline_artifacts` | Full stage envelope per run for traceability |
| `agent_logs` | Raw LLM inputs/outputs with token counts and cost |

## Cost Tracking

Every LLM call is logged with token counts and estimated USD cost. The `/api/costs` endpoint provides daily and per-agent cost breakdowns. Model costs are tracked in `src/agents/llm_router.py`:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| Claude Opus 4.6 | $15.00 | $75.00 |
| Claude Sonnet 4.5 | $3.00 | $15.00 |
| GPT-4.1 | $2.00 | $8.00 |

## Tests

179 tests covering all modules:

```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_signals/ -v       # Signal models + confluence + cooldown
pytest tests/test_agents/ -v        # LLM agent schemas + routing
pytest tests/test_backtest/ -v      # Validation gate + metrics
pytest tests/test_governance/ -v    # Decay detection + retrain policy
pytest tests/test_portfolio/ -v     # Position sizing + trade plans
pytest tests/test_contracts.py -v   # Data contract enforcement
```

## License

Private project. All rights reserved.
