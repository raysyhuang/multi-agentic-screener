# Multi-Agentic Short-Term Stock Screener

A fully autonomous stock screening system that scans NYSE/NASDAQ daily to surface 1-2 high-conviction short-term trade candidates (5-15 day holds, targeting 5-10% per trade). Built as an agentic runtime — not a static pipeline — it plans its own execution, reasons with tools, learns from past runs, retries on failure, and self-verifies before releasing picks.

The system runs autonomously on a daily schedule: morning pipeline at 6 AM ET, afternoon position checks at 4:30 PM ET, and weekly meta-reviews on Sundays.

## Architecture

```
L0  Scheduler          APScheduler (cron), triggers daily at 6:00 AM ET
L1  Data Ingestion     Async Python — Polygon + FMP + yfinance + FRED (semaphore-controlled)
L2  Feature Engine     pandas, pandas-ta, numpy — 30+ technical indicators
L3  Signal & Filter    Rule-based models + regime gate + confluence + cooldown + correlation
L4  Autonomous Runtime Planner → [Interpret → Skills → Debate → Risk Gate] → Verifier
L5  Validation         7-check NoSilentPass gate + Deflated Sharpe Ratio
L6  Output             PostgreSQL, Telegram alerts, HTML reports, FastAPI
L7  Governance         Audit trail, decay detection, retrain policy, portfolio construction
```

Layers 1-3 and 5 produce useful candidates with zero LLM cost. Layer 4 adds autonomous reasoning with memory, tool-use, and self-correction. Layer 7 provides model lifecycle management and position sizing.

## Autonomous Runtime (Layer 4)

The core of the system is an autonomous runtime that plans, executes, verifies, and retries — similar to how agentic coding tools operate. Five integrated subsystems:

### Planner

An LLM planner (GPT-5.2-mini) decomposes the daily goal into an execution plan with step dependencies and a cost budget. If the planner fails, a default linear plan mirrors the proven pipeline — guaranteeing no regression.

### Agent Pipeline with Retry

Four agents with intentional model diversity for cross-validation. Each agent call is wrapped in a `RetryResult` that automatically retries on parse errors and API failures, with correction prompts and cost caps per candidate:

| Agent | Model | Role |
|---|---|---|
| Signal Interpreter | Claude Sonnet 4.5 | Reads features, produces thesis + confidence 0-100 + risk flags |
| Adversarial Validator | GPT-5.2 | Attacks thesis from a different model's perspective (bull/bear debate) |
| Risk Gatekeeper | Claude Opus 4.6 | Final approve/veto/adjust — fail-safe to VETO on any error |
| Meta-Analyst (weekly) | Claude Opus 4.6 | Reviews 30-day performance + divergence stats, detects biases, suggests adjustments |

The pipeline narrows a universe of ~200 stocks down to 1-2 picks:

```
~3000 NYSE/NASDAQ  →  ~200 filtered  →  ~50 scored  →  10 interpreted
→  5 debated  →  1-2 risk-gated  →  7-check validation  →  Final picks
```

### Memory Service

Agents have access to both episodic and working memory:

- **Episodic memory** (DB-backed) — how many times a ticker has been signaled, approved, or vetoed; win rate and recent outcomes; model performance by regime over the last 90 days
- **Working memory** (in-process) — tracks the current run's state: which tickers have been interpreted/vetoed/approved, accumulated cost, verifier feedback from previous retry rounds

Memory context is injected into agent prompts so the risk gate knows "we signaled AAPL 3 days ago and it lost 4%" before deciding.

### Tool-Use

Agents can request specific data during reasoning rather than having everything pre-loaded into the prompt. The risk gate can look up earnings dates, check sector exposure, or query model performance stats on demand:

| Tool | Description |
|---|---|
| `lookup_price_history` | Recent OHLCV bars via DataAggregator |
| `lookup_ticker_outcomes` | Past trading results from the outcomes table |
| `get_model_stats` | Signal model performance in a specific regime |
| `check_earnings_date` | Upcoming earnings within 14 days |
| `get_sector_exposure` | Current portfolio sector concentration |

Tools are backed by real DB queries and data APIs when a session is provided, with stub handlers for testing. The LLM router supports multi-turn tool-use loops for both Anthropic and OpenAI APIs (capped at 3 rounds to prevent runaway loops).

### Skill Engine

Declarative YAML playbooks that fire automatically based on context:

- **Pre-Earnings** — When a candidate has earnings within 14 days: checks beat history, caps position size at 5%
- **High-Volatility** — In bear/choppy regimes: tightens stops, reduces position size
- **Sector Rotation** — Always: checks portfolio sector exposure before adding a new position

Skills execute tool calls and inject prompt addons into downstream agents (e.g., the adversarial debate receives "earnings in 5 days, beat history: 4/4 quarters" as additional context).

### Verifier with Verify-Redo-Converge Loop

After the pipeline completes, a verifier agent (GPT-5.2-mini) checks output against the plan's acceptance criteria. If the verifier flags issues:

1. Suggestions are injected into working memory
2. Targeted candidates are re-run through the full pipeline
3. Retry agents receive the verifier's feedback via memory context
4. Re-verification checks if the issues were resolved

This loop runs up to 2 rounds, converging when the verifier passes or the cost budget is exhausted. The pipeline tracks convergence state: `converged`, `budget_exhausted`, or `max_retries`.

The verifier is **semi-gating**: it can trigger retries and inject feedback, but it cannot block picks that passed the risk gate. Safety remains fail-closed at the risk gate level.

### Cost Circuit Breaker

A global cost budget (`max_run_cost_usd`, default $2.00) is enforced at three levels:

- **Per-candidate**: each agent retry is cost-capped at $0.50
- **Between stages**: budget check after interpretation and debate stages — early stop if exhausted
- **Retry loop**: verifier retries halt when the run budget is exceeded

Typical run cost: ~$0.30-0.45 including retries and verification.

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
- Market breadth (% of universe above 20d SMA)

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

## Project Structure

```
src/
├── config.py                   # Environment + typed settings (incl. cost caps, retry limits)
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
├── agents/                     # L4 — Autonomous runtime
│   ├── base.py                 # Base agent + Pydantic output schemas
│   ├── llm_router.py           # Routes to Claude/GPT, multi-turn tool-use, cost tracking
│   ├── orchestrator.py         # Plan → execute → verify → retry loop
│   ├── planner.py              # LLM planner with default fallback
│   ├── verifier.py             # Output verification, retry suggestions
│   ├── signal_interpreter.py   # Claude Sonnet — thesis generation with retry
│   ├── adversarial.py          # GPT — bull/bear debate with retry
│   ├── risk_gate.py            # Claude Opus — approve/veto/adjust with retry
│   ├── meta_analyst.py         # Claude Opus — weekly self-review
│   ├── retry.py                # RetryPolicy, RetryResult, AttemptRecord
│   ├── quality.py              # Output quality checks for each agent
│   └── tools.py                # ToolRegistry, 5 built-in tools (stub + DB-backed)
├── memory/                     # Agent memory
│   ├── episodic.py             # DB-backed historical recall (ticker history, model stats)
│   ├── working.py              # In-process run state + verifier feedback
│   └── service.py              # Unified interface, caches episodic queries
├── skills/                     # Declarative skill playbooks
│   ├── engine.py               # YAML loader, precondition matching, template rendering
│   └── definitions/
│       ├── pre_earnings.yaml   # Earnings proximity analysis
│       ├── high_volatility.yaml # Bear/choppy regime adjustments
│       └── sector_rotation.yaml # Portfolio sector exposure check
├── governance/                 # Model lifecycle management
│   ├── artifacts.py            # Audit trail, GovernanceContext per run
│   ├── divergence_ledger.py    # Per-decision LLM attribution (VETO/PROMOTE/RESIZE)
│   ├── threshold_manager.py    # Threshold adjustment proposals with dry-run
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
    ├── models.py               # 8 SQLAlchemy models (DailyRun, Signal, Outcome, DivergenceEvent, etc.)
    └── session.py              # Async PostgreSQL connection

api/
└── app.py                      # FastAPI — reports, signals, outcomes, costs, artifacts

tests/                          # 379 tests across all modules
```

## Data Contracts

Every pipeline stage communicates through typed `StageEnvelope` wrappers with `extra="forbid"` enforcement. Seven stage payloads are defined in `src/contracts.py`:

```
DataIngest → Feature → SignalPrefilter → Regime → AgentReview → Validation → FinalOutput
```

All payloads inherit from `StrictModel` (Pydantic `extra="forbid"`) — unknown fields cause immediate rejection.

## Safety Mechanisms

- **Fail-closed pipeline** — any unhandled exception guarantees a NoTrade DB record and Telegram alert, never a silent abort
- **Risk gate fail-safe** — LLM errors default to VETO, not APPROVE; fundamental rejections are never retried
- **Verifier is semi-gating** — can trigger retries and inject feedback, but cannot block approved picks
- **Cost circuit breaker** — global run budget ($2.00), per-candidate caps ($0.50), tool round caps (3 max)
- **30-day paper trading gate** — LIVE mode blocked until 30+ days of paper trading with positive profit factor
- **Correlation filter** — drops picks with >0.75 return correlation to prevent concentrated risk
- **Signal cooldown** — suppresses re-triggering same ticker for 5 calendar days
- **Confluence detection** — multi-model agreement boosts confidence (2+ models = 10%+ score bonus)
- **Decay detection** — rolling metrics monitor for hit rate collapse, MAE expansion, or negative expectancy
- **Divergence ledger** — tracks every LLM override (VETO/PROMOTE/RESIZE) with counterfactual return deltas; portfolio-level aggregation feeds weekly meta-analyst for second-order self-evaluation
- **Governance audit trail** — every run captures regime, model versions, decay status, config hash, and git commit
- **Portfolio construction** — Kelly/volatility-scaled sizing with regime exposure multipliers and liquidity caps
- **No look-ahead** — signals fire on day T close, execute at T+1 open
- **Realistic costs** — 10 bps slippage + commissions in all validation
- **Memory-safe batching** — semaphore-controlled concurrency (20 max) + batch+GC for Heroku 512 MB

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

- Python 3.11+
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
OPENAI_API_KEY             # GPT API (adversarial validator, planner, verifier)
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

Eight tables in PostgreSQL:

| Table | Purpose |
|---|---|
| `daily_runs` | One row per pipeline execution (regime, universe size, duration, execution mode) |
| `candidates` | All tickers that passed scoring (composite score, features) |
| `signals` | Final approved picks (entry, stop, target, thesis, debate summary) |
| `outcomes` | Tracks actual P&L vs predictions (entry date, exit, max favorable/adverse) |
| `pipeline_artifacts` | Full stage envelope per run for traceability |
| `agent_logs` | Raw LLM inputs/outputs with token counts and cost |
| `divergence_events` | Per-decision LLM attribution — VETO/PROMOTE/RESIZE with reason codes |
| `divergence_outcomes` | Counterfactual results — agentic vs quant return delta per divergence |

## Cost Tracking

Every LLM call is logged with token counts and estimated USD cost. The `/api/costs` endpoint provides daily and per-agent cost breakdowns:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| Claude Opus 4.6 | $15.00 | $75.00 |
| Claude Sonnet 4.5 | $3.00 | $15.00 |
| Claude Haiku 4.5 | $0.80 | $4.00 |
| GPT-5.2 | $2.00 | $8.00 |
| GPT-5.2-mini | $0.40 | $1.60 |
| o3-mini | $1.10 | $4.40 |

## Tests

379 tests covering all modules:

```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_agents/ -v        # Agents, retry, tools, planner, verifier, quality
pytest tests/test_signals/ -v       # Signal models + confluence + cooldown
pytest tests/test_memory/ -v        # Episodic, working memory, memory service
pytest tests/test_skills/ -v        # Skill engine, YAML loading, precondition matching
pytest tests/test_backtest/ -v      # Validation gate + metrics
pytest tests/test_governance/ -v    # Decay detection, retrain policy, divergence ledger, thresholds
pytest tests/test_portfolio/ -v     # Position sizing + trade plans
pytest tests/test_contracts.py -v   # Data contract enforcement
```

## License

Private project. All rights reserved.
