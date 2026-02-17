"""FastAPI application — serves HTML reports, JSON data, and the dashboard SPA."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from fastapi import Query

from src.config import get_settings
from src.db.session import init_db, close_db, get_session
from src.db.models import DailyRun, Signal, Outcome, AgentLog
from src.output.report import generate_daily_report, generate_performance_report
from src.output.performance import get_performance_summary

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    settings = get_settings()
    if not settings.api_secret_key:
        logger.warning(
            "API_SECRET_KEY is not set — POST/mutation endpoints are disabled. "
            "Set API_SECRET_KEY in .env or environment for full API access."
        )
    yield
    await close_db()


app = FastAPI(
    title="Multi-Agentic Screener",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
_settings = get_settings()
_origins = [o.strip() for o in _settings.allowed_origins.split(",") if o.strip()] if _settings.allowed_origins else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or [],  # empty = no cross-origin allowed (same-origin only)
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Paths that skip auth entirely (not under /api/)
_AUTH_EXEMPT_PREFIXES = ("/health", "/static", "/docs", "/openapi.json", "/redoc")


@app.middleware("http")
async def api_auth_middleware(request: Request, call_next) -> Response:
    """Require Bearer token only for mutation (POST/PUT/PATCH/DELETE) on /api/* routes.

    GET/read endpoints are public — the dashboard and report pages need them.
    POST/mutation endpoints require auth, and fail-closed when API_SECRET_KEY is not set.
    """
    settings = get_settings()
    path = request.url.path

    # Skip auth for non-API paths
    if not path.startswith("/api/"):
        return await call_next(request)

    # Skip auth for exempt prefixes
    if any(path.startswith(prefix) for prefix in _AUTH_EXEMPT_PREFIXES):
        return await call_next(request)

    # GET requests are public (dashboard, reports, read-only data)
    if request.method == "GET":
        return await call_next(request)

    # --- Mutation endpoints (POST/PUT/PATCH/DELETE) require auth below ---

    # If API_SECRET_KEY is not configured, fail-closed for mutations
    if not settings.api_secret_key:
        return JSONResponse(
            status_code=503,
            content={"detail": "API_SECRET_KEY not configured — mutation endpoints disabled"},
        )

    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid Authorization header. Use: Authorization: Bearer <key>"},
        )

    token = auth_header[7:]  # strip "Bearer "
    if token != settings.api_secret_key:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid API key"},
        )

    return await call_next(request)


# Mount static files
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """List all available daily reports."""
    async with get_session() as session:
        result = await session.execute(
            select(DailyRun).order_by(DailyRun.run_date.desc()).limit(30)
        )
        runs = result.scalars().all()

    links = []
    for run in runs:
        links.append(
            f'<li><a href="/report/{run.run_date}">{run.run_date}</a> '
            f'— {run.regime} — {run.candidates_scored} scored</li>'
        )

    html = f"""<!DOCTYPE html>
<html><head><title>Screener Reports</title>
<style>
body {{ font-family: monospace; background: #0f1117; color: #e4e4e7; padding: 2rem; max-width: 600px; margin: 0 auto; }}
a {{ color: #3b82f6; }}
li {{ margin: 0.5rem 0; }}
</style></head>
<body>
<h1>Daily Reports</h1>
<ul>{''.join(links) if links else '<li>No reports yet. Run the pipeline first.</li>'}</ul>
<p><a href="/performance">30-Day Performance</a> | <a href="/dashboard">Dashboard</a></p>
</body></html>"""
    return HTMLResponse(html)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the SPA dashboard."""
    html_path = STATIC_DIR / "dashboard.html"
    if not html_path.is_file():
        raise HTTPException(404, "Dashboard not found")
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/report/{report_date}", response_class=HTMLResponse)
async def daily_report(report_date: str):
    """Serve a daily HTML report."""
    try:
        rd = date.fromisoformat(report_date)
    except ValueError:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")

    async with get_session() as session:
        run_result = await session.execute(
            select(DailyRun).where(DailyRun.run_date == rd)
        )
        run = run_result.scalar_one_or_none()
        if not run:
            raise HTTPException(404, f"No report for {report_date}")

        sig_result = await session.execute(
            select(Signal).where(Signal.run_date == rd)
        )
        signals = sig_result.scalars().all()

    picks = []
    for s in signals:
        if s.risk_gate_decision == "APPROVE" or s.risk_gate_decision == "ADJUST":
            picks.append({
                "ticker": s.ticker,
                "direction": s.direction,
                "signal_model": s.signal_model,
                "entry_price": s.entry_price,
                "stop_loss": s.stop_loss,
                "target_1": s.target_1,
                "holding_period": s.holding_period_days,
                "confidence": s.confidence,
                "thesis": s.interpreter_thesis,
                "debate_summary": s.debate_summary,
                "risk_flags": [],
            })

    html = generate_daily_report(
        run_date=rd,
        regime=run.regime,
        regime_details=run.regime_details or {},
        picks=picks,
        vetoed=[],
        pipeline_stats={
            "universe_size": run.universe_size,
            "candidates_scored": run.candidates_scored,
        },
    )
    return HTMLResponse(html)


@app.get("/performance", response_class=HTMLResponse)
async def performance_page():
    """30-day performance summary."""
    data = await get_performance_summary(days=30)
    html = generate_performance_report(data, period_days=30)
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# Dashboard API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/dashboard/signals")
async def dashboard_signals():
    """Return the latest run + approved signals shaped for the dashboard."""
    async with get_session() as session:
        run_result = await session.execute(
            select(DailyRun).order_by(DailyRun.run_date.desc()).limit(1)
        )
        run = run_result.scalar_one_or_none()
        if not run:
            return {"run_date": None, "regime": None, "signals": []}

        sig_result = await session.execute(
            select(Signal).where(Signal.run_date == run.run_date)
        )
        signals = sig_result.scalars().all()

    approved = [
        {
            "ticker": s.ticker,
            "direction": s.direction,
            "signal_model": s.signal_model,
            "entry_price": s.entry_price,
            "stop_loss": s.stop_loss,
            "target_1": s.target_1,
            "holding_period_days": s.holding_period_days,
            "confidence": s.confidence,
            "thesis": s.interpreter_thesis,
            "regime": s.regime,
        }
        for s in signals
        if s.risk_gate_decision in ("APPROVE", "ADJUST")
    ]

    return {
        "run_date": str(run.run_date),
        "regime": run.regime,
        "signals": approved,
    }


@app.get("/api/dashboard/performance")
async def dashboard_performance():
    """Return performance data with an equity curve for charting."""
    data = await get_performance_summary(days=30)

    if data.get("total_signals", 0) == 0:
        return data

    # Build equity curve from closed outcomes
    async with get_session() as session:
        from datetime import timedelta
        cutoff = date.today() - timedelta(days=30)
        result = await session.execute(
            select(Outcome).where(
                Outcome.still_open == False,
                Outcome.entry_date >= cutoff,
            ).order_by(Outcome.exit_date.asc())
        )
        closed = result.scalars().all()

    equity_curve = []
    cumulative = 0.0
    for o in closed:
        pnl = o.pnl_pct or 0.0
        cumulative += pnl
        equity_curve.append({
            "time": str(o.exit_date) if o.exit_date else str(o.entry_date),
            "value": round(cumulative, 4),
        })

    data["equity_curve"] = equity_curve
    return data


# ---------------------------------------------------------------------------
# Dashboard Chart Endpoints (PR4)
# ---------------------------------------------------------------------------

@app.get("/api/dashboard/equity-curve")
async def dashboard_equity_curve(days: int = Query(default=90, le=365)):
    """Cumulative walk-forward returns (equity curve)."""
    from src.output.performance import get_equity_curve
    return {"equity_curve": await get_equity_curve(days)}


@app.get("/api/dashboard/drawdown")
async def dashboard_drawdown(days: int = Query(default=90, le=365)):
    """Drawdown curve (area series, red)."""
    from src.output.performance import get_drawdown_curve
    return {"drawdown": await get_drawdown_curve(days)}


@app.get("/api/dashboard/return-distribution")
async def dashboard_return_distribution(days: int = Query(default=90, le=365)):
    """Return distribution by signal model."""
    from src.output.performance import get_return_distribution
    return {"distribution": await get_return_distribution(days)}


@app.get("/api/dashboard/regime-matrix")
async def dashboard_regime_matrix(days: int = Query(default=180, le=365)):
    """Win rate x model x regime matrix."""
    from src.output.performance import get_regime_matrix
    return {"matrix": await get_regime_matrix(days)}


@app.get("/api/dashboard/calibration")
async def dashboard_calibration():
    """Confidence calibration curve (confidence vs actual hit rate)."""
    data = await get_performance_summary(days=90)
    return {"calibration": data.get("confidence_calibration", [])}


@app.get("/api/dashboard/mode-comparison")
async def dashboard_mode_comparison():
    """LLM uplift: quant_only vs hybrid vs agentic_full metrics."""
    from src.output.performance import get_mode_comparison
    return {"comparison": await get_mode_comparison()}


@app.get("/api/cache-stats")
async def cache_stats():
    """Return data cache performance statistics."""
    from src.data.aggregator import DataAggregator
    agg = DataAggregator()
    return agg.get_cache_stats()


# ---------------------------------------------------------------------------
# Existing JSON API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/signals/{report_date}")
async def signals_json(report_date: str):
    """Return signals as JSON for a given date."""
    try:
        rd = date.fromisoformat(report_date)
    except ValueError:
        raise HTTPException(400, "Invalid date")

    async with get_session() as session:
        result = await session.execute(
            select(Signal).where(Signal.run_date == rd)
        )
        signals = result.scalars().all()

    return [
        {
            "ticker": s.ticker,
            "direction": s.direction,
            "signal_model": s.signal_model,
            "entry_price": s.entry_price,
            "stop_loss": s.stop_loss,
            "target_1": s.target_1,
            "confidence": s.confidence,
            "regime": s.regime,
            "risk_gate_decision": s.risk_gate_decision,
        }
        for s in signals
    ]


@app.get("/api/runs")
async def list_runs(limit: int = Query(default=30, le=100)):
    """List all pipeline runs with summary stats."""
    async with get_session() as session:
        result = await session.execute(
            select(DailyRun).order_by(DailyRun.run_date.desc()).limit(limit)
        )
        runs = result.scalars().all()

    return [
        {
            "run_date": str(r.run_date),
            "regime": r.regime,
            "universe_size": r.universe_size,
            "candidates_scored": r.candidates_scored,
            "pipeline_duration_s": r.pipeline_duration_s,
        }
        for r in runs
    ]


@app.get("/api/outcomes")
async def list_outcomes(
    date_from: str | None = Query(default=None, description="YYYY-MM-DD"),
    date_to: str | None = Query(default=None, description="YYYY-MM-DD"),
    regime: str | None = Query(default=None),
    model: str | None = Query(default=None),
    confidence_min: float | None = Query(default=None),
    still_open: bool | None = Query(default=None),
):
    """Query outcomes with optional filters by date, regime, model, confidence."""
    # Validate date parameters upfront
    try:
        parsed_from = date.fromisoformat(date_from) if date_from else None
        parsed_to = date.fromisoformat(date_to) if date_to else None
    except ValueError:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")

    async with get_session() as session:
        query = select(Outcome, Signal).join(Signal, Outcome.signal_id == Signal.id)

        if parsed_from:
            query = query.where(Outcome.entry_date >= parsed_from)
        if parsed_to:
            query = query.where(Outcome.entry_date <= parsed_to)
        if regime:
            query = query.where(Signal.regime == regime)
        if model:
            query = query.where(Signal.signal_model == model)
        if confidence_min is not None:
            query = query.where(Signal.confidence >= confidence_min)
        if still_open is not None:
            query = query.where(Outcome.still_open == still_open)

        query = query.order_by(Outcome.entry_date.desc()).limit(100)
        result = await session.execute(query)
        rows = result.all()

    return [
        {
            "ticker": outcome.ticker,
            "entry_date": str(outcome.entry_date),
            "entry_price": outcome.entry_price,
            "exit_date": str(outcome.exit_date) if outcome.exit_date else None,
            "exit_price": outcome.exit_price,
            "exit_reason": outcome.exit_reason,
            "pnl_pct": outcome.pnl_pct,
            "max_favorable": outcome.max_favorable,
            "max_adverse": outcome.max_adverse,
            "still_open": outcome.still_open,
            "signal_model": signal.signal_model,
            "regime": signal.regime,
            "confidence": signal.confidence,
            "direction": signal.direction,
        }
        for outcome, signal in rows
    ]


@app.get("/api/outcomes/{ticker}")
async def ticker_outcomes(ticker: str):
    """Get all outcomes for a specific ticker."""
    async with get_session() as session:
        result = await session.execute(
            select(Outcome, Signal)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(Outcome.ticker == ticker.upper())
            .order_by(Outcome.entry_date.desc())
        )
        rows = result.all()

    if not rows:
        raise HTTPException(404, f"No outcomes for {ticker}")

    return [
        {
            "entry_date": str(o.entry_date),
            "entry_price": o.entry_price,
            "exit_date": str(o.exit_date) if o.exit_date else None,
            "exit_price": o.exit_price,
            "exit_reason": o.exit_reason,
            "pnl_pct": o.pnl_pct,
            "still_open": o.still_open,
            "signal_model": s.signal_model,
            "regime": s.regime,
            "confidence": s.confidence,
        }
        for o, s in rows
    ]


@app.get("/api/costs")
async def cost_summary(days: int = Query(default=30, le=90)):
    """Daily and per-agent cost breakdown over the specified period."""
    async with get_session() as session:
        cutoff = date.today() - __import__("datetime").timedelta(days=days)
        result = await session.execute(
            select(AgentLog).where(AgentLog.run_date >= cutoff)
        )
        logs = result.scalars().all()

    by_date: dict[str, float] = {}
    by_agent: dict[str, float] = {}
    total_cost = 0.0
    total_tokens_in = 0
    total_tokens_out = 0

    for log in logs:
        cost = log.cost_usd or 0.0
        total_cost += cost
        total_tokens_in += log.tokens_in or 0
        total_tokens_out += log.tokens_out or 0

        d = str(log.run_date)
        by_date[d] = by_date.get(d, 0.0) + cost
        by_agent[log.agent_name] = by_agent.get(log.agent_name, 0.0) + cost

    return {
        "period_days": days,
        "total_cost_usd": round(total_cost, 4),
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "by_date": {k: round(v, 4) for k, v in sorted(by_date.items())},
        "by_agent": {k: round(v, 4) for k, v in sorted(by_agent.items(), key=lambda x: -x[1])},
    }


@app.get("/api/artifacts/{run_id}")
async def get_artifacts(run_id: str):
    """Return all pipeline stage artifacts for a given run."""
    from src.db.models import PipelineArtifact

    async with get_session() as session:
        result = await session.execute(
            select(PipelineArtifact)
            .where(PipelineArtifact.run_id == run_id)
            .order_by(PipelineArtifact.created_at)
        )
        artifacts = result.scalars().all()

    if not artifacts:
        raise HTTPException(404, f"No artifacts for run_id={run_id}")

    return [
        {
            "stage": a.stage,
            "status": a.status,
            "payload": a.payload,
            "errors": a.errors,
            "created_at": str(a.created_at),
        }
        for a in artifacts
    ]


@app.get("/api/meta-reviews")
async def list_meta_reviews(limit: int = Query(default=10, le=50)):
    """Return recent meta-analyst reviews."""
    async with get_session() as session:
        result = await session.execute(
            select(AgentLog)
            .where(AgentLog.agent_name == "meta_analyst")
            .order_by(AgentLog.run_date.desc())
            .limit(limit)
        )
        reviews = result.scalars().all()

    return [
        {
            "run_date": str(r.run_date),
            "model_used": r.model_used,
            "output": r.output_data,
            "cost_usd": r.cost_usd,
        }
        for r in reviews
    ]


# ---------------------------------------------------------------------------
# Threshold Management Endpoints (PR6)
# ---------------------------------------------------------------------------

@app.get("/api/thresholds")
async def get_thresholds():
    """Return current threshold values."""
    from src.governance.threshold_manager import get_current_thresholds
    return {"thresholds": get_current_thresholds()}


@app.get("/api/thresholds/history")
async def threshold_history(limit: int = Query(default=20, le=50)):
    """Return threshold snapshot history."""
    from src.governance.threshold_manager import get_snapshot_history
    return {"history": get_snapshot_history(limit)}


@app.post("/api/thresholds/apply/{run_date}")
@limiter.limit("10/minute")
async def apply_threshold_snapshot(request: Request, run_date: str):
    """Apply a dry-run threshold snapshot (human approval step)."""
    from src.governance.threshold_manager import apply_snapshot
    try:
        date.fromisoformat(run_date)
    except ValueError:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")

    success = apply_snapshot(run_date)
    if not success:
        raise HTTPException(404, f"No applicable snapshot for {run_date}")
    return {"status": "applied", "run_date": run_date}


# ---------------------------------------------------------------------------
# Divergence Ledger Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/divergence/events")
async def divergence_events(
    days: int = Query(default=30, le=365),
    event_type: str | None = Query(default=None, description="VETO|PROMOTE|RESIZE"),
    regime: str | None = Query(default=None),
    resolved: bool | None = Query(default=None),
):
    """Return divergence events with optional filters."""
    from datetime import timedelta
    from src.db.models import DivergenceEvent

    async with get_session() as session:
        cutoff = date.today() - timedelta(days=days)
        query = select(DivergenceEvent).where(DivergenceEvent.run_date >= cutoff)

        if event_type:
            query = query.where(DivergenceEvent.event_type == event_type.upper())
        if regime:
            query = query.where(DivergenceEvent.regime == regime)
        if resolved is not None:
            query = query.where(DivergenceEvent.outcome_resolved == resolved)

        query = query.order_by(DivergenceEvent.run_date.desc()).limit(200)
        result = await session.execute(query)
        events = result.scalars().all()

    return [
        {
            "id": e.id,
            "run_id": e.run_id,
            "run_date": str(e.run_date),
            "ticker": e.ticker,
            "event_type": e.event_type,
            "execution_mode": e.execution_mode,
            "quant_rank": e.quant_rank,
            "agentic_rank": e.agentic_rank,
            "quant_size": e.quant_size,
            "agentic_size": e.agentic_size,
            "quant_score": e.quant_score,
            "agentic_score": e.agentic_score,
            "reason_codes": e.reason_codes,
            "llm_cost_usd": e.llm_cost_usd,
            "confidence": e.confidence,
            "regime": e.regime,
            "outcome_resolved": e.outcome_resolved,
        }
        for e in events
    ]


@app.get("/api/divergence/summary")
async def divergence_summary(days: int = Query(default=60, le=365)):
    """Aggregated divergence stats: by event type, regime, confidence bucket."""
    from datetime import timedelta
    from src.db.models import DivergenceEvent, DivergenceOutcome

    async with get_session() as session:
        cutoff = date.today() - timedelta(days=days)
        query = (
            select(DivergenceEvent, DivergenceOutcome)
            .outerjoin(DivergenceOutcome, DivergenceOutcome.divergence_id == DivergenceEvent.id)
            .where(DivergenceEvent.run_date >= cutoff)
        )
        result = await session.execute(query)
        rows = result.all()

    if not rows:
        return {
            "period_days": days,
            "total_events": 0,
            "by_event_type": {},
            "by_regime": {},
            "by_confidence_bucket": {},
            "total_llm_cost": 0.0,
            "cost_per_positive_divergence": None,
            "overall_improvement_rate": None,
        }

    # Aggregate by event type
    by_type: dict[str, dict] = {}
    by_regime: dict[str, dict[str, dict]] = {}
    by_conf: dict[str, dict] = {}
    total_cost = 0.0
    positive_count = 0

    for event, outcome in rows:
        et = event.event_type
        total_cost += event.llm_cost_usd or 0.0

        # By event type
        if et not in by_type:
            by_type[et] = {"events": 0, "resolved": 0, "improved": 0, "deltas": []}
        by_type[et]["events"] += 1
        if outcome:
            by_type[et]["resolved"] += 1
            if outcome.improved_vs_quant:
                by_type[et]["improved"] += 1
                positive_count += 1
            if outcome.return_delta is not None:
                by_type[et]["deltas"].append(outcome.return_delta)

        # By regime
        reg = event.regime or "unknown"
        if reg not in by_regime:
            by_regime[reg] = {}
        if et not in by_regime[reg]:
            by_regime[reg][et] = {"events": 0, "resolved": 0, "improved": 0, "deltas": []}
        by_regime[reg][et]["events"] += 1
        if outcome:
            by_regime[reg][et]["resolved"] += 1
            if outcome.improved_vs_quant:
                by_regime[reg][et]["improved"] += 1
            if outcome.return_delta is not None:
                by_regime[reg][et]["deltas"].append(outcome.return_delta)

        # By confidence bucket
        conf = event.confidence
        if conf is not None:
            bucket = "high" if conf >= 70 else "medium" if conf >= 40 else "low"
        else:
            bucket = "unknown"
        if bucket not in by_conf:
            by_conf[bucket] = {"events": 0, "resolved": 0, "improved": 0, "deltas": []}
        by_conf[bucket]["events"] += 1
        if outcome:
            by_conf[bucket]["resolved"] += 1
            if outcome.improved_vs_quant:
                by_conf[bucket]["improved"] += 1
            if outcome.return_delta is not None:
                by_conf[bucket]["deltas"].append(outcome.return_delta)

    def _stats(d: dict) -> dict:
        resolved = d["resolved"]
        return {
            "events": d["events"],
            "resolved": resolved,
            "win_rate": round(d["improved"] / resolved, 4) if resolved else None,
            "avg_return_delta": round(sum(d["deltas"]) / len(d["deltas"]), 4) if d["deltas"] else None,
        }

    total_resolved = sum(v["resolved"] for v in by_type.values())

    return {
        "period_days": days,
        "total_events": sum(v["events"] for v in by_type.values()),
        "by_event_type": {k: _stats(v) for k, v in by_type.items()},
        "by_regime": {
            reg: {et: _stats(v) for et, v in types.items()}
            for reg, types in by_regime.items()
        },
        "by_confidence_bucket": {k: _stats(v) for k, v in by_conf.items()},
        "total_llm_cost": round(total_cost, 4),
        "cost_per_positive_divergence": (
            round(total_cost / positive_count, 4) if positive_count > 0 else None
        ),
        "overall_improvement_rate": (
            round(positive_count / total_resolved, 4) if total_resolved > 0 else None
        ),
    }


# ---------------------------------------------------------------------------
# Near-Miss Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/near-misses")
async def near_misses(
    days: int = Query(default=30, le=365),
    stage: str | None = Query(default=None, description="debate|risk_gate"),
    resolved: bool | None = Query(default=None),
):
    """Return near-miss signals with optional filters."""
    from datetime import timedelta
    from src.db.models import NearMiss

    async with get_session() as session:
        cutoff = date.today() - timedelta(days=days)
        query = select(NearMiss).where(NearMiss.run_date >= cutoff)

        if stage:
            query = query.where(NearMiss.stage == stage)
        if resolved is not None:
            query = query.where(NearMiss.outcome_resolved == resolved)

        query = query.order_by(NearMiss.run_date.desc()).limit(200)
        result = await session.execute(query)
        rows = result.scalars().all()

    return [
        {
            "id": nm.id,
            "run_date": str(nm.run_date),
            "ticker": nm.ticker,
            "stage": nm.stage,
            "debate_verdict": nm.debate_verdict,
            "net_conviction": nm.net_conviction,
            "signal_model": nm.signal_model,
            "regime": nm.regime,
            "entry_price": nm.entry_price,
            "stop_loss": nm.stop_loss,
            "target_price": nm.target_price,
            "timeframe_days": nm.timeframe_days,
            "counterfactual_return": nm.counterfactual_return,
            "counterfactual_exit_reason": nm.counterfactual_exit_reason,
            "outcome_resolved": nm.outcome_resolved,
        }
        for nm in rows
    ]


@app.get("/api/near-misses/summary")
async def near_misses_summary(days: int = Query(default=30, le=365)):
    """Aggregated near-miss stats including counterfactual resolution data."""
    from src.output.performance import get_near_miss_stats
    stats = await get_near_miss_stats(days)
    if stats is None:
        return {"period_days": days, "total_near_misses": 0}
    return stats


# ---------------------------------------------------------------------------
# Position Health Endpoints
# ---------------------------------------------------------------------------


def _metric_to_dict(m, *, include_details: bool = True) -> dict:
    """Serialize a PositionDailyMetric row to a JSON-safe dict."""
    d = {
        "ticker": m.ticker,
        "signal_id": m.signal_id,
        "metric_date": str(m.metric_date),
        "promising_score": m.promising_score,
        "health_state": m.health_state,
        "trend_score": m.trend_score,
        "momentum_score": m.momentum_score,
        "volume_score": m.volume_score,
        "risk_score": m.risk_score,
        "regime_score": m.regime_score,
        "current_price": m.current_price,
        "pnl_pct": m.pnl_pct,
        "atr_14": m.atr_14,
        "days_held": m.days_held,
        "score_velocity": m.score_velocity,
        "hard_invalidation": m.hard_invalidation,
        "invalidation_reason": m.invalidation_reason,
    }
    if include_details:
        d["details"] = m.details
    return d


@app.get("/api/positions/health")
async def positions_health(
    include_details: bool = Query(default=False, description="Include component breakdowns"),
):
    """Current health cards for all open positions (latest daily metric per open Outcome)."""
    from src.db.models import PositionDailyMetric

    async with get_session() as session:
        # Get open outcomes
        open_result = await session.execute(
            select(Outcome).where(Outcome.still_open == True)
        )
        open_outcomes = open_result.scalars().all()

        if not open_outcomes:
            return {"positions": []}

        cards = []
        for outcome in open_outcomes:
            metric_result = await session.execute(
                select(PositionDailyMetric)
                .where(PositionDailyMetric.signal_id == outcome.signal_id)
                .order_by(PositionDailyMetric.metric_date.desc())
                .limit(1)
            )
            metric = metric_result.scalar_one_or_none()
            if metric:
                cards.append(_metric_to_dict(metric, include_details=include_details))

    return {"positions": cards}


@app.get("/api/positions/health/signal/{signal_id}/history")
async def position_health_history_by_signal(
    signal_id: int,
    days: int = Query(default=30, le=365),
):
    """Time series of daily health metrics for a specific signal (trade).

    Use this instead of ticker history when the same ticker has been traded
    multiple times — this returns metrics for one specific trade only.
    """
    from datetime import timedelta
    from src.db.models import PositionDailyMetric

    async with get_session() as session:
        cutoff = date.today() - timedelta(days=days)
        result = await session.execute(
            select(PositionDailyMetric)
            .where(
                PositionDailyMetric.signal_id == signal_id,
                PositionDailyMetric.metric_date >= cutoff,
            )
            .order_by(PositionDailyMetric.metric_date.asc())
        )
        metrics = result.scalars().all()

    if not metrics:
        raise HTTPException(404, f"No health history for signal_id={signal_id}")

    return {
        "signal_id": signal_id,
        "ticker": metrics[0].ticker,
        "history": [_metric_to_dict(m) for m in metrics],
    }


@app.get("/api/positions/health/{ticker}/history")
async def position_health_history(
    ticker: str,
    days: int = Query(default=30, le=365),
):
    """Time series of daily health metrics for a ticker (all trades).

    Note: if the same ticker has been traded multiple times, this returns
    metrics for all trades combined. Use /signal/{signal_id}/history for
    per-trade history.
    """
    from datetime import timedelta
    from src.db.models import PositionDailyMetric

    async with get_session() as session:
        cutoff = date.today() - timedelta(days=days)
        result = await session.execute(
            select(PositionDailyMetric)
            .where(
                PositionDailyMetric.ticker == ticker.upper(),
                PositionDailyMetric.metric_date >= cutoff,
            )
            .order_by(PositionDailyMetric.metric_date.asc())
        )
        metrics = result.scalars().all()

    if not metrics:
        raise HTTPException(404, f"No health history for {ticker}")

    return {
        "ticker": ticker.upper(),
        "history": [_metric_to_dict(m) for m in metrics],
    }


# ── Cross-Engine Endpoints ──────────────────────────────────────────────────


@app.post("/api/cross-engine/collect")
async def cross_engine_collect(request: Request, background_tasks: BackgroundTasks):
    """Trigger evening cross-engine collection (Steps 10-14).

    Called by the evening GitHub Actions workflow or Heroku scheduler
    after all three external engines have finished running.
    Runs in the background so the HTTP response returns immediately.
    """
    from src.main import run_evening_collection

    background_tasks.add_task(run_evening_collection)
    return {"status": "started", "message": "Evening cross-engine collection triggered"}


@app.get("/api/cross-engine/latest")
async def cross_engine_latest():
    """Latest cross-engine synthesis results."""
    from src.db.models import CrossEngineSynthesis

    async with get_session() as session:
        result = await session.execute(
            select(CrossEngineSynthesis)
            .order_by(CrossEngineSynthesis.run_date.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()

    if not row:
        raise HTTPException(404, "No cross-engine synthesis found")

    return {
        "run_date": str(row.run_date),
        "regime_consensus": row.regime_consensus,
        "engines_reporting": row.engines_reporting,
        "convergent_tickers": row.convergent_tickers,
        "portfolio_recommendation": row.portfolio_recommendation,
        "executive_summary": row.executive_summary,
        "verifier_notes": row.verifier_notes,
        "credibility_weights": row.credibility_weights,
    }


@app.get("/api/cross-engine/credibility")
async def cross_engine_credibility():
    """Current engine credibility weights and hit rates."""
    from src.db.models import CrossEngineSynthesis

    async with get_session() as session:
        result = await session.execute(
            select(CrossEngineSynthesis)
            .order_by(CrossEngineSynthesis.run_date.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()

    if not row or not row.credibility_weights:
        return {"engines": {}, "note": "No credibility data yet — need 10+ resolved picks per engine"}

    return {"engines": row.credibility_weights, "as_of": str(row.run_date)}


@app.get("/api/cross-engine/convergence/{date_str}")
async def cross_engine_convergence(date_str: str):
    """Which tickers appeared in multiple engines on a given date."""
    from src.db.models import ExternalEngineResult

    try:
        query_date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(400, "Invalid date format, use YYYY-MM-DD")

    async with get_session() as session:
        result = await session.execute(
            select(ExternalEngineResult).where(ExternalEngineResult.run_date == query_date)
        )
        rows = result.scalars().all()

    if not rows:
        raise HTTPException(404, f"No engine results for {date_str}")

    # Extract tickers per engine
    ticker_engines: dict[str, list[str]] = {}
    for row in rows:
        payload = row.payload or {}
        picks = payload.get("picks", [])
        for pick in picks:
            ticker = pick.get("ticker", "")
            if ticker:
                ticker_engines.setdefault(ticker, []).append(row.engine_name)

    # Filter to convergent (2+ engines)
    convergent = {
        ticker: engines
        for ticker, engines in ticker_engines.items()
        if len(engines) >= 2
    }

    return {
        "date": date_str,
        "total_engines": len(rows),
        "convergent_tickers": convergent,
        "all_tickers": {t: e for t, e in ticker_engines.items()},
    }


@app.get("/api/cross-engine/history")
async def cross_engine_history(
    limit: int = Query(default=30, le=90),
):
    """Synthesis history for the last N days."""
    from src.db.models import CrossEngineSynthesis

    async with get_session() as session:
        result = await session.execute(
            select(CrossEngineSynthesis)
            .order_by(CrossEngineSynthesis.run_date.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

    return [
        {
            "run_date": str(row.run_date),
            "regime_consensus": row.regime_consensus,
            "engines_reporting": row.engines_reporting,
            "convergent_count": len(row.convergent_tickers) if row.convergent_tickers else 0,
            "portfolio_size": len(row.portfolio_recommendation) if row.portfolio_recommendation else 0,
            "executive_summary": row.executive_summary,
        }
        for row in rows
    ]


@app.get("/health")
async def health_check():
    """Health check for Heroku / uptime monitors.

    Tests DB connectivity and verifies critical API keys are present.
    Returns 200 with status=healthy, or 503 with status=degraded and issue list.
    """
    issues: list[str] = []

    # Test DB connectivity
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
    except Exception as e:
        issues.append(f"database: {e}")

    # Check critical API keys
    settings = get_settings()
    if not settings.polygon_api_key and not settings.fmp_api_key:
        issues.append("no data provider API key configured (polygon or fmp)")
    if settings.execution_mode != "quant_only":
        if not settings.anthropic_api_key and not settings.openai_api_key:
            issues.append("no LLM API key configured (anthropic or openai)")

    if issues:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "issues": issues},
        )
    return {"status": "healthy"}
