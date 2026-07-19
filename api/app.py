"""FastAPI application — serves HTML reports, JSON data, and the dashboard SPA."""

from __future__ import annotations

import logging
import glob
import math
import re
from collections import Counter
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sqlalchemy as sa
from sqlalchemy import func, select, text
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return 204 No Content to silence favicon 404s."""
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the dashboard as the main entry page."""
    html_path = STATIC_DIR / "dashboard.html"
    if not html_path.is_file():
        raise HTTPException(404, "Dashboard not found")
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/reports", response_class=HTMLResponse)
async def reports_index():
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
async def performance_page(mode: str | None = "quant_only"):
    """30-day performance summary, filtered by execution mode."""
    data = await get_performance_summary(days=30, execution_mode=mode)
    html = generate_performance_report(data, period_days=30)
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# Dashboard API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/dashboard/overview")
async def dashboard_overview():
    """Lightweight overview for dashboard landing page."""
    from src.config import get_settings
    settings = get_settings()
    async with get_session() as session:
        # Latest run
        run_result = await session.execute(
            select(DailyRun).order_by(DailyRun.run_date.desc()).limit(1)
        )
        run = run_result.scalar_one_or_none()

        # Today's signals count
        signals_count = 0
        approved_count = 0
        if run:
            sig_result = await session.execute(
                select(func.count(Signal.id)).where(Signal.run_date == run.run_date)
            )
            signals_count = sig_result.scalar() or 0
            approved_result = await session.execute(
                select(func.count(Signal.id)).where(
                    Signal.run_date == run.run_date,
                    Signal.risk_gate_decision == "APPROVE",
                )
            )
            approved_count = approved_result.scalar() or 0

        # Performance summary (last 90d)
        perf = {}
        try:
            from src.output.performance import get_performance_summary
            perf = await get_performance_summary(days=90, execution_mode="quant_only") or {}
        except Exception:
            pass

        overall = perf.get("overall", {})
        risk = perf.get("risk_metrics", {})

        return {
            "profile": settings.production_profile,
            "regime": run.regime if run else None,
            "run_date": str(run.run_date) if run else None,
            "pipeline_duration_s": run.pipeline_duration_s if run else None,
            "signals_total": signals_count,
            "signals_approved": approved_count,
            "total_trades": perf.get("total_signals", 0),
            "win_rate": overall.get("win_rate"),
            "avg_pnl": overall.get("avg_pnl"),
            "sharpe": risk.get("sharpe_ratio"),
            "sortino": risk.get("sortino_ratio"),
            "profit_factor": risk.get("profit_factor"),
            "max_drawdown_pct": risk.get("max_drawdown_pct"),
            "expectancy": risk.get("expectancy"),
            "execution_mode": settings.execution_mode,
            "trading_mode": settings.trading_mode,
            "score_tiered_stops": settings.score_tiered_stops_enabled,
        }


def _resolve_source(source: str | None) -> str | None:
    """Translate the dashboard ``source`` query param to a DB filter value.

    ``mas_official`` (default) and ``mr_manual_sleeve`` map to themselves.
    ``all`` (or empty) returns ``None`` so the helper skips the filter.
    """
    if source is None or source == "" or source.lower() == "all":
        return None
    return source


@app.get("/api/dashboard/signals")
async def dashboard_signals(source: str | None = "mas_official"):
    """Return the latest run + approved signals shaped for the dashboard."""
    source_filter = _resolve_source(source)

    async with get_session() as session:
        run_result = await session.execute(
            select(DailyRun).order_by(DailyRun.run_date.desc()).limit(1)
        )
        run = run_result.scalar_one_or_none()
        if not run:
            return {
                "run_date": None,
                "regime": None,
                "signals": [],
                "meta": {
                    "total_signals": 0,
                    "approved_signals": 0,
                    "signal_source": source_filter or "all",
                },
                "empty_reason": "No completed pipeline runs yet.",
            }

        sig_query = select(Signal).where(Signal.run_date == run.run_date)
        if source_filter is not None:
            sig_query = sig_query.where(Signal.signal_source == source_filter)
        sig_result = await session.execute(sig_query)
        signals = sig_result.scalars().all()

    approved = [
        {
            "ticker": s.ticker,
            "direction": s.direction,
            "signal_model": s.signal_model,
            "signal_source": s.signal_source,
            "also_in_mas": s.also_in_mas,
            "suppressed_by_cross_model_ranking": s.suppressed_by_cross_model_ranking,
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

    total_signals = len(signals)
    approved_signals = len(approved)
    decisions = Counter((s.risk_gate_decision or "UNKNOWN") for s in signals)
    decision_breakdown = {k: int(v) for k, v in sorted(decisions.items())}

    empty_reason = None
    if total_signals == 0:
        scope = source_filter or "all sources"
        empty_reason = (
            f"No signal records were saved for the latest run yet ({scope}). "
            "If a run is in progress, refresh after completion."
        )
    elif approved_signals == 0:
        breakdown = ", ".join(f"{k}:{v}" for k, v in decision_breakdown.items()) or "none"
        empty_reason = (
            f"{total_signals} signals were generated, but none passed the risk gate "
            f"(decisions: {breakdown})."
        )

    return {
        "run_date": str(run.run_date),
        "regime": run.regime,
        "signals": approved,
        "meta": {
            "total_signals": total_signals,
            "approved_signals": approved_signals,
            "decision_breakdown": decision_breakdown,
            "signal_source": source_filter or "all",
        },
        "empty_reason": empty_reason,
    }


@app.get("/api/dashboard/performance")
async def dashboard_performance(
    mode: str | None = "quant_only",
    source: str | None = "mas_official",
):
    """Return performance data with an equity curve for charting."""
    source_filter = _resolve_source(source)
    data = await get_performance_summary(
        days=90, execution_mode=mode, signal_source=source_filter,
    )

    if data.get("total_signals", 0) == 0:
        return data

    # Reuse the daily-aggregated equity curve (avoids duplicate timestamps)
    from src.output.performance import get_equity_curve
    data["equity_curve"] = await get_equity_curve(
        days=90, execution_mode=mode, signal_source=source_filter,
    )
    return data


# ---------------------------------------------------------------------------
# Dashboard Chart Endpoints (PR4)
# ---------------------------------------------------------------------------

@app.get("/api/dashboard/equity-curve")
async def dashboard_equity_curve(
    days: int = Query(default=90, le=365),
    mode: str | None = "quant_only",
    source: str | None = "mas_official",
):
    """Cumulative walk-forward returns (equity curve)."""
    from src.output.performance import get_equity_curve
    return {
        "equity_curve": await get_equity_curve(
            days, execution_mode=mode, signal_source=_resolve_source(source),
        )
    }


@app.get("/api/dashboard/drawdown")
async def dashboard_drawdown(
    days: int = Query(default=90, le=365),
    mode: str | None = "quant_only",
    source: str | None = "mas_official",
):
    """Drawdown curve (area series, red)."""
    from src.output.performance import get_drawdown_curve
    return {
        "drawdown": await get_drawdown_curve(
            days, execution_mode=mode, signal_source=_resolve_source(source),
        )
    }


@app.get("/api/dashboard/trades")
async def dashboard_trades(
    days: int = Query(default=90, le=365),
    mode: str | None = "quant_only",
    include_open: bool = Query(default=True),
    source: str | None = "mas_official",
):
    """Individual trade history with signal and outcome details."""
    from src.output.performance import get_trades_list
    return {
        "trades": await get_trades_list(
            days,
            execution_mode=mode,
            include_open=include_open,
            signal_source=_resolve_source(source),
        )
    }


@app.get("/api/dashboard/return-distribution")
async def dashboard_return_distribution(
    days: int = Query(default=90, le=365),
    source: str | None = "mas_official",
):
    """Return distribution by signal model."""
    from src.output.performance import get_return_distribution
    return {
        "distribution": await get_return_distribution(
            days, signal_source=_resolve_source(source),
        )
    }


@app.get("/api/dashboard/regime-matrix")
async def dashboard_regime_matrix(
    days: int = Query(default=180, le=365),
    source: str | None = "mas_official",
):
    """Win rate x model x regime matrix."""
    from src.output.performance import get_regime_matrix
    return {
        "matrix": await get_regime_matrix(
            days, signal_source=_resolve_source(source),
        )
    }


@app.get("/api/dashboard/calibration")
async def dashboard_calibration(
    mode: str | None = "quant_only",
    source: str | None = "mas_official",
):
    """Confidence calibration curve (confidence vs actual hit rate)."""
    data = await get_performance_summary(
        days=90, execution_mode=mode, signal_source=_resolve_source(source),
    )
    return {"calibration": data.get("confidence_calibration", [])}


@app.get("/api/dashboard/mode-comparison")
async def dashboard_mode_comparison():
    """Performance comparison across execution modes (quant_only vs legacy)."""
    from src.output.performance import get_mode_comparison
    return {"comparison": await get_mode_comparison()}


@app.get("/api/dashboard/dataset-health")
async def dashboard_dataset_health():
    """Return the latest dataset health report from the most recent DailyRun."""
    async with get_session() as session:
        run_result = await session.execute(
            select(DailyRun).order_by(DailyRun.run_date.desc()).limit(1)
        )
        run = run_result.scalar_one_or_none()

    if not run or not run.dataset_health:
        return {"run_date": None, "health": None}

    return {
        "run_date": str(run.run_date),
        "health": run.dataset_health,
    }


@app.get("/api/config/features")
async def config_features():
    """Return feature flags for the frontend."""
    settings = get_settings()
    return {
        "sniper_enabled": settings.sniper_enabled,
    }


@app.get("/api/dashboard/pipeline-health")
async def dashboard_pipeline_health():
    """Return the latest pipeline health report from the most recent DailyRun."""
    async with get_session() as session:
        run_result = await session.execute(
            select(DailyRun).order_by(DailyRun.run_date.desc()).limit(1)
        )
        run = run_result.scalar_one_or_none()

    if not run:
        return {"run_date": None, "pipeline_health": None, "dataset_health": None}

    return {
        "run_date": str(run.run_date),
        "pipeline_health": run.pipeline_health,
        "dataset_health": run.dataset_health,
    }


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
async def signals_json(report_date: str, source: str | None = "mas_official"):
    """Return signals as JSON for a given date.

    Defaults to official MAS picks. Use ``source=all`` to include manual sleeve
    records as well.
    """
    try:
        rd = date.fromisoformat(report_date)
    except ValueError:
        raise HTTPException(400, "Invalid date")

    source_filter = _resolve_source(source)
    async with get_session() as session:
        query = select(Signal).where(Signal.run_date == rd)
        if source_filter is not None:
            query = query.where(Signal.signal_source == source_filter)
        result = await session.execute(query)
        signals = result.scalars().all()

    return [
        {
            "ticker": s.ticker,
            "direction": s.direction,
            "signal_model": s.signal_model,
            "signal_source": s.signal_source,
            "also_in_mas": s.also_in_mas,
            "suppressed_by_cross_model_ranking": s.suppressed_by_cross_model_ranking,
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


def _safe_float(val: float | None) -> float | None:
    """Convert NaN/inf to None for JSON-safe serialization."""
    if val is None:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


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
            "entry_price": _safe_float(outcome.entry_price),
            "exit_date": str(outcome.exit_date) if outcome.exit_date else None,
            "exit_price": _safe_float(outcome.exit_price),
            "exit_reason": outcome.exit_reason,
            "pnl_pct": _safe_float(outcome.pnl_pct),
            "max_favorable": _safe_float(outcome.max_favorable),
            "max_adverse": _safe_float(outcome.max_adverse),
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
            "entry_price": _safe_float(o.entry_price),
            "exit_date": str(o.exit_date) if o.exit_date else None,
            "exit_price": _safe_float(o.exit_price),
            "exit_reason": o.exit_reason,
            "pnl_pct": _safe_float(o.pnl_pct),
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
        from datetime import timedelta
        cutoff = date.today() - timedelta(days=days)
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


# ---------------------------------------------------------------------------
# Shadow Tracks API
# ---------------------------------------------------------------------------

@app.get("/api/tracks")
async def list_tracks():
    """List all shadow tracks with status and config."""
    from src.db.models import ShadowTrack

    async with get_session() as session:
        result = await session.execute(
            select(ShadowTrack).order_by(ShadowTrack.generation.desc(), ShadowTrack.name)
        )
        tracks = result.scalars().all()

    return {
        "tracks": [
            {
                "id": t.id,
                "name": t.name,
                "generation": t.generation,
                "parent_track": t.parent_track,
                "status": t.status,
                "config": t.config,
                "description": t.description,
                "created_at": str(t.created_at) if t.created_at else None,
            }
            for t in tracks
        ],
        "total": len(tracks),
    }


@app.get("/api/tracks/{name}/picks")
async def track_picks(name: str, limit: int = Query(default=50, ge=1, le=200)):
    """Pick history for one track."""
    from src.db.models import ShadowTrack, ShadowTrackPick

    async with get_session() as session:
        track_result = await session.execute(
            select(ShadowTrack).where(ShadowTrack.name == name)
        )
        track = track_result.scalar_one_or_none()
        if not track:
            raise HTTPException(404, f"Track '{name}' not found")

        picks_result = await session.execute(
            select(ShadowTrackPick)
            .where(ShadowTrackPick.track_id == track.id)
            .order_by(ShadowTrackPick.run_date.desc())
            .limit(limit)
        )
        picks = picks_result.scalars().all()

    return {
        "track_name": name,
        "picks": [
            {
                "run_date": str(p.run_date),
                "ticker": p.ticker,
                "direction": p.direction,
                "strategy": p.strategy,
                "entry_price": p.entry_price,
                "stop_loss": p.stop_loss,
                "target_price": p.target_price,
                "confidence": p.confidence,
                "holding_period": p.holding_period,
                "weight_pct": p.weight_pct,
                "source_engines": p.source_engines,
                "outcome_resolved": p.outcome_resolved,
                "actual_return": p.actual_return,
                "exit_reason": p.exit_reason,
                "days_held": p.days_held,
            }
            for p in picks
        ],
    }


@app.get("/api/tracks/{name}/equity")
async def track_equity(name: str):
    """Equity curve from snapshots for one track."""
    from src.db.models import ShadowTrack, ShadowTrackSnapshot

    async with get_session() as session:
        track_result = await session.execute(
            select(ShadowTrack).where(ShadowTrack.name == name)
        )
        track = track_result.scalar_one_or_none()
        if not track:
            raise HTTPException(404, f"Track '{name}' not found")

        snaps_result = await session.execute(
            select(ShadowTrackSnapshot)
            .where(ShadowTrackSnapshot.track_id == track.id)
            .order_by(ShadowTrackSnapshot.snapshot_date.asc())
        )
        snapshots = snaps_result.scalars().all()

    return {
        "track_name": name,
        "snapshots": [
            {
                "date": str(s.snapshot_date),
                "total_picks": s.total_picks,
                "resolved_picks": s.resolved_picks,
                "win_rate": s.win_rate,
                "avg_return_pct": s.avg_return_pct,
                "total_return": s.total_return,
                "sharpe_ratio": s.sharpe_ratio,
                "profit_factor": s.profit_factor,
                "max_drawdown": s.max_drawdown,
            }
            for s in snapshots
        ],
    }


@app.post("/api/tracks/{name}/promote")
async def promote_track(name: str):
    """Copy winner config to production Settings (auth required via middleware)."""
    from src.db.models import ShadowTrack

    async with get_session() as session:
        result = await session.execute(
            select(ShadowTrack).where(ShadowTrack.name == name)
        )
        track = result.scalar_one_or_none()
        if not track:
            raise HTTPException(404, f"Track '{name}' not found")

        if track.status != "active":
            raise HTTPException(400, f"Track '{name}' is not active (status={track.status})")

        track.status = "promoted"

    return {
        "status": "promoted",
        "track_name": name,
        "config": track.config,
        "note": "Config extracted. Apply overrides to .env or Settings manually to activate in production.",
    }


# ---------------------------------------------------------------------------
# Telegram Log — read/ingest messages from MAS and external engines
# ---------------------------------------------------------------------------


@app.get("/api/telegram/log")
async def telegram_log(
    hours: int = Query(default=24, ge=1, le=168),
    source: str = Query(default="all"),
):
    """Return recent Telegram messages logged by MAS and engines."""
    from datetime import datetime, timedelta, timezone
    from src.db.models import TelegramLog

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    async with get_session() as session:
        stmt = select(TelegramLog).where(TelegramLog.sent_at >= cutoff)
        if source != "all":
            stmt = stmt.where(TelegramLog.source == source)
        stmt = stmt.order_by(TelegramLog.sent_at.desc()).limit(100)
        result = await session.execute(stmt)
        rows = result.scalars().all()

    return {
        "hours": hours,
        "source_filter": source,
        "count": len(rows),
        "messages": [
            {
                "id": r.id,
                "source": r.source,
                "message_text": r.message_text,
                "sent_at": r.sent_at.isoformat() if r.sent_at else None,
                "chat_id": r.chat_id,
                "message_id": r.message_id,
            }
            for r in rows
        ],
    }


@app.post("/api/telegram/ingest")
async def telegram_ingest(request: Request):
    """Receive a Telegram message from an external engine and log it.

    Body: {"source": "koocore_d", "message": "...", "chat_id": "...", "message_id": 123}
    Auth: Bearer token via existing middleware.
    """
    from src.db.models import TelegramLog

    body = await request.json()
    source = body.get("source", "unknown")
    message = body.get("message", "")
    if not message:
        raise HTTPException(400, "message is required")

    valid_sources = {"koocore_d", "gemini_stst", "top3_7d", "mas"}
    if source not in valid_sources:
        raise HTTPException(400, f"source must be one of {valid_sources}")

    async with get_session() as session:
        session.add(TelegramLog(
            source=source,
            message_text=message,
            chat_id=body.get("chat_id"),
            message_id=body.get("message_id"),
        ))

    return {"status": "ok", "source": source, "chars": len(message)}
