"""FastAPI application — serves HTML reports, JSON data, and the dashboard SPA."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from fastapi import Query

from src.db.session import init_db, close_db, get_session
from src.db.models import DailyRun, Signal, Outcome, AgentLog
from src.output.report import generate_daily_report, generate_performance_report
from src.output.performance import get_performance_summary

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title="Multi-Agentic Screener",
    version="0.1.0",
    lifespan=lifespan,
)

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


@app.get("/health")
async def health_check():
    """Health check for Heroku / uptime monitors."""
    return {"status": "ok"}
