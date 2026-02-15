"""FastAPI application — serves HTML reports and JSON data."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import select

from src.db.session import init_db, close_db, get_session
from src.db.models import DailyRun, Signal
from src.output.report import generate_daily_report, generate_performance_report
from src.output.performance import get_performance_summary


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
<p><a href="/performance">30-Day Performance</a></p>
</body></html>"""
    return HTMLResponse(html)


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
