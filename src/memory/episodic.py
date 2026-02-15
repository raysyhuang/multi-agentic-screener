"""Episodic memory — DB-backed historical recall.

Queries the outcomes and signals tables to provide agents with
historical context about tickers and model performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class TickerHistory:
    """Historical performance for a specific ticker."""
    ticker: str
    times_signaled: int = 0
    times_approved: int = 0
    times_vetoed: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float | None = None
    avg_pnl_pct: float | None = None
    recent_outcomes: list[dict] = field(default_factory=list)


@dataclass
class ModelPerformance:
    """Performance stats for a signal model in a specific regime."""
    signal_model: str
    regime: str
    total_signals: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float | None = None
    avg_pnl_pct: float | None = None
    avg_max_adverse: float | None = None


@dataclass
class EpisodicContext:
    """Combined episodic context for a candidate."""
    ticker_history: TickerHistory | None = None
    model_performance: ModelPerformance | None = None


async def get_ticker_history(
    session: AsyncSession,
    ticker: str,
    lookback_days: int = 90,
) -> TickerHistory:
    """Get historical signal and outcome data for a ticker."""
    cutoff = date.today() - timedelta(days=lookback_days)
    history = TickerHistory(ticker=ticker)

    # Count signals
    result = await session.execute(
        text("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE risk_gate_decision = 'APPROVE') as approved,
                COUNT(*) FILTER (WHERE risk_gate_decision = 'VETO') as vetoed
            FROM signals
            WHERE ticker = :ticker AND run_date >= :cutoff
        """),
        {"ticker": ticker, "cutoff": cutoff},
    )
    row = result.fetchone()
    if row:
        history.times_signaled = row.total or 0
        history.times_approved = row.approved or 0
        history.times_vetoed = row.vetoed or 0

    # Get outcomes
    result = await session.execute(
        text("""
            SELECT
                o.pnl_pct, o.exit_reason, o.max_adverse,
                o.entry_date, o.exit_date
            FROM outcomes o
            JOIN signals s ON o.signal_id = s.id
            WHERE o.ticker = :ticker AND o.entry_date >= :cutoff
                AND o.still_open = false
            ORDER BY o.entry_date DESC
            LIMIT 10
        """),
        {"ticker": ticker, "cutoff": cutoff},
    )
    outcomes = result.fetchall()

    for o in outcomes:
        pnl = o.pnl_pct
        if pnl is not None:
            if pnl > 0:
                history.win_count += 1
            else:
                history.loss_count += 1
        history.recent_outcomes.append({
            "pnl_pct": pnl,
            "exit_reason": o.exit_reason,
            "max_adverse": o.max_adverse,
            "entry_date": str(o.entry_date) if o.entry_date else None,
            "exit_date": str(o.exit_date) if o.exit_date else None,
        })

    total_closed = history.win_count + history.loss_count
    if total_closed > 0:
        history.win_rate = round(history.win_count / total_closed * 100, 1)
        pnls = [o["pnl_pct"] for o in history.recent_outcomes if o["pnl_pct"] is not None]
        if pnls:
            history.avg_pnl_pct = round(sum(pnls) / len(pnls), 2)

    return history


async def get_model_regime_performance(
    session: AsyncSession,
    signal_model: str,
    regime: str,
    lookback_days: int = 90,
) -> ModelPerformance:
    """Get performance stats for a signal model in a specific regime."""
    cutoff = date.today() - timedelta(days=lookback_days)
    perf = ModelPerformance(signal_model=signal_model, regime=regime)

    result = await session.execute(
        text("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE o.pnl_pct > 0) as wins,
                COUNT(*) FILTER (WHERE o.pnl_pct <= 0) as losses,
                AVG(o.pnl_pct) as avg_pnl,
                AVG(o.max_adverse) as avg_max_adverse
            FROM signals s
            JOIN outcomes o ON o.signal_id = s.id
            WHERE s.signal_model = :model
                AND s.regime = :regime
                AND s.run_date >= :cutoff
                AND o.still_open = false
        """),
        {"model": signal_model, "regime": regime, "cutoff": cutoff},
    )
    row = result.fetchone()
    if row and row.total:
        perf.total_signals = row.total
        perf.win_count = row.wins or 0
        perf.loss_count = row.losses or 0
        if perf.total_signals > 0:
            perf.win_rate = round(perf.win_count / perf.total_signals * 100, 1)
        if row.avg_pnl is not None:
            perf.avg_pnl_pct = round(row.avg_pnl, 2)
        if row.avg_max_adverse is not None:
            perf.avg_max_adverse = round(row.avg_max_adverse, 2)

    return perf


async def build_episodic_context(
    session: AsyncSession,
    ticker: str,
    signal_model: str,
    regime: str,
) -> EpisodicContext:
    """Build combined episodic context for a candidate."""
    ticker_history = await get_ticker_history(session, ticker)
    model_perf = await get_model_regime_performance(session, signal_model, regime)
    return EpisodicContext(
        ticker_history=ticker_history,
        model_performance=model_perf,
    )
