"""SQLAlchemy ORM models for the screener database."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DailyRun(Base):
    """One row per daily pipeline execution."""

    __tablename__ = "daily_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False, index=True, unique=True)
    regime: Mapped[str] = mapped_column(String(20), nullable=False)  # bull / bear / choppy
    regime_details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    universe_size: Mapped[int] = mapped_column(Integer, nullable=False)
    candidates_scored: Mapped[int] = mapped_column(Integer, nullable=False)
    pipeline_duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    execution_mode: Mapped[str | None] = mapped_column(
        String(20), nullable=True, default="agentic_full"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Candidate(Base):
    """A ticker that passed the universe filter on a given run date."""

    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    close_price: Mapped[float] = mapped_column(Float, nullable=False)
    avg_daily_volume: Mapped[float] = mapped_column(Float, nullable=False)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    signal_model: Mapped[str] = mapped_column(String(30), nullable=False)  # breakout / mean_rev / catalyst
    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Signal(Base):
    """Final picks after LLM agent pipeline."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # LONG / SHORT
    signal_model: Mapped[str] = mapped_column(String(30), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    target_1: Mapped[float] = mapped_column(Float, nullable=False)
    target_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_period_days: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)  # 0-100

    # LLM agent outputs
    interpreter_thesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    debate_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_gate_decision: Mapped[str] = mapped_column(String(10), nullable=False)  # APPROVE / VETO / ADJUST
    risk_gate_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    regime: Mapped[str] = mapped_column(String(20), nullable=False)
    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Outcome(Base):
    """Tracks actual performance of each signal."""

    __tablename__ = "outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_date: Mapped[date] = mapped_column(Date, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(20), nullable=True)  # target / stop / expiry
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_dollars: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_favorable: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_adverse: Mapped[float | None] = mapped_column(Float, nullable=True)
    still_open: Mapped[bool] = mapped_column(Boolean, default=True)
    daily_prices: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PipelineArtifact(Base):
    """Persisted stage envelope — one row per pipeline stage per run for full traceability."""

    __tablename__ = "pipeline_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False)  # success / failed
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    errors: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AgentLog(Base):
    """Stores raw LLM agent inputs/outputs for debugging and meta-analysis."""

    __tablename__ = "agent_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    agent_name: Mapped[str] = mapped_column(String(30), nullable=False)
    model_used: Mapped[str] = mapped_column(String(50), nullable=False)
    ticker: Mapped[str | None] = mapped_column(String(10), nullable=True)
    input_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    output_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DivergenceEvent(Base):
    """One record per diff per ticker at decision time — tracks where LLM diverged from quant baseline."""

    __tablename__ = "divergence_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(10), nullable=False)  # VETO / PROMOTE / RESIZE
    execution_mode: Mapped[str] = mapped_column(String(20), nullable=False)

    quant_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    agentic_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quant_size: Mapped[float] = mapped_column(Float, nullable=False, default=5.0)
    agentic_size: Mapped[float] = mapped_column(Float, nullable=False, default=5.0)
    quant_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    agentic_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    reason_codes: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    llm_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime: Mapped[str | None] = mapped_column(String(20), nullable=True)

    quant_baseline_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Denormalized quant trade params for counterfactual simulation
    quant_entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    quant_stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    quant_target_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    quant_holding_period: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quant_direction: Mapped[str | None] = mapped_column(String(10), nullable=True)

    outcome_resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class NearMiss(Base):
    """Signals rejected by debate or risk gate — tracks what was filtered and how close it was."""

    __tablename__ = "near_misses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    # Where it was filtered
    stage: Mapped[str] = mapped_column(String(15), nullable=False)  # debate / risk_gate

    # Debate data (always present — every near-miss went through debate)
    debate_verdict: Mapped[str] = mapped_column(String(10), nullable=False)
    net_conviction: Mapped[float] = mapped_column(Float, nullable=False)
    bull_conviction: Mapped[float | None] = mapped_column(Float, nullable=True)
    bear_conviction: Mapped[float | None] = mapped_column(Float, nullable=True)
    key_risk: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Risk gate data (only if stage == "risk_gate")
    risk_gate_decision: Mapped[str | None] = mapped_column(String(10), nullable=True)
    risk_gate_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Signal context
    interpreter_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_model: Mapped[str | None] = mapped_column(String(30), nullable=True)
    regime: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Trade params for future counterfactual
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    timeframe_days: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Counterfactual (resolved later — NOT in this plan)
    counterfactual_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    counterfactual_exit_reason: Mapped[str | None] = mapped_column(String(20), nullable=True)
    outcome_resolved: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PositionDailyMetric(Base):
    """Daily health card snapshot for each open position."""

    __tablename__ = "position_daily_metrics"
    __table_args__ = (
        UniqueConstraint("signal_id", "metric_date", name="uq_daily_metric_signal_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("signals.id"), nullable=False, index=True
    )
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    metric_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Scores
    promising_score: Mapped[float] = mapped_column(Float, nullable=False)
    health_state: Mapped[str] = mapped_column(String(10), nullable=False)
    trend_score: Mapped[float] = mapped_column(Float, nullable=False)
    momentum_score: Mapped[float] = mapped_column(Float, nullable=False)
    volume_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    regime_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Key raw indicators
    current_price: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=False)
    rsi_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr_stop_distance: Mapped[float | None] = mapped_column(Float, nullable=True)
    rvol: Mapped[float | None] = mapped_column(Float, nullable=True)
    days_held: Mapped[int] = mapped_column(Integer, nullable=False)

    # Velocity
    score_velocity: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Regime context (for analyzing EXIT causes from regime flips)
    signal_regime: Mapped[str | None] = mapped_column(String(20), nullable=True)
    current_regime: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Full component breakdowns
    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Invalidation
    hard_invalidation: Mapped[bool] = mapped_column(Boolean, default=False)
    invalidation_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SignalExitEvent(Base):
    """Records health-driven exit events for learning loop."""

    __tablename__ = "signal_exit_events"
    __table_args__ = (
        UniqueConstraint("signal_id", "exit_reason", name="uq_exit_event_signal_reason"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("signals.id"), nullable=False, index=True
    )
    outcome_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("outcomes.id"), nullable=True, index=True
    )
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    exit_date: Mapped[date] = mapped_column(Date, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=False)
    exit_reason: Mapped[str] = mapped_column(String(20), nullable=False)
    signal_model: Mapped[str] = mapped_column(String(30), nullable=False)
    health_state_at_exit: Mapped[str] = mapped_column(String(10), nullable=False)
    promising_score_at_exit: Mapped[float] = mapped_column(Float, nullable=False)
    invalidation_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    days_held: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DivergenceOutcome(Base):
    """Attached after trade closes — scores the LLM divergence against realized outcomes."""

    __tablename__ = "divergence_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    divergence_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("divergence_events.id"), nullable=False, index=True
    )

    agentic_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    agentic_exit_reason: Mapped[str | None] = mapped_column(String(20), nullable=True)
    quant_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    quant_exit_reason: Mapped[str | None] = mapped_column(String(20), nullable=True)

    max_adverse_excursion: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_favorable_excursion: Mapped[float | None] = mapped_column(Float, nullable=True)

    return_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    improved_vs_quant: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
