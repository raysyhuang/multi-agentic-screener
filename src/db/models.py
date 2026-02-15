"""SQLAlchemy ORM models for the screener database."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
