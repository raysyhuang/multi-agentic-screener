"""Data contracts — typed payloads between every pipeline stage.

Implements the stage envelope and 7 stage-specific payloads from
docs/data_contracts.md. Every stage output is wrapped in a StageEnvelope
to enforce strict stage boundaries and prevent hidden coupling.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ── Core Envelope ──────────────────────────────────────────────────────────

class StageStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class StageName(str, Enum):
    DATA_INGEST = "data_ingest"
    FEATURE = "feature"
    SIGNAL_PREFILTER = "signal_prefilter"
    REGIME = "regime"
    AGENT_REVIEW = "agent_review"
    VALIDATION = "validation"
    FINAL_OUTPUT = "final_output"


class StageError(BaseModel):
    code: str
    message: str
    detail: str | None = None


class StageEnvelope(BaseModel):
    """Core envelope wrapping every stage output."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    stage: StageName
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: StageStatus = StageStatus.SUCCESS
    payload: Any  # stage-specific payload
    errors: list[StageError] = Field(default_factory=list)

    @model_validator(mode="after")
    def _errors_imply_failed(self) -> StageEnvelope:
        if self.errors and self.status == StageStatus.SUCCESS:
            self.status = StageStatus.FAILED
        return self


# ── Stage 1: DataIngestPayload ──────────────────────────────────────────

class TickerSnapshot(BaseModel):
    ticker: str
    last_price: float
    volume: int
    avg_volume_20d: float | None = None
    market_cap: float | None = None
    source_provenance: str = "polygon"


class DataIngestPayload(BaseModel):
    asof_date: date
    universe: list[TickerSnapshot]


# ── Stage 2: FeaturePayload ─────────────────────────────────────────────

class TickerFeatures(BaseModel):
    ticker: str
    returns_5d: float | None = None
    returns_10d: float | None = None
    rsi_14: float | None = None
    atr_pct: float | None = None
    rvol_20d: float | None = None
    distance_from_sma20: float | None = None
    distance_from_sma50: float | None = None
    feature_quality_flags: list[str] = Field(default_factory=list)


class FeaturePayload(BaseModel):
    asof_date: date
    ticker_features: list[TickerFeatures]


# ── Stage 3: SignalPrefilterPayload ──────────────────────────────────────

class CandidateScores(BaseModel):
    ticker: str
    model_scores: dict[str, float] = Field(
        description="Scores per signal model (breakout, mean_reversion, catalyst)"
    )
    aggregate_score: float
    prefilter_flags: list[str] = Field(default_factory=list)


class SignalPrefilterPayload(BaseModel):
    asof_date: date
    candidates: list[CandidateScores]


# ── Stage 4: RegimePayload ───────────────────────────────────────────────

class RegimeInfo(BaseModel):
    label: str = Field(description="bull, bear, or choppy")
    confidence: float
    signals_allowed: list[str] = Field(
        description="Which signal models can fire under this regime"
    )


class RegimePayload(BaseModel):
    asof_date: date
    regime: RegimeInfo
    gated_candidates: list[str] = Field(
        description="Tickers that passed the regime gate"
    )


# ── Stage 5: AgentReviewPayload ──────────────────────────────────────────

class TickerReview(BaseModel):
    ticker: str
    signal_thesis: str
    signal_confidence: float
    counter_thesis: str | None = None
    confidence_adjustment: float = 0.0
    risk_decision: str = Field(description="approve, veto, or resize")
    risk_notes: str = ""


class AgentReviewPayload(BaseModel):
    ticker_reviews: list[TickerReview]


# ── Stage 6: ValidationPayload ───────────────────────────────────────────

class LeakageChecks(BaseModel):
    asof_timestamp_present: bool = True
    next_bar_execution_enforced: bool = True
    future_data_columns_found: list[str] = Field(default_factory=list)


class FragilityMetrics(BaseModel):
    slippage_sensitivity: float = 0.0
    threshold_sensitivity: float = 0.0
    confidence_calibration_bucket: str = "medium"


class ValidationPayload(BaseModel):
    leakage_checks: LeakageChecks
    fragility_metrics: FragilityMetrics
    validation_status: str = Field(description="pass or fail")
    checks: dict[str, str] = Field(
        default_factory=dict,
        description="Map of check_name -> pass|fail",
    )
    fragility_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0-1, lower is safer"
    )
    key_risks: list[str] = Field(default_factory=list)
    notes: str = ""


# ── Stage 7: FinalOutputPayload ──────────────────────────────────────────

class FinalPick(BaseModel):
    ticker: str
    entry_zone: float
    stop_loss: float
    targets: list[float]
    confidence: float
    regime_context: str
    validation_card: dict | None = None


class FinalOutputPayload(BaseModel):
    decision: str = Field(description="Top1To2 or NoTrade")
    picks: list[FinalPick] = Field(default_factory=list, max_length=2)
    no_trade_reason: str | None = None

    @model_validator(mode="after")
    def _no_trade_requires_reason(self) -> FinalOutputPayload:
        if self.decision == "NoTrade" and not self.no_trade_reason:
            raise ValueError("no_trade_reason is required when decision is NoTrade")
        return self
