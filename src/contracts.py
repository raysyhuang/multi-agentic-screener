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

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base for all contract models — unknown fields are forbidden."""

    model_config = ConfigDict(extra="forbid")


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


class StageError(StrictModel):
    code: str
    message: str
    detail: str | None = None


class StageEnvelope(BaseModel):
    """Core envelope wrapping every stage output.

    Uses extra='forbid' to prevent accidental extra fields on the envelope
    itself, while payload: Any allows any stage-specific content.
    """

    model_config = ConfigDict(extra="forbid")

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

class TickerSnapshot(StrictModel):
    ticker: str
    last_price: float
    volume: int
    avg_volume_20d: float | None = None
    market_cap: float | None = None
    source_provenance: str = "polygon"


class DataIngestPayload(StrictModel):
    asof_date: date
    universe: list[TickerSnapshot]


# ── Stage 2: FeaturePayload ─────────────────────────────────────────────

class TickerFeatures(StrictModel):
    ticker: str
    returns_5d: float | None = None
    returns_10d: float | None = None
    rsi_14: float | None = None
    atr_pct: float | None = None
    rvol_20d: float | None = None
    distance_from_sma20: float | None = None
    distance_from_sma50: float | None = None
    feature_quality_flags: list[str] = Field(default_factory=list)


class FeaturePayload(StrictModel):
    asof_date: date
    ticker_features: list[TickerFeatures]


# ── Stage 3: SignalPrefilterPayload ──────────────────────────────────────

class CandidateScores(StrictModel):
    ticker: str
    model_scores: dict[str, float] = Field(
        description="Scores per signal model (breakout, mean_reversion, catalyst)"
    )
    aggregate_score: float
    prefilter_flags: list[str] = Field(default_factory=list)


class SignalPrefilterPayload(StrictModel):
    asof_date: date
    candidates: list[CandidateScores]


# ── Stage 4: RegimePayload ───────────────────────────────────────────────

class RegimeInfo(StrictModel):
    label: str = Field(description="bull, bear, or choppy")
    confidence: float
    signals_allowed: list[str] = Field(
        description="Which signal models can fire under this regime"
    )


class RegimePayload(StrictModel):
    asof_date: date
    regime: RegimeInfo
    gated_candidates: list[str] = Field(
        description="Tickers that passed the regime gate"
    )


# ── Stage 5: AgentReviewPayload ──────────────────────────────────────────

class TickerReview(StrictModel):
    ticker: str
    signal_thesis: str
    signal_confidence: float
    counter_thesis: str | None = None
    confidence_adjustment: float = 0.0
    risk_decision: str = Field(description="approve, veto, or resize")
    risk_notes: str = ""


class AgentReviewPayload(StrictModel):
    ticker_reviews: list[TickerReview]


# ── Stage 6: ValidationPayload ───────────────────────────────────────────

class LeakageChecks(StrictModel):
    asof_timestamp_present: bool = True
    next_bar_execution_enforced: bool = True
    future_data_columns_found: list[str] = Field(default_factory=list)


class FragilityMetrics(StrictModel):
    slippage_sensitivity: float = 0.0
    threshold_sensitivity: float = 0.0
    confidence_calibration_bucket: str = "medium"


class ValidationPayload(StrictModel):
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

class FinalPick(StrictModel):
    ticker: str
    entry_zone: float
    stop_loss: float
    targets: list[float]
    confidence: float
    regime_context: str
    validation_card: dict | None = None


class FinalOutputPayload(StrictModel):
    decision: str = Field(description="Top1To2 or NoTrade")
    picks: list[FinalPick] = Field(default_factory=list, max_length=2)
    no_trade_reason: str | None = None

    @model_validator(mode="after")
    def _no_trade_requires_reason(self) -> FinalOutputPayload:
        if self.decision == "NoTrade" and not self.no_trade_reason:
            raise ValueError("no_trade_reason is required when decision is NoTrade")
        return self


# ── Position Health Card ───────────────────────────────────────────────────


class HealthState(str, Enum):
    ON_TRACK = "on_track"
    WATCH = "watch"
    EXIT = "exit"


class HealthComponent(StrictModel):
    name: str
    score: float = Field(ge=0, le=100)
    weight: float = Field(ge=0, le=1)
    weighted_score: float = Field(ge=0, le=100)
    details: dict[str, float | str | None] = Field(default_factory=dict)


class PositionHealthCard(StrictModel):
    # Component scores
    trend_health: HealthComponent
    momentum_health: HealthComponent
    volume_confirmation: HealthComponent
    risk_integrity: HealthComponent
    regime_alignment: HealthComponent

    # Composite
    promising_score: float = Field(ge=0, le=100)
    state: HealthState
    previous_state: HealthState | None = None
    state_changed: bool = False
    score_velocity: float | None = None

    # Hard invalidation
    hard_invalidation: bool = False
    invalidation_reason: str | None = None

    # Path metrics
    days_held: int
    expected_hold_days: int
    pnl_pct: float
    mfe_pct: float
    mae_pct: float
    current_price: float
    atr_14: float | None = None
    atr_stop_distance: float | None = None

    # Identity
    signal_id: int
    ticker: str
    signal_model: str
    as_of_date: date


class ExitEvent(StrictModel):
    signal_id: int
    ticker: str
    exit_date: date
    exit_price: float
    entry_price: float
    pnl_pct: float
    exit_reason: str
    health_state_at_exit: HealthState
    promising_score_at_exit: float
    invalidation_reason: str | None = None
    days_held: int
    signal_model: str


class HealthCardConfig(StrictModel):
    trend_weight: float = 0.30
    momentum_weight: float = 0.25
    volume_weight: float = 0.15
    risk_weight: float = 0.20
    regime_weight: float = 0.10
    on_track_min: float = 70.0
    watch_min: float = 50.0


# ── External Engine Contract ───────────────────────────────────────────────


class EnginePick(StrictModel):
    """A single pick from an external engine.

    Convention: ``metadata["strategies"]`` is a ``list[str]`` of contributing
    sub-strategy tags (e.g. ``["kc_weekly", "kc_pro30"]`` or
    ``["gem_momentum_breakout", "gem_options_bullish"]``).  This is used by
    the credibility tracker for strategy-level convergence scoring.
    """

    ticker: str
    strategy: str  # "breakout", "mean_reversion", "swing", "momentum", etc.
    entry_price: float
    stop_loss: float | None = None
    target_price: float | None = None
    confidence: float = Field(ge=0, le=100, description="0-100 normalized confidence")
    holding_period_days: int
    thesis: str | None = None
    risk_factors: list[str] = Field(default_factory=list)
    raw_score: float | None = None
    metadata: dict = Field(default_factory=dict)


class EngineResultPayload(StrictModel):
    """Standardized result payload from any external screening engine.

    All engines expose GET /api/engine/results returning this schema.
    """

    engine_name: str  # "koocore_d", "gemini_stst"
    engine_version: str
    run_date: str  # YYYY-MM-DD
    run_timestamp: str  # ISO 8601
    regime: str | None = None
    picks: list[EnginePick]
    candidates_screened: int
    pipeline_duration_s: float | None = None
    status: str = "success"
