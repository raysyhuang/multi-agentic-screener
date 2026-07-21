"""Shared pipeline data types — LLM-free.

These schemas and dataclasses describe the *shape* of pipeline output
(interpretations, debate/risk-gate results, and the per-run PipelineRun /
PipelineResult containers). They are consumed by the quant-only pipeline
(`src/main.py::_build_quant_only_result`) and by governance, neither of which
should have to import the LLM agent stack.

They previously lived in the LLM agent stack, which pulled a heavy third-party
agent tree into the import graph at process start even in quant_only mode. That
stack has been removed; these types now live here with no LLM dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


# --- Structured output schemas ---

class RiskFlag(str, Enum):
    EARNINGS_IMMINENT = "earnings_imminent"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    SECTOR_CORRELATION = "sector_correlation"
    REGIME_MISMATCH = "regime_mismatch"
    OVEREXTENDED = "overextended"
    NEWS_RISK = "news_risk"


class SignalInterpretation(BaseModel):
    """Output schema for the Signal Interpreter (also filled by quant stubs)."""
    ticker: str
    thesis: str = Field(description="2-4 sentence investment thesis")
    confidence: float = Field(ge=0, le=100, description="Confidence score 0-100")
    key_drivers: list[str] = Field(description="Top 3 factors driving the signal")
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    suggested_entry: float
    suggested_stop: float
    suggested_target: float
    timeframe_days: int


class DebatePosition(BaseModel):
    """One side of the adversarial debate."""
    position: str = Field(description="BULL or BEAR")
    argument: str = Field(description="Core argument in 2-3 sentences")
    evidence: list[str] = Field(description="Specific data points supporting the position")
    weakness: str = Field(description="Acknowledged weakness of this position")
    conviction: float = Field(ge=0, le=100)


class DebateResult(BaseModel):
    """Output of the full adversarial debate."""
    ticker: str
    bull_case: DebatePosition
    bear_case: DebatePosition
    rebuttal_summary: str = Field(description="Summary of the rebuttal exchange")
    final_verdict: str = Field(description="PROCEED, CAUTIOUS, or REJECT")
    net_conviction: float = Field(ge=0, le=100)
    key_risk: str = Field(description="Single biggest risk identified")


class GateDecision(str, Enum):
    APPROVE = "APPROVE"
    VETO = "VETO"
    ADJUST = "ADJUST"


class RiskGateOutput(BaseModel):
    """Output schema for the Risk Gatekeeper (also filled by quant stubs)."""
    ticker: str
    decision: GateDecision
    reasoning: str = Field(description="2-3 sentence reasoning for the decision")
    position_size_pct: float = Field(ge=0, le=100, description="Suggested portfolio allocation %")
    adjusted_stop: float | None = Field(default=None, description="Adjusted stop if ADJUST")
    adjusted_target: float | None = Field(default=None, description="Adjusted target if ADJUST")
    correlation_warning: str | None = None
    regime_note: str | None = None


class ThresholdAdjustment(BaseModel):
    """A suggested threshold change (used by governance/threshold_manager)."""
    parameter: str = Field(description="Config parameter name (e.g., 'vix_high_threshold')")
    current_value: float
    suggested_value: float
    reasoning: str = Field(description="Why this adjustment is suggested")
    confidence: float = Field(ge=0, le=100, description="Confidence in this adjustment")
    evidence_sample_size: int = Field(ge=0, description="Number of trades supporting this suggestion")


# --- Per-run containers ---

@dataclass
class PipelineResult:
    ticker: str
    signal_model: str
    direction: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float | None
    holding_period: int
    confidence: float
    interpretation: SignalInterpretation
    debate: DebateResult
    risk_gate: RiskGateOutput
    features: dict
    signal_source: str = "mas_official"
    max_entry_price: float | None = None
    also_in_mas: bool = False
    suppressed_by_cross_model_ranking: bool = False


@dataclass
class NearMissRecord:
    """In-memory record of a signal rejected at debate or risk gate."""
    ticker: str
    stage: str  # "debate" or "risk_gate"
    debate_verdict: str
    net_conviction: float
    bull_conviction: float | None = None
    bear_conviction: float | None = None
    key_risk: str | None = None
    risk_gate_decision: str | None = None
    risk_gate_reasoning: str | None = None
    interpreter_confidence: float | None = None
    signal_model: str | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None
    timeframe_days: int | None = None


@dataclass
class PipelineRun:
    run_date: date
    regime: str
    regime_details: dict
    candidates_scored: int
    interpreted: int
    debated: int
    approved: list[PipelineResult]
    vetoed: list[str]
    agent_logs: list[dict]
    convergence_state: str = "converged"
    near_misses: list[NearMissRecord] = field(default_factory=list)
