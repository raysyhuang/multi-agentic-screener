"""Base agent class and structured output schemas (Pydantic models)."""

from __future__ import annotations

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
    """Output schema for Signal Interpreter agent."""
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
    """Output schema for Risk Gatekeeper agent."""
    ticker: str
    decision: GateDecision
    reasoning: str = Field(description="2-3 sentence reasoning for the decision")
    position_size_pct: float = Field(ge=0, le=100, description="Suggested portfolio allocation %")
    adjusted_stop: float | None = Field(default=None, description="Adjusted stop if ADJUST")
    adjusted_target: float | None = Field(default=None, description="Adjusted target if ADJUST")
    correlation_warning: str | None = None
    regime_note: str | None = None


class ThresholdAdjustment(BaseModel):
    """A suggested threshold change from the Meta-Analyst."""
    parameter: str = Field(description="Config parameter name (e.g., 'vix_high_threshold')")
    current_value: float
    suggested_value: float
    reasoning: str = Field(description="Why this adjustment is suggested")
    confidence: float = Field(ge=0, le=100, description="Confidence in this adjustment")
    evidence_sample_size: int = Field(ge=0, description="Number of trades supporting this suggestion")


class AgentAdjustment(BaseModel):
    """A suggested change to agent behavior from the Meta-Analyst."""
    agent: str = Field(description="Agent to adjust: debate, risk_gate, or interpreter")
    condition: str = Field(description="When/where to apply")
    adjustment: str = Field(description="What to change")
    reasoning: str = Field(description="Why, with evidence")


class MetaAnalysis(BaseModel):
    """Output schema for weekly Meta-Analyst agent."""
    analysis_period: str
    total_signals: int
    win_rate: float
    avg_pnl_pct: float
    best_model: str
    worst_model: str
    regime_accuracy: float
    biases_detected: list[str]
    threshold_adjustments: list[ThresholdAdjustment] = Field(
        description="Suggested parameter changes",
        default_factory=list,
    )
    summary: str
    divergence_assessment: str | None = Field(
        default=None,
        description="Assessment of LLM overlay contribution to portfolio performance",
    )
    agent_adjustments: list[AgentAdjustment] = Field(
        default_factory=list,
        description="Suggested changes to agent behavior based on divergence patterns",
    )


# --- Base agent class ---

class BaseAgent:
    """Base class for all LLM agents."""

    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model
        self.last_call_meta: dict = {}  # tokens_in, tokens_out, latency_ms, cost_usd

    def _store_meta(self, llm_result: dict) -> None:
        """Store metadata from the last LLM call for cost/usage tracking."""
        self.last_call_meta = {
            "model": llm_result.get("model", self.model),
            "tokens_in": llm_result.get("tokens_in", 0),
            "tokens_out": llm_result.get("tokens_out", 0),
            "latency_ms": llm_result.get("latency_ms", 0),
            "cost_usd": llm_result.get("cost_usd", 0.0),
        }

    def _build_system_prompt(self) -> str:
        """Override in subclasses."""
        raise NotImplementedError

    def _build_user_prompt(self, **kwargs) -> str:
        """Override in subclasses."""
        raise NotImplementedError
