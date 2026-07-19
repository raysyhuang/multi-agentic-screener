"""Base agent class and structured output schemas (Pydantic models).

The quant-consumed schemas now live in `src/pipeline_types.py` (LLM-free) and
are re-exported here for backward compatibility. This module keeps only the
LLM-specific schemas and the BaseAgent class.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.pipeline_types import (  # noqa: F401  (re-exported for backward compat)
    DebatePosition,
    DebateResult,
    GateDecision,
    RiskFlag,
    RiskGateOutput,
    SignalInterpretation,
    ThresholdAdjustment,
)


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
    regime_accuracy: float | None = Field(
        default=None, description="Regime detection accuracy 0-1 (null if insufficient data)"
    )
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
