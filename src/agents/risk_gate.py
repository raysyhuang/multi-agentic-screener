"""Risk Gatekeeper agent — Claude Opus (highest-stakes decision).

Final approve/veto/adjust on each candidate. Reviews against:
- Portfolio state (existing positions, correlation)
- Regime context
- Debate outcome
- Position sizing
"""

from __future__ import annotations

import json
import logging

from src.agents.base import BaseAgent, RiskGateOutput, GateDecision
from src.agents.llm_router import call_llm
from src.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior portfolio risk manager making the final approval decision on trade candidates.

This is the highest-stakes decision in the pipeline. Your job is to PROTECT capital first, capture opportunity second.

DECISION FRAMEWORK:
- APPROVE: Signal is strong, risk is managed, regime supports it. Position can proceed.
- ADJUST: Signal has merit but needs tighter stops, smaller size, or different targets.
- VETO: Risk outweighs reward. Reasons include: regime mismatch, excessive correlation, poor risk/reward, unconvincing debate outcome, or insufficient data.

RULES:
1. If the debate verdict was REJECT, you should almost certainly VETO (unless you see strong mitigating factors).
2. If the debate verdict was CAUTIOUS, default to ADJUST with reduced position size.
3. Maximum position size: 10% of portfolio for high-conviction, 5% for moderate.
4. Consider correlation: if we already have positions in the same sector, reduce size.
5. In BEAR regime, max position size drops to 5% (3% for non-mean-reversion signals).
6. Always provide specific, actionable reasoning.

Respond with JSON:
{
  "ticker": "string",
  "decision": "APPROVE | VETO | ADJUST",
  "reasoning": "2-3 sentence reasoning",
  "position_size_pct": 0-100,
  "adjusted_stop": float or null,
  "adjusted_target": float or null,
  "correlation_warning": "string or null",
  "regime_note": "string or null"
}"""


class RiskGateAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__("risk_gatekeeper", settings.risk_gate_model)

    async def evaluate(
        self,
        ticker: str,
        interpretation: dict,
        debate_result: dict,
        signal_data: dict,
        regime_context: dict,
        existing_positions: list[dict] | None = None,
    ) -> RiskGateOutput | None:
        """Make final approve/veto/adjust decision."""
        user_prompt = self._build_user_prompt(
            ticker=ticker,
            interpretation=interpretation,
            debate_result=debate_result,
            signal_data=signal_data,
            regime_context=regime_context,
            existing_positions=existing_positions or [],
        )

        try:
            result = await call_llm(
                model=self.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1000,
                temperature=0.2,  # low temp for risk decisions
            )
        except Exception as e:
            logger.error("Risk gate failed for %s: %s", ticker, e)
            # Fail safe: veto on error — no meta to store
            return RiskGateOutput(
                ticker=ticker,
                decision=GateDecision.VETO,
                reasoning=f"Risk gate agent error: {e}. Defaulting to VETO for safety.",
                position_size_pct=0,
            )

        self._store_meta(result)
        content = result["content"]
        if isinstance(content, str):
            logger.warning("Risk gate returned non-JSON for %s", ticker)
            return RiskGateOutput(
                ticker=ticker,
                decision=GateDecision.VETO,
                reasoning="Failed to produce structured output. Defaulting to VETO.",
                position_size_pct=0,
            )

        try:
            return RiskGateOutput(**content)
        except Exception as e:
            logger.warning("Failed to parse risk gate output for %s: %s", ticker, e)
            # Try to salvage what we can
            decision_str = content.get("decision", "VETO").upper()
            decision = GateDecision.VETO
            if decision_str == "APPROVE":
                decision = GateDecision.APPROVE
            elif decision_str == "ADJUST":
                decision = GateDecision.ADJUST

            return RiskGateOutput(
                ticker=ticker,
                decision=decision,
                reasoning=content.get("reasoning", "Partial parse"),
                position_size_pct=content.get("position_size_pct", 0),
                adjusted_stop=content.get("adjusted_stop"),
                adjusted_target=content.get("adjusted_target"),
            )

    def _build_user_prompt(self, **kwargs) -> str:
        data = {
            "ticker": kwargs["ticker"],
            "signal_interpretation": kwargs["interpretation"],
            "debate_result": kwargs["debate_result"],
            "signal_data": kwargs["signal_data"],
            "regime": kwargs["regime_context"],
            "existing_positions": kwargs["existing_positions"],
        }
        return f"Evaluate this trade candidate for final approval.\n\n{json.dumps(data, indent=2, default=str)}"

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT
