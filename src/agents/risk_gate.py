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
from src.agents.retry import (
    RetryPolicy,
    RetryResult,
    AttemptRecord,
    FailureReason,
    build_retry_prompt_suffix,
)
from src.agents.quality import check_risk_gate_quality
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
        retry_policy: RetryPolicy | None = None,
        memory_context: dict | None = None,
    ) -> RetryResult[RiskGateOutput]:
        """Make final approve/veto/adjust decision with retry support."""
        settings = get_settings()
        policy = retry_policy or RetryPolicy(
            max_attempts=settings.agent_max_retry_attempts,
            max_total_cost_usd=settings.agent_retry_cost_cap_usd,
            retry_on_low_quality=settings.agent_retry_on_low_quality,
        )

        result = RetryResult[RiskGateOutput]()
        base_prompt = self._build_user_prompt(
            ticker=ticker,
            interpretation=interpretation,
            debate_result=debate_result,
            signal_data=signal_data,
            regime_context=regime_context,
            existing_positions=existing_positions or [],
            memory_context=memory_context,
        )
        prompt = base_prompt

        for attempt_num in range(1, policy.max_attempts + 1):
            if result.total_cost_usd >= policy.max_total_cost_usd:
                break

            try:
                llm_result = await call_llm(
                    model=self.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    max_tokens=1000,
                    temperature=0.2,
                )
            except Exception as e:
                logger.error("Risk gate failed for %s: %s", ticker, e)
                result.add_attempt(AttemptRecord(
                    attempt_num=attempt_num, success=False,
                    failure_reason=FailureReason.LLM_API_ERROR, error_message=str(e),
                ))
                if attempt_num < policy.max_attempts and policy.should_retry(FailureReason.LLM_API_ERROR):
                    prompt = base_prompt + build_retry_prompt_suffix(
                        attempt_num + 1, FailureReason.LLM_API_ERROR, str(e)
                    )
                    continue
                # Fail safe: veto on error
                result.value = RiskGateOutput(
                    ticker=ticker,
                    decision=GateDecision.VETO,
                    reasoning=f"Risk gate agent error: {e}. Defaulting to VETO for safety.",
                    position_size_pct=0,
                )
                return result

            self._store_meta(llm_result)
            tokens = llm_result.get("tokens_in", 0) + llm_result.get("tokens_out", 0)
            result.add_cost(tokens, llm_result.get("cost_usd", 0.0))

            content = llm_result["content"]
            raw = llm_result.get("raw", str(content))

            if isinstance(content, str):
                logger.warning("Risk gate returned non-JSON for %s", ticker)
                result.add_attempt(AttemptRecord(
                    attempt_num=attempt_num, success=False,
                    failure_reason=FailureReason.PARSE_ERROR,
                    error_message="Risk gate returned non-JSON",
                    raw_output=raw,
                ))
                if attempt_num < policy.max_attempts and policy.should_retry(FailureReason.PARSE_ERROR):
                    prompt = base_prompt + build_retry_prompt_suffix(
                        attempt_num + 1, FailureReason.PARSE_ERROR, "Response was not valid JSON"
                    )
                    continue
                # Fail safe: veto on parse failure
                result.value = RiskGateOutput(
                    ticker=ticker,
                    decision=GateDecision.VETO,
                    reasoning="Failed to produce structured output. Defaulting to VETO.",
                    position_size_pct=0,
                )
                return result

            # Parse the output
            try:
                gate_output = RiskGateOutput(**content)
            except Exception as e:
                logger.warning("Failed to parse risk gate output for %s: %s", ticker, e)
                # Try to salvage
                decision_str = content.get("decision", "VETO").upper()
                decision = GateDecision.VETO
                if decision_str == "APPROVE":
                    decision = GateDecision.APPROVE
                elif decision_str == "ADJUST":
                    decision = GateDecision.ADJUST

                gate_output = RiskGateOutput(
                    ticker=ticker,
                    decision=decision,
                    reasoning=content.get("reasoning", "Partial parse"),
                    position_size_pct=content.get("position_size_pct", 0),
                    adjusted_stop=content.get("adjusted_stop"),
                    adjusted_target=content.get("adjusted_target"),
                )

            # Check for FUNDAMENTAL_REJECT — legitimate VETO should not be retried
            if gate_output.decision == GateDecision.VETO and len(gate_output.reasoning) >= 10:
                result.value = gate_output
                result.add_attempt(AttemptRecord(
                    attempt_num=attempt_num, success=True,
                ))
                return result

            # Quality check
            quality = check_risk_gate_quality(gate_output)
            if not quality.passed:
                result.add_attempt(AttemptRecord(
                    attempt_num=attempt_num, success=False,
                    failure_reason=FailureReason.LOW_QUALITY,
                    error_message=quality.summary,
                    raw_output=raw,
                ))
                if (
                    policy.retry_on_low_quality
                    and attempt_num < policy.max_attempts
                    and policy.should_retry(FailureReason.LOW_QUALITY)
                ):
                    prompt = base_prompt + build_retry_prompt_suffix(
                        attempt_num + 1, FailureReason.LOW_QUALITY, quality.summary
                    )
                    continue
                # Accept low quality rather than returning None
                result.value = gate_output
                return result

            # Success
            result.value = gate_output
            result.add_attempt(AttemptRecord(attempt_num=attempt_num, success=True))
            return result

        # All retries exhausted — fail safe to VETO
        if result.value is None:
            result.value = RiskGateOutput(
                ticker=ticker,
                decision=GateDecision.VETO,
                reasoning="All retry attempts exhausted. Defaulting to VETO for safety.",
                position_size_pct=0,
            )
        return result

    def _build_user_prompt(self, **kwargs) -> str:
        data: dict = {
            "ticker": kwargs["ticker"],
            "signal_interpretation": kwargs["interpretation"],
            "debate_result": kwargs["debate_result"],
            "signal_data": kwargs["signal_data"],
            "regime": kwargs["regime_context"],
            "existing_positions": kwargs["existing_positions"],
        }
        memory_context = kwargs.get("memory_context")
        if memory_context:
            data["memory"] = memory_context
        return f"Evaluate this trade candidate for final approval.\n\n{json.dumps(data, indent=2, default=str)}"

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT
