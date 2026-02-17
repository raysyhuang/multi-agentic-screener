"""Signal Interpreter agent — Claude Sonnet.

Reads structured features and signal scores, produces an investment thesis
with confidence rating and risk flags. All numeric claims must trace to
API-returned data (LLM grounding rule).
"""

from __future__ import annotations

import json
import logging

from src.agents.base import BaseAgent, SignalInterpretation
from src.agents.llm_router import call_llm
from src.agents.retry import (
    RetryPolicy,
    RetryResult,
    AttemptRecord,
    FailureReason,
    build_retry_prompt_suffix,
)
from src.agents.quality import check_interpretation_quality
from src.config import get_settings
from src.signals.ranker import RankedCandidate

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a quantitative equity analyst specializing in short-term trading signals (5-15 day holds).

Your role is to interpret structured signal data and produce a concise investment thesis.

RULES:
1. Every numeric claim MUST reference the provided data. Never use information from your training data about specific stocks.
2. Be specific: cite exact RSI values, volume ratios, ATR levels, and price levels from the input.
3. Identify risk flags honestly — missing a risk is worse than missing an opportunity.
4. Confidence should reflect data quality and signal alignment, not certainty about the outcome.
5. Confidence > 80 requires: strong technical + fundamental alignment + favorable regime.
6. Confidence < 40 means the signal is marginal at best.

Respond with a JSON object matching this schema:
{
  "ticker": "string",
  "thesis": "2-4 sentence investment thesis",
  "confidence": 0-100,
  "key_drivers": ["factor 1", "factor 2", "factor 3"],
  "risk_flags": ["flag1", "flag2"],
  "suggested_entry": float,
  "suggested_stop": float,
  "suggested_target": float,
  "timeframe_days": int
}

Valid risk_flags: earnings_imminent, high_volatility, low_liquidity, sector_correlation, regime_mismatch, overextended, news_risk"""


class SignalInterpreterAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__("signal_interpreter", settings.signal_interpreter_model)

    async def interpret(
        self,
        candidate: RankedCandidate,
        regime_context: dict,
        retry_policy: RetryPolicy | None = None,
        memory_context: dict | None = None,
    ) -> RetryResult[SignalInterpretation]:
        """Generate thesis for a ranked candidate with retry support."""
        settings = get_settings()
        policy = retry_policy or RetryPolicy(
            max_attempts=settings.agent_max_retry_attempts,
            max_total_cost_usd=settings.agent_retry_cost_cap_usd,
            retry_on_low_quality=settings.agent_retry_on_low_quality,
        )

        result = RetryResult[SignalInterpretation]()
        base_prompt = self._build_user_prompt(candidate, regime_context, memory_context)
        prompt = base_prompt

        for attempt_num in range(1, policy.max_attempts + 1):
            # Cost cap check
            if result.total_cost_usd >= policy.max_total_cost_usd:
                logger.warning(
                    "Cost cap reached for %s (%.4f >= %.4f), stopping retries",
                    candidate.ticker, result.total_cost_usd, policy.max_total_cost_usd,
                )
                break

            # Call LLM
            try:
                llm_result = await call_llm(
                    model=self.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    max_tokens=1500,
                    temperature=0.3,
                )
            except Exception as e:
                logger.error("Signal interpreter failed for %s: %s", candidate.ticker, e)
                record = AttemptRecord(
                    attempt_num=attempt_num,
                    success=False,
                    failure_reason=FailureReason.LLM_API_ERROR,
                    error_message=str(e),
                )
                result.add_attempt(record)
                if attempt_num < policy.max_attempts and policy.should_retry(FailureReason.LLM_API_ERROR):
                    prompt = base_prompt + build_retry_prompt_suffix(
                        attempt_num + 1, FailureReason.LLM_API_ERROR, str(e)
                    )
                    continue
                break

            self._store_meta(llm_result)
            tokens = llm_result.get("tokens_in", 0) + llm_result.get("tokens_out", 0)
            cost = llm_result.get("cost_usd", 0.0)
            result.add_cost(tokens, cost)

            content = llm_result["content"]
            raw = llm_result.get("raw", str(content))

            # Check for JSON parse failure (content is still a string)
            if isinstance(content, str):
                logger.warning("Signal interpreter returned non-JSON for %s", candidate.ticker)
                record = AttemptRecord(
                    attempt_num=attempt_num,
                    success=False,
                    failure_reason=FailureReason.PARSE_ERROR,
                    error_message="LLM returned non-JSON response",
                    raw_output=raw,
                )
                result.add_attempt(record)
                if attempt_num < policy.max_attempts and policy.should_retry(FailureReason.PARSE_ERROR):
                    prompt = base_prompt + build_retry_prompt_suffix(
                        attempt_num + 1, FailureReason.PARSE_ERROR, "Response was not valid JSON"
                    )
                    continue
                break

            # Parse into Pydantic model
            try:
                interpretation = SignalInterpretation(**content)
            except Exception as e:
                logger.warning("Failed to parse interpretation for %s: %s", candidate.ticker, e)
                record = AttemptRecord(
                    attempt_num=attempt_num,
                    success=False,
                    failure_reason=FailureReason.PARSE_ERROR,
                    error_message=str(e),
                    raw_output=raw,
                )
                result.add_attempt(record)
                if attempt_num < policy.max_attempts and policy.should_retry(FailureReason.PARSE_ERROR):
                    prompt = base_prompt + build_retry_prompt_suffix(
                        attempt_num + 1, FailureReason.PARSE_ERROR, str(e)
                    )
                    continue
                break

            # Quality check
            quality = check_interpretation_quality(interpretation)
            if not quality.passed:
                logger.info(
                    "Quality check failed for %s: %s", candidate.ticker, quality.summary
                )
                if (
                    policy.retry_on_low_quality
                    and attempt_num < policy.max_attempts
                    and policy.should_retry(FailureReason.LOW_QUALITY)
                ):
                    record = AttemptRecord(
                        attempt_num=attempt_num,
                        success=False,
                        failure_reason=FailureReason.LOW_QUALITY,
                        error_message=quality.summary,
                        raw_output=raw,
                    )
                    result.add_attempt(record)
                    prompt = base_prompt + build_retry_prompt_suffix(
                        attempt_num + 1, FailureReason.LOW_QUALITY, quality.summary
                    )
                    continue
                # Retries exhausted on low quality — mark as failed with quality warning
                result.value = interpretation
                result.add_attempt(AttemptRecord(
                    attempt_num=attempt_num,
                    success=False,
                    failure_reason=FailureReason.LOW_QUALITY,
                    error_message=quality.summary,
                    quality_warning=True,
                ))
                logger.warning(
                    "Accepting low-quality interpretation for %s (quality_warning=True): %s",
                    candidate.ticker, quality.summary,
                )
                return result

            # Success
            result.value = interpretation
            result.add_attempt(AttemptRecord(attempt_num=attempt_num, success=True))
            return result

        return result

    def _build_user_prompt(
        self,
        candidate: RankedCandidate,
        regime_context: dict,
        memory_context: dict | None = None,
    ) -> str:
        data: dict = {
            "ticker": candidate.ticker,
            "signal_model": candidate.signal_model,
            "raw_score": candidate.raw_score,
            "regime_adjusted_score": candidate.regime_adjusted_score,
            "direction": candidate.direction,
            "entry_price": candidate.entry_price,
            "stop_loss": candidate.stop_loss,
            "target_1": candidate.target_1,
            "target_2": candidate.target_2,
            "holding_period": candidate.holding_period,
            "component_scores": candidate.components,
            "features": candidate.features,
            "regime": regime_context,
        }
        if memory_context:
            data["memory"] = memory_context
        return f"Analyze this signal candidate and produce your thesis:\n\n{json.dumps(data, indent=2, default=str)}"

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT
