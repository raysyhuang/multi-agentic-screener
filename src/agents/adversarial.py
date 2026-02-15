"""Adversarial Validator agent — GPT (model diversity for cross-validation).

Runs a structured bull/bear debate on each signal thesis.
2-3 rounds: bullish thesis → bearish attack → rebuttal → final assessment.
"""

from __future__ import annotations

import json
import logging

from src.agents.base import BaseAgent, DebateResult, DebatePosition
from src.agents.llm_router import call_llm
from src.agents.retry import (
    RetryPolicy,
    RetryResult,
    AttemptRecord,
    FailureReason,
    build_retry_prompt_suffix,
)
from src.agents.quality import check_debate_quality
from src.config import get_settings

logger = logging.getLogger(__name__)

BEAR_SYSTEM_PROMPT = """You are a skeptical risk analyst. Your job is to find holes in bullish investment theses.

You MUST:
1. Attack the thesis with specific counter-arguments based ONLY on the provided data.
2. Identify the weakest assumption in the bull case.
3. Consider regime mismatch, sector risks, and technical divergences.
4. Be quantitative: cite specific levels where the thesis breaks down.
5. Give an honest conviction score for the BEAR case.

Respond with JSON:
{
  "position": "BEAR",
  "argument": "Core bear argument in 2-3 sentences",
  "evidence": ["specific data point 1", "specific data point 2", "specific data point 3"],
  "weakness": "The weakest part of the bear case",
  "conviction": 0-100
}"""

REBUTTAL_SYSTEM_PROMPT = """You are a debate moderator reviewing a bull/bear exchange on a stock trade.

Given the bull thesis, bear attack, and the underlying data, provide:
1. A summary of the key points of contention
2. A final verdict: PROCEED (bull wins), CAUTIOUS (mixed), or REJECT (bear wins)
3. Net conviction (0-100) reflecting the probability-weighted outcome
4. The single biggest unresolved risk

Respond with JSON:
{
  "ticker": "string",
  "bull_case": {bull position object},
  "bear_case": {bear position object},
  "rebuttal_summary": "Summary of the debate exchange",
  "final_verdict": "PROCEED | CAUTIOUS | REJECT",
  "net_conviction": 0-100,
  "key_risk": "Single biggest risk"
}"""


class AdversarialAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__("adversarial_validator", settings.adversarial_model)

    async def debate(
        self,
        ticker: str,
        bull_thesis: str,
        signal_data: dict,
        regime_context: dict,
        retry_policy: RetryPolicy | None = None,
        memory_context: dict | None = None,
    ) -> RetryResult[DebateResult]:
        """Run a structured adversarial debate with retry support.

        Round 1: Bear attack on the bull thesis (GPT for model diversity)
        Round 2: Rebuttal synthesis (GPT continues for consistency)
        """
        settings = get_settings()
        policy = retry_policy or RetryPolicy(
            max_attempts=settings.agent_max_retry_attempts,
            max_total_cost_usd=settings.agent_retry_cost_cap_usd,
            retry_on_low_quality=settings.agent_retry_on_low_quality,
        )

        result = RetryResult[DebateResult]()

        # --- Round 1: Bear case (with retries) ---
        bear_case = await self._run_bear_round(
            ticker, bull_thesis, signal_data, regime_context, policy, result, memory_context
        )
        if bear_case is None:
            return result

        # Construct bull case from the thesis
        bull_case = DebatePosition(
            position="BULL",
            argument=bull_thesis,
            evidence=[str(v) for v in list(signal_data.get("component_scores", {}).values())[:3]],
            weakness="Dependent on regime continuation",
            conviction=signal_data.get("confidence", 60),
        )

        # --- Round 2: Rebuttal (with retries) ---
        debate_result = await self._run_rebuttal_round(
            ticker, bull_case, bear_case, signal_data, regime_context, policy, result, memory_context
        )
        if debate_result is None:
            # Return partial result — bear round succeeded but rebuttal failed
            partial = DebateResult(
                ticker=ticker,
                bull_case=bull_case,
                bear_case=bear_case,
                rebuttal_summary="Debate incomplete due to error",
                final_verdict="CAUTIOUS",
                net_conviction=50,
                key_risk="Debate process failed",
            )
            result.value = partial
            return result

        result.value = debate_result
        return result

    async def _run_bear_round(
        self,
        ticker: str,
        bull_thesis: str,
        signal_data: dict,
        regime_context: dict,
        policy: RetryPolicy,
        result: RetryResult,
        memory_context: dict | None = None,
    ) -> DebatePosition | None:
        """Run the bear case round with retries."""
        base_prompt = self._build_bear_prompt(ticker, bull_thesis, signal_data, regime_context, memory_context)
        prompt = base_prompt

        for attempt_num in range(1, policy.max_attempts + 1):
            if result.total_cost_usd >= policy.max_total_cost_usd:
                break

            try:
                bear_result = await call_llm(
                    model=self.model,
                    system_prompt=BEAR_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    max_tokens=1000,
                    temperature=0.4,
                )
            except Exception as e:
                logger.error("Bear case generation failed for %s: %s", ticker, e)
                result.add_attempt(AttemptRecord(
                    attempt_num=attempt_num, success=False,
                    failure_reason=FailureReason.LLM_API_ERROR, error_message=str(e),
                ))
                if attempt_num < policy.max_attempts and policy.should_retry(FailureReason.LLM_API_ERROR):
                    prompt = base_prompt + build_retry_prompt_suffix(
                        attempt_num + 1, FailureReason.LLM_API_ERROR, str(e)
                    )
                    continue
                return None

            self._store_meta(bear_result)
            tokens = bear_result.get("tokens_in", 0) + bear_result.get("tokens_out", 0)
            result.add_cost(tokens, bear_result.get("cost_usd", 0.0))

            bear_content = bear_result["content"]
            raw = bear_result.get("raw", str(bear_content))

            if isinstance(bear_content, str):
                logger.warning("Bear case returned non-JSON for %s", ticker)
                result.add_attempt(AttemptRecord(
                    attempt_num=attempt_num, success=False,
                    failure_reason=FailureReason.PARSE_ERROR,
                    error_message="Bear case returned non-JSON",
                    raw_output=raw,
                ))
                if attempt_num < policy.max_attempts and policy.should_retry(FailureReason.PARSE_ERROR):
                    prompt = base_prompt + build_retry_prompt_suffix(
                        attempt_num + 1, FailureReason.PARSE_ERROR, "Response was not valid JSON"
                    )
                    continue
                return None

            try:
                bear_case = DebatePosition(**bear_content)
            except Exception:
                bear_case = DebatePosition(
                    position="BEAR",
                    argument=bear_content.get("argument", "Unable to parse"),
                    evidence=bear_content.get("evidence", []),
                    weakness=bear_content.get("weakness", "N/A"),
                    conviction=bear_content.get("conviction", 50),
                )

            result.add_attempt(AttemptRecord(attempt_num=attempt_num, success=True))
            return bear_case

        return None

    async def _run_rebuttal_round(
        self,
        ticker: str,
        bull_case: DebatePosition,
        bear_case: DebatePosition,
        signal_data: dict,
        regime_context: dict,
        policy: RetryPolicy,
        result: RetryResult,
        memory_context: dict | None = None,
    ) -> DebateResult | None:
        """Run the rebuttal/synthesis round with retries."""
        base_prompt = self._build_rebuttal_prompt(ticker, bull_case, bear_case, signal_data, regime_context, memory_context)
        prompt = base_prompt

        for attempt_num in range(1, policy.max_attempts + 1):
            if result.total_cost_usd >= policy.max_total_cost_usd:
                break

            try:
                rebuttal_result = await call_llm(
                    model=self.model,
                    system_prompt=REBUTTAL_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    max_tokens=1500,
                    temperature=0.3,
                )
            except Exception as e:
                logger.error("Rebuttal synthesis failed for %s: %s", ticker, e)
                result.add_attempt(AttemptRecord(
                    attempt_num=attempt_num, success=False,
                    failure_reason=FailureReason.LLM_API_ERROR, error_message=str(e),
                ))
                if attempt_num < policy.max_attempts and policy.should_retry(FailureReason.LLM_API_ERROR):
                    prompt = base_prompt + build_retry_prompt_suffix(
                        attempt_num + 1, FailureReason.LLM_API_ERROR, str(e)
                    )
                    continue
                return None

            # Accumulate cost from both rounds
            bear_meta = self.last_call_meta.copy()
            self._store_meta(rebuttal_result)
            self.last_call_meta["tokens_in"] += bear_meta.get("tokens_in", 0)
            self.last_call_meta["tokens_out"] += bear_meta.get("tokens_out", 0)
            self.last_call_meta["cost_usd"] += bear_meta.get("cost_usd", 0.0)
            self.last_call_meta["latency_ms"] += bear_meta.get("latency_ms", 0)

            tokens = rebuttal_result.get("tokens_in", 0) + rebuttal_result.get("tokens_out", 0)
            result.add_cost(tokens, rebuttal_result.get("cost_usd", 0.0))

            rebuttal_content = rebuttal_result["content"]
            raw = rebuttal_result.get("raw", str(rebuttal_content))

            if isinstance(rebuttal_content, str):
                # Try to salvage as partial result
                debate = DebateResult(
                    ticker=ticker,
                    bull_case=bull_case,
                    bear_case=bear_case,
                    rebuttal_summary=rebuttal_content[:500],
                    final_verdict="CAUTIOUS",
                    net_conviction=50,
                    key_risk="Unable to parse structured debate output",
                )
                # Check quality — if it passes, accept it
                quality = check_debate_quality(debate)
                if quality.passed:
                    result.add_attempt(AttemptRecord(attempt_num=attempt_num, success=True))
                    return debate

                result.add_attempt(AttemptRecord(
                    attempt_num=attempt_num, success=False,
                    failure_reason=FailureReason.PARSE_ERROR,
                    error_message="Rebuttal returned non-JSON",
                    raw_output=raw,
                ))
                if attempt_num < policy.max_attempts and policy.should_retry(FailureReason.PARSE_ERROR):
                    prompt = base_prompt + build_retry_prompt_suffix(
                        attempt_num + 1, FailureReason.PARSE_ERROR, "Response was not valid JSON"
                    )
                    continue
                # Return the partial result anyway — better than nothing
                return debate

            try:
                rebuttal_content["bull_case"] = bull_case.model_dump()
                rebuttal_content["bear_case"] = bear_case.model_dump()
                debate = DebateResult(**rebuttal_content)
            except Exception as e:
                logger.warning("Failed to parse debate result for %s: %s", ticker, e)
                debate = DebateResult(
                    ticker=ticker,
                    bull_case=bull_case,
                    bear_case=bear_case,
                    rebuttal_summary=rebuttal_content.get("rebuttal_summary", "Parse error"),
                    final_verdict=rebuttal_content.get("final_verdict", "CAUTIOUS"),
                    net_conviction=rebuttal_content.get("net_conviction", 50),
                    key_risk=rebuttal_content.get("key_risk", "Unknown"),
                )

            # Quality check
            quality = check_debate_quality(debate)
            if not quality.passed and policy.retry_on_low_quality:
                result.add_attempt(AttemptRecord(
                    attempt_num=attempt_num, success=False,
                    failure_reason=FailureReason.LOW_QUALITY,
                    error_message=quality.summary,
                ))
                if attempt_num < policy.max_attempts and policy.should_retry(FailureReason.LOW_QUALITY):
                    prompt = base_prompt + build_retry_prompt_suffix(
                        attempt_num + 1, FailureReason.LOW_QUALITY, quality.summary
                    )
                    continue

            result.add_attempt(AttemptRecord(attempt_num=attempt_num, success=True))
            return debate

        return None

    def _build_bear_prompt(
        self, ticker: str, bull_thesis: str, signal_data: dict, regime_context: dict,
        memory_context: dict | None = None,
    ) -> str:
        data: dict = {
            "ticker": ticker,
            "bull_thesis": bull_thesis,
            "signal_data": signal_data,
            "regime": regime_context,
        }
        if memory_context:
            data["memory"] = memory_context
        return f"Attack the following bullish thesis. Use ONLY the provided data.\n\n{json.dumps(data, indent=2, default=str)}"

    def _build_rebuttal_prompt(
        self, ticker: str, bull: DebatePosition, bear: DebatePosition,
        signal_data: dict, regime_context: dict,
        memory_context: dict | None = None,
    ) -> str:
        data: dict = {
            "ticker": ticker,
            "bull_case": bull.model_dump(),
            "bear_case": bear.model_dump(),
            "signal_data": signal_data,
            "regime": regime_context,
        }
        if memory_context:
            data["memory"] = memory_context
        return f"Synthesize this debate and render a final verdict.\n\n{json.dumps(data, indent=2, default=str)}"

    def _build_system_prompt(self) -> str:
        return BEAR_SYSTEM_PROMPT

    def _build_user_prompt(self, **kwargs) -> str:
        return ""
