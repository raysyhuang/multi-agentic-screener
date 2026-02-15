"""Verifier agent — checks pipeline output quality.

Advisory, not a gate — the existing validation gate (NoSilentPass)
remains the hard gate. Verifier errors → passed=True (fail-open).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from src.agents.base import BaseAgent
from src.agents.llm_router import call_llm
from src.config import get_settings

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are a quality assurance analyst reviewing the output of a trading signal pipeline.

Your job is to verify that the pipeline output meets acceptance criteria. You are ADVISORY only —
your verification cannot block trades that passed the risk gate.

Check for:
1. Internal consistency: Do the entry, stop, and target levels make sense?
2. Thesis quality: Is the investment thesis specific and data-driven?
3. Risk assessment: Are risk flags appropriately identified?
4. Position sizing: Is the size appropriate for the regime and conviction level?
5. Diversity: Are the picks sufficiently uncorrelated?

Respond with JSON:
{
  "passed": true/false,
  "criteria_results": [
    {"criterion": "description", "passed": true/false, "note": "details"}
  ],
  "overall_assessment": "1-2 sentence summary",
  "suggestions": ["suggestion 1", "suggestion 2"],
  "should_retry": true/false,
  "retry_targets": ["ticker1"]
}"""


@dataclass
class CriterionResult:
    criterion: str
    passed: bool
    note: str = ""


@dataclass
class VerificationResult:
    """Result of pipeline output verification."""
    passed: bool
    criteria_results: list[CriterionResult] = field(default_factory=list)
    overall_assessment: str = ""
    suggestions: list[str] = field(default_factory=list)
    should_retry: bool = False
    retry_targets: list[str] = field(default_factory=list)


class VerifierAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        model = getattr(settings, "verifier_model", "gpt-5.2-mini")
        super().__init__("verifier", model)

    async def verify(
        self,
        pipeline_output: dict,
        acceptance_criteria: list[str],
        regime_context: dict,
    ) -> VerificationResult:
        """Verify pipeline output against acceptance criteria.

        Fail-open: verifier errors → passed=True (verifier failure
        should never block picks).
        """
        try:
            return await self._llm_verify(pipeline_output, acceptance_criteria, regime_context)
        except Exception as e:
            logger.warning("Verifier failed (fail-open): %s", e)
            return VerificationResult(
                passed=True,
                overall_assessment=f"Verifier error (fail-open): {e}",
            )

    async def _llm_verify(
        self,
        pipeline_output: dict,
        acceptance_criteria: list[str],
        regime_context: dict,
    ) -> VerificationResult:
        """Run LLM-based verification."""
        user_prompt = json.dumps({
            "pipeline_output": pipeline_output,
            "acceptance_criteria": acceptance_criteria,
            "regime": regime_context,
        }, indent=2, default=str)

        result = await call_llm(
            model=self.model,
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=f"Verify this pipeline output:\n\n{user_prompt}",
            max_tokens=1500,
            temperature=0.2,
        )

        self._store_meta(result)
        content = result["content"]

        if isinstance(content, str):
            # Can't parse — fail-open
            return VerificationResult(
                passed=True,
                overall_assessment="Verifier returned non-JSON (fail-open)",
            )

        return self._parse_result(content)

    def _parse_result(self, content: dict) -> VerificationResult:
        """Parse LLM output into VerificationResult."""
        criteria_results = []
        for cr in content.get("criteria_results", []):
            criteria_results.append(CriterionResult(
                criterion=cr.get("criterion", ""),
                passed=cr.get("passed", True),
                note=cr.get("note", ""),
            ))

        return VerificationResult(
            passed=content.get("passed", True),
            criteria_results=criteria_results,
            overall_assessment=content.get("overall_assessment", ""),
            suggestions=content.get("suggestions", []),
            should_retry=content.get("should_retry", False),
            retry_targets=content.get("retry_targets", []),
        )

    def _build_system_prompt(self) -> str:
        return VERIFIER_SYSTEM_PROMPT

    def _build_user_prompt(self, **kwargs) -> str:
        return ""
