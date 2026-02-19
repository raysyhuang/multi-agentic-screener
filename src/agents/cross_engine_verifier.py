"""Cross-Engine Verifier Agent — LLM credibility audit before synthesis.

Audits engine picks against historical accuracy, resolves conflicts,
detects anomalies, and recommends weight adjustments for the current
market regime. Acts as a verification layer before final synthesis.
"""

from __future__ import annotations

import asyncio
import json
import logging

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.agents.llm_router import call_llm
from src.config import get_settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior cross-engine verification analyst auditing stock picks from multiple independent screening engines.

Your job is to verify credibility, resolve conflicts, and ensure quality before picks are synthesized into a final portfolio.

AUDIT TASKS:
1. CREDIBILITY AUDIT: Review each engine's picks against their historical accuracy stats.
   - Flag engines picking outside their usual profile (anomalous behavior).
   - Note if an engine with poor recent performance is making high-confidence picks.

2. CONFLICT RESOLUTION: When engines disagree on a ticker (one bullish, another absent or negative):
   - Analyze which engine's reasoning and track record is stronger for this type of pick.
   - Note if the disagreement itself is informative (e.g., momentum vs reversion conflict).

3. RED FLAG DETECTION: Identify suspicious patterns:
   - Engine producing far more or fewer picks than usual.
   - Picks clustering in one sector (concentration risk).
   - Confidence scores that seem uncalibrated vs historical hit rates.

4. WEIGHT RECOMMENDATION: Based on current regime context:
   - In BEAR regime: recommend higher weight for mean-reversion engines.
   - In BULL regime: recommend higher weight for momentum/breakout engines.
   - In CHOPPY regime: recommend conservative weighting across all engines.

Respond with JSON:
{
  "verified_picks": [
    {
      "ticker": "string",
      "engines_agreeing": ["engine_name1", "engine_name2"],
      "verification_status": "verified | flagged | rejected",
      "adjusted_confidence": 0-100,
      "notes": "string"
    }
  ],
  "weight_adjustments": {
    "engine_name": {"weight_multiplier": float, "reason": "string"}
  },
  "red_flags": ["string"],
  "regime_recommendation": "string",
  "summary": "2-3 sentence overall assessment"
}"""


class VerifiedPick(BaseModel):
    ticker: str
    engines_agreeing: list[str]
    verification_status: str = Field(description="verified, flagged, or rejected")
    adjusted_confidence: float = Field(ge=0, le=100)
    notes: str = ""


class WeightAdjustment(BaseModel):
    weight_multiplier: float
    reason: str


class VerifierOutput(BaseModel):
    verified_picks: list[VerifiedPick]
    weight_adjustments: dict[str, WeightAdjustment] = Field(default_factory=dict)
    red_flags: list[str] = Field(default_factory=list)
    regime_recommendation: str = ""
    summary: str = ""


class CrossEngineVerifierAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__("cross_engine_verifier", settings.cross_engine_model)

    async def verify(
        self,
        engine_results: list[dict],
        credibility_stats: dict,
        regime_context: dict,
    ) -> VerifierOutput:
        """Verify engine picks and produce credibility-audited output."""
        user_prompt = self._build_user_prompt(
            engine_results=engine_results,
            credibility_stats=credibility_stats,
            regime_context=regime_context,
        )

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                llm_result = await call_llm(
                    model=self.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_tokens=3000,
                    temperature=0.2,
                )
                self._store_meta(llm_result)

                content = llm_result.get("content")
                if isinstance(content, dict):
                    output = VerifierOutput.model_validate(content)
                else:
                    logger.warning("Verifier returned non-dict response, using empty output")
                    output = VerifierOutput(
                        verified_picks=[],
                        summary="Verifier failed to produce structured output",
                    )

                logger.info(
                    "Cross-engine verifier: %d verified picks, %d red flags, cost=$%.4f",
                    len(output.verified_picks),
                    len(output.red_flags),
                    llm_result.get("cost_usd", 0),
                )
                return output

            except Exception as e:
                last_error = e
                if attempt == 0:
                    logger.warning(
                        "Cross-engine verifier attempt %d failed: %s; retrying once",
                        attempt + 1,
                        e,
                    )
                    await asyncio.sleep(1)
                    continue
                logger.exception("Cross-engine verifier failed after retry")

        # Fail-open: return all picks as-is without verification
        return VerifierOutput(
            verified_picks=[],
            summary=f"Verification skipped due to error: {last_error}",
        )

    def _build_user_prompt(
        self,
        engine_results: list[dict],
        credibility_stats: dict,
        regime_context: dict,
    ) -> str:
        return f"""## Engine Results

{json.dumps(engine_results, indent=2, default=str)}

## Engine Credibility Stats (30-day rolling)

{json.dumps(credibility_stats, indent=2, default=str)}

## Current Regime Context

{json.dumps(regime_context, indent=2, default=str)}

Perform your credibility audit and provide your verification assessment."""
