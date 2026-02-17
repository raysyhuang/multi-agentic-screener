"""Cross-Engine Synthesizer Agent — produces final portfolio from verified engine picks.

Takes verified, weighted picks and produces:
- Convergent tickers with combined conviction scores
- Unique high-conviction opportunities (strong single-engine picks)
- Final portfolio recommendation (top 3-5 weighted positions)
- Executive summary for Telegram
- Regime consensus assessment
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.agents.llm_router import call_llm
from src.config import get_settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a portfolio synthesis agent combining stock picks from multiple independent screening engines into a final portfolio recommendation.

Your inputs are:
1. Verified and weighted picks from 4 engines (Multi-Agentic Screener, KooCore-D, Gemini STST, Top3-7D)
2. Convergence data (which tickers appear in multiple engines)
3. Credibility weights per engine
4. Regime context

YOUR TASK:
1. CONVERGENT PICKS: Identify tickers picked by 2+ engines. These are highest conviction.
   - Weight them by combined engine credibility and convergence multiplier.
   - Note which engines agree and their respective strategies.

2. UNIQUE OPPORTUNITIES: Find strong single-engine picks worth including.
   - Only include if the engine has high credibility AND the pick has high confidence.
   - Be selective — single-engine picks need stronger justification.

3. PORTFOLIO CONSTRUCTION: Build a final recommendation of 3-5 positions.
   - Convergent picks get priority.
   - Diversify across strategies (don't over-concentrate in one approach).
   - Consider regime alignment (breakout in bull, reversion in bear).
   - Provide specific entry, stop, and target for each.

4. EXECUTIVE SUMMARY: Write a concise 2-3 sentence summary suitable for Telegram.

5. REGIME CONSENSUS: Assess regime agreement across engines.

Respond with JSON:
{
  "convergent_picks": [
    {
      "ticker": "string",
      "engines": ["string"],
      "combined_score": float,
      "strategy_consensus": "string",
      "entry_price": float,
      "stop_loss": float,
      "target_price": float,
      "holding_period_days": int,
      "thesis": "string"
    }
  ],
  "unique_picks": [
    {
      "ticker": "string",
      "engine": "string",
      "confidence": float,
      "strategy": "string",
      "justification": "string"
    }
  ],
  "portfolio": [
    {
      "ticker": "string",
      "weight_pct": float,
      "source": "convergent | unique",
      "entry_price": float,
      "stop_loss": float,
      "target_price": float,
      "holding_period_days": int
    }
  ],
  "regime_consensus": "string",
  "executive_summary": "string"
}"""


class ConvergentPick(BaseModel):
    ticker: str
    engines: list[str]
    combined_score: float
    strategy_consensus: str = ""
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    holding_period_days: int = 7
    thesis: str = ""


class UniquePick(BaseModel):
    ticker: str
    engine: str
    confidence: float
    strategy: str = ""
    justification: str = ""


class PortfolioPosition(BaseModel):
    ticker: str
    weight_pct: float
    source: str = "convergent"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    holding_period_days: int = 7


class SynthesizerOutput(BaseModel):
    convergent_picks: list[ConvergentPick] = Field(default_factory=list)
    unique_picks: list[UniquePick] = Field(default_factory=list)
    portfolio: list[PortfolioPosition] = Field(default_factory=list)
    regime_consensus: str = ""
    executive_summary: str = ""


class CrossEngineSynthesizerAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__("cross_engine_synthesizer", settings.cross_engine_model)

    async def synthesize(
        self,
        weighted_picks: list[dict],
        verifier_output: dict,
        credibility_weights: dict,
        regime_context: dict,
        screener_picks: list[dict] | None = None,
    ) -> SynthesizerOutput:
        """Synthesize verified picks into final portfolio recommendation."""
        user_prompt = self._build_user_prompt(
            weighted_picks=weighted_picks,
            verifier_output=verifier_output,
            credibility_weights=credibility_weights,
            regime_context=regime_context,
            screener_picks=screener_picks or [],
        )

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
                output = SynthesizerOutput.model_validate(content)
            else:
                logger.warning("Synthesizer returned non-dict response, using empty output")
                output = SynthesizerOutput(
                    executive_summary="Synthesis failed to produce structured output",
                )

            logger.info(
                "Cross-engine synthesis: %d convergent, %d unique, %d portfolio positions, cost=$%.4f",
                len(output.convergent_picks),
                len(output.unique_picks),
                len(output.portfolio),
                llm_result.get("cost_usd", 0),
            )
            return output

        except Exception as e:
            logger.error("Cross-engine synthesizer failed: %s", e)
            return SynthesizerOutput(
                executive_summary=f"Synthesis error: {e}",
            )

    def _build_user_prompt(
        self,
        weighted_picks: list[dict],
        verifier_output: dict,
        credibility_weights: dict,
        regime_context: dict,
        screener_picks: list[dict],
    ) -> str:
        return f"""## Weighted Picks (sorted by combined conviction score)

{json.dumps(weighted_picks, indent=2, default=str)}

## Verifier Assessment

{json.dumps(verifier_output, indent=2, default=str)}

## Engine Credibility Weights

{json.dumps(credibility_weights, indent=2, default=str)}

## Multi-Agentic Screener's Own Picks (for reference)

{json.dumps(screener_picks, indent=2, default=str)}

## Current Regime Context

{json.dumps(regime_context, indent=2, default=str)}

Synthesize these into a final portfolio recommendation."""
