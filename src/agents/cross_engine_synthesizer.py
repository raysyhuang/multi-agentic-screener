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

from pydantic import BaseModel, Field, model_validator

from src.agents.base import BaseAgent
from src.agents.llm_router import call_llm
from src.config import get_settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a portfolio synthesis agent combining stock picks from multiple independent screening engines into a final portfolio recommendation.

Your inputs are:
1. Verified and weighted picks from engines (Multi-Agentic Screener, KooCore-D, Gemini STST)
2. Convergence data (which tickers appear in multiple engines)
3. Strategy-level convergence data (independent sub-strategy signals per ticker)
4. Credibility weights per engine
5. Regime context

STRATEGY-LEVEL INTELLIGENCE:
Each pick carries strategy tags (e.g. "kc_weekly", "gem_momentum_breakout") identifying
the independent sub-strategies that contributed to it. Use these for richer analysis:
  - 3 independent strategy signals from 2 engines > 2 engine-level agreements with 1 strategy each
  - Cross-engine strategy agreement (e.g. KooCore momentum + Gemini momentum) is stronger
    than same-engine multi-strategy (e.g. KooCore weekly + KooCore pro30)
  - effective_signal_count captures this: cross-engine signals count as 1.0, same-engine extras as 0.5

YOUR TASK:
1. CONVERGENT PICKS: Identify tickers picked by 2+ engines. These are highest conviction.
   - Weight them by combined engine credibility and convergence multiplier.
   - Note which engines agree, their strategies, and strategy_tags.
   - Highlight cross-strategy convergence (e.g. momentum + reversion on same ticker).

2. UNIQUE OPPORTUNITIES: Find strong single-engine picks worth including.
   - Only include if the engine has high credibility AND the pick has high confidence.
   - Picks with multiple independent strategy tags from the same engine get a credibility boost.
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

    @model_validator(mode="before")
    @classmethod
    def _coerce_nulls(cls, values: dict) -> dict:
        """LLM may emit null for numeric fields — coerce to 0.0."""
        for key in ("entry_price", "stop_loss", "target_price", "combined_score"):
            if key in values and values[key] is None:
                values[key] = 0.0
        if "holding_period_days" in values and values["holding_period_days"] is None:
            values["holding_period_days"] = 7
        return values


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

    @model_validator(mode="before")
    @classmethod
    def _coerce_nulls(cls, values: dict) -> dict:
        """LLM may emit null for numeric fields — coerce to 0.0."""
        for key in ("entry_price", "stop_loss", "target_price", "weight_pct"):
            if key in values and values[key] is None:
                values[key] = 0.0
        if "holding_period_days" in values and values["holding_period_days"] is None:
            values["holding_period_days"] = 7
        return values


class SynthesizerOutput(BaseModel):
    convergent_picks: list[ConvergentPick] = Field(default_factory=list)
    unique_picks: list[UniquePick] = Field(default_factory=list)
    portfolio: list[PortfolioPosition] = Field(default_factory=list)
    regime_consensus: str = ""
    executive_summary: str = ""


def _backfill_prices(
    output: SynthesizerOutput,
    weighted_picks: list[dict],
) -> None:
    """Fill in missing prices on portfolio/convergent items from engine data.

    The LLM often omits or nulls price fields. We look up the ticker in
    the original weighted_picks and copy entry/stop/target when the output
    has them at zero (the coerced default).
    """
    lookup: dict[str, dict] = {wp["ticker"]: wp for wp in weighted_picks}

    for pos in output.portfolio:
        wp = lookup.get(pos.ticker)
        if wp is None:
            continue
        if pos.entry_price == 0.0 and wp.get("entry_price"):
            pos.entry_price = float(wp["entry_price"])
        if pos.stop_loss == 0.0 and wp.get("stop_loss"):
            pos.stop_loss = float(wp["stop_loss"])
        if pos.target_price == 0.0 and wp.get("target_price"):
            pos.target_price = float(wp["target_price"])

    for pick in output.convergent_picks:
        wp = lookup.get(pick.ticker)
        if wp is None:
            continue
        if pick.entry_price == 0.0 and wp.get("entry_price"):
            pick.entry_price = float(wp["entry_price"])
        if pick.stop_loss == 0.0 and wp.get("stop_loss"):
            pick.stop_loss = float(wp["stop_loss"])
        if pick.target_price == 0.0 and wp.get("target_price"):
            pick.target_price = float(wp["target_price"])


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

            # Backfill prices from weighted picks when the LLM omitted them
            _backfill_prices(output, weighted_picks)

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
