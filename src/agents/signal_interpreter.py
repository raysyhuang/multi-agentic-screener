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

    async def interpret(self, candidate: RankedCandidate, regime_context: dict) -> SignalInterpretation | None:
        """Generate thesis for a ranked candidate."""
        user_prompt = self._build_user_prompt(candidate, regime_context)

        try:
            result = await call_llm(
                model=self.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1500,
                temperature=0.3,
            )
        except Exception as e:
            logger.error("Signal interpreter failed for %s: %s", candidate.ticker, e)
            return None

        content = result["content"]
        if isinstance(content, str):
            logger.warning("Signal interpreter returned non-JSON for %s", candidate.ticker)
            return None

        try:
            return SignalInterpretation(**content)
        except Exception as e:
            logger.warning("Failed to parse interpretation for %s: %s", candidate.ticker, e)
            return None

    def _build_user_prompt(self, candidate: RankedCandidate, regime_context: dict) -> str:
        data = {
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
        return f"Analyze this signal candidate and produce your thesis:\n\n{json.dumps(data, indent=2, default=str)}"

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT
