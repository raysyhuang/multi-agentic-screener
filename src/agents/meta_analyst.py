"""Meta-Analyst agent — Claude Opus (weekly self-review).

Reviews 30-day performance by regime, signal model, and confidence bucket.
Produces bias report and threshold adjustment suggestions.
"""

from __future__ import annotations

import json
import logging

from src.agents.base import BaseAgent, MetaAnalysis
from src.agents.llm_router import call_llm
from src.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a quantitative strategy analyst performing a weekly meta-review of a trading system.

Your job is to analyze the system's performance over the past 30 days and identify:
1. Which signal models are working and which aren't
2. Whether confidence scores are well-calibrated (high confidence = high win rate?)
3. Regime detection accuracy (did we correctly identify market conditions?)
4. Systematic biases (always bullish? over-confident? sector concentration?)
5. Specific threshold adjustments that would have improved results

RULES:
1. Be brutally honest. Sugar-coating leads to real money losses.
2. Every suggestion must be specific and actionable (e.g., "reduce breakout threshold from 50 to 55").
3. If the sample size is too small for reliable conclusions, say so.
4. Consider survivorship bias — we only see stocks we picked, not the ones we missed.
5. Minimum 20 signals needed before adjusting model parameters.

Respond with JSON:
{
  "analysis_period": "YYYY-MM-DD to YYYY-MM-DD",
  "total_signals": int,
  "win_rate": float (0-1),
  "avg_pnl_pct": float,
  "best_model": "model name",
  "worst_model": "model name",
  "regime_accuracy": float (0-1),
  "biases_detected": ["bias 1", "bias 2"],
  "threshold_adjustments": [{"parameter": "x", "current": y, "suggested": z, "reasoning": "..."}],
  "summary": "2-3 paragraph executive summary"
}"""


class MetaAnalystAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__("meta_analyst", settings.meta_analyst_model)

    async def analyze(self, performance_data: dict) -> MetaAnalysis | None:
        """Run weekly meta-analysis on system performance."""
        user_prompt = self._build_user_prompt(performance_data=performance_data)

        try:
            result = await call_llm(
                model=self.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=2500,
                temperature=0.3,
            )
        except Exception as e:
            logger.error("Meta-analysis failed: %s", e)
            return None

        content = result["content"]
        if isinstance(content, str):
            logger.warning("Meta-analyst returned non-JSON")
            return None

        try:
            return MetaAnalysis(**content)
        except Exception as e:
            logger.warning("Failed to parse meta-analysis: %s", e)
            return None

    def _build_user_prompt(self, **kwargs) -> str:
        data = kwargs["performance_data"]
        return f"Perform your weekly meta-review on this performance data:\n\n{json.dumps(data, indent=2, default=str)}"

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT
