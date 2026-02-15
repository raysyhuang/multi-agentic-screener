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

If divergence data is provided (under the "divergence" key), also analyze:
6. LLM contribution: Is the agentic overlay adding or destroying portfolio value?
   Look at net_portfolio_delta and run_level_deltas — individual wins do NOT guarantee portfolio uplift.
7. Event type effectiveness: VETO/PROMOTE/RESIZE win rates and return deltas independently.
8. Reason code analysis: Which reason codes correlate with positive/negative outcomes?
9. Regime-specific LLM value: In which regimes does the LLM overlay add most/least value?
10. Cost efficiency: Is LLM cost justified? Report net_delta_per_dollar.

RULES:
1. Be brutally honest. Sugar-coating leads to real money losses.
2. Every suggestion must be specific and actionable (e.g., "reduce breakout threshold from 50 to 55").
3. If the sample size is too small for reliable conclusions, say so.
4. Consider survivorship bias — we only see stocks we picked, not the ones we missed.
5. Minimum 20 signals needed before adjusting model parameters.
6. If any divergence bucket (event type, reason code, or regime) has fewer than 10 resolved events,
   treat it as exploratory only — describe the pattern but do NOT recommend structural changes.
7. When analyzing divergence data, distinguish micro-level uplift (individual trade delta) from
   portfolio-level uplift (sum of all deltas in a run). High individual win rate with net negative
   portfolio delta means losses are large when the LLM is wrong.
8. For agent_adjustments, be specific about which agent and conditions
   (e.g., "reduce debate aggressiveness in bull regimes where VETO win rate is below 40%").

Respond with JSON:
{
  "analysis_period": "YYYY-MM-DD to YYYY-MM-DD",
  "total_signals": int,
  "win_rate": float (0-1),
  "avg_pnl_pct": float,
  "best_model": "model name",
  "worst_model": "model name",
  "regime_accuracy": float (0-1) or null if insufficient data,
  "biases_detected": ["bias 1", "bias 2"],
  "threshold_adjustments": [{"parameter": "param_name", "current_value": 50.0, "suggested_value": 55.0, "reasoning": "why", "confidence": 70.0, "evidence_sample_size": 25}],
  "summary": "2-3 paragraph executive summary",
  "divergence_assessment": "2-3 paragraph assessment of LLM overlay contribution (or null if no divergence data)",
  "agent_adjustments": [{"agent": "debate|risk_gate|interpreter", "condition": "...", "adjustment": "...", "reasoning": "..."}]
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
                max_tokens=3500,
                temperature=0.3,
            )
        except Exception as e:
            logger.error("Meta-analysis failed: %s", e)
            return None

        content = result["content"]
        if isinstance(content, str):
            logger.warning("Meta-analyst returned non-JSON")
            return None

        # Sanitize threshold_adjustments — LLMs may produce malformed entries
        # when data is sparse (e.g., None values, prose instead of numbers)
        raw_adjustments = content.get("threshold_adjustments", [])
        if isinstance(raw_adjustments, list):
            numeric_keys = {"current_value", "suggested_value", "confidence", "evidence_sample_size"}
            required_keys = numeric_keys | {"parameter", "reasoning"}
            valid = []
            for adj in raw_adjustments:
                if not isinstance(adj, dict) or not required_keys.issubset(adj.keys()):
                    logger.debug("Dropping threshold_adjustment (missing keys): %s", adj)
                    continue
                # Check numeric fields are actually numeric (not None/str)
                if any(not isinstance(adj.get(k), (int, float)) for k in numeric_keys):
                    logger.debug("Dropping threshold_adjustment (non-numeric values): %s", adj)
                    continue
                valid.append(adj)
            content["threshold_adjustments"] = valid

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
