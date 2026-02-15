"""Unified memory service — combines episodic and working memory.

Provides a single interface for agents to access both historical
context (episodic) and current run state (working memory).
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from src.memory.episodic import (
    EpisodicContext,
    build_episodic_context,
)
from src.memory.working import WorkingMemory

logger = logging.getLogger(__name__)

# Minimum thresholds for including episodic data in prompts
MIN_SIGNALS_FOR_TICKER_HISTORY = 1
MIN_TRADES_FOR_MODEL_STATS = 5


class MemoryService:
    """Unified memory interface for agents.

    Caches episodic queries per ticker within a run to avoid redundant DB calls.
    """

    def __init__(self, working: WorkingMemory, session: AsyncSession | None = None):
        self.working = working
        self.session = session
        self._episodic_cache: dict[str, EpisodicContext] = {}

    async def get_context_for_candidate(
        self, ticker: str, signal_model: str
    ) -> dict:
        """Get combined episodic + working memory for a candidate.

        Returns a dict formatted for inclusion in agent prompts.
        Only includes history if meaningful (meets minimum thresholds).
        """
        context: dict = {
            "working_memory": self.working.to_prompt_context(),
        }

        if self.session is None:
            return context

        # Check cache first
        cache_key = f"{ticker}:{signal_model}"
        if cache_key not in self._episodic_cache:
            try:
                self._episodic_cache[cache_key] = await build_episodic_context(
                    self.session, ticker, signal_model, self.working.regime,
                )
            except Exception as e:
                logger.warning("Failed to build episodic context for %s: %s", ticker, e)
                return context

        episodic = self._episodic_cache[cache_key]

        # Only include ticker history if we have meaningful data
        if (
            episodic.ticker_history
            and episodic.ticker_history.times_signaled >= MIN_SIGNALS_FOR_TICKER_HISTORY
        ):
            th = episodic.ticker_history
            context["ticker_history"] = {
                "times_signaled": th.times_signaled,
                "times_approved": th.times_approved,
                "times_vetoed": th.times_vetoed,
                "win_rate": th.win_rate,
                "avg_pnl_pct": th.avg_pnl_pct,
                "recent_outcomes": th.recent_outcomes[:5],
            }

        # Only include model stats if we have enough data
        if (
            episodic.model_performance
            and episodic.model_performance.total_signals >= MIN_TRADES_FOR_MODEL_STATS
        ):
            mp = episodic.model_performance
            context["model_performance"] = {
                "signal_model": mp.signal_model,
                "regime": mp.regime,
                "total_signals": mp.total_signals,
                "win_rate": mp.win_rate,
                "avg_pnl_pct": mp.avg_pnl_pct,
                "avg_max_adverse": mp.avg_max_adverse,
            }

        return context
