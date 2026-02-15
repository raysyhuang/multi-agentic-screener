"""Integration tests for the extended meta-analyst with divergence data."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.meta_analyst import MetaAnalystAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_llm_response(**overrides) -> dict:
    """Minimal valid MetaAnalysis JSON from the LLM."""
    base = {
        "analysis_period": "2025-03-01 to 2025-03-31",
        "total_signals": 30,
        "win_rate": 0.60,
        "avg_pnl_pct": 1.5,
        "best_model": "breakout",
        "worst_model": "catalyst",
        "regime_accuracy": 0.75,
        "biases_detected": ["Overconfident in choppy regime"],
        "threshold_adjustments": [],
        "summary": "System performed well overall.",
    }
    base.update(overrides)
    return base


def _performance_data_with_divergence() -> dict:
    """Performance data enriched with divergence stats."""
    return {
        "period_days": 30,
        "total_signals": 25,
        "overall": {"trades": 25, "win_rate": 0.60, "avg_pnl": 1.5},
        "divergence": {
            "period_days": 30,
            "total_events": 10,
            "total_resolved": 8,
            "overall_improvement_rate": 0.625,
            "net_portfolio_delta": 3.5,
            "by_event_type": {
                "VETO": {"events": 5, "resolved": 4, "win_rate": 0.75, "avg_return_delta": 1.2, "total_cost": 0.20},
            },
            "by_reason_code": {},
            "by_regime": {},
            "run_level_deltas": [],
            "run_level_trend": None,
            "cost_efficiency": {
                "total_llm_cost": 0.50,
                "total_positive_delta_generated": 5.0,
                "cost_per_positive_divergence": 0.10,
                "net_delta_per_dollar": 7.0,
            },
        },
    }


def _performance_data_without_divergence() -> dict:
    return {
        "period_days": 30,
        "total_signals": 25,
        "overall": {"trades": 25, "win_rate": 0.60, "avg_pnl": 1.5},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnalyzeWithDivergenceData:
    @pytest.mark.asyncio
    async def test_divergence_key_in_prompt_and_new_fields_in_output(self):
        llm_response = _base_llm_response(
            divergence_assessment="The LLM overlay is net positive with a portfolio delta of +3.5%.",
            agent_adjustments=[
                {
                    "agent": "risk_gate",
                    "condition": "bull regime with VIX < 15",
                    "adjustment": "Reduce veto aggressiveness",
                    "reasoning": "VETO win rate is only 40% in low-vol bull markets.",
                },
            ],
        )

        captured_prompt = {}

        async def mock_call_llm(*, model, system_prompt, user_prompt, max_tokens, temperature):
            captured_prompt["user"] = user_prompt
            captured_prompt["system"] = system_prompt
            return {"content": llm_response}

        agent = MetaAnalystAgent.__new__(MetaAnalystAgent)
        agent.name = "meta_analyst"
        agent.model = "claude-opus-4-20250514"

        with patch("src.agents.meta_analyst.call_llm", side_effect=mock_call_llm):
            result = await agent.analyze(_performance_data_with_divergence())

        # Divergence data should appear in user prompt
        assert "divergence" in captured_prompt["user"]
        assert "net_portfolio_delta" in captured_prompt["user"]

        # New fields should be populated
        assert result is not None
        assert result.divergence_assessment is not None
        assert "net positive" in result.divergence_assessment
        assert len(result.agent_adjustments) == 1
        assert result.agent_adjustments[0].agent == "risk_gate"


class TestAnalyzeWithoutDivergenceData:
    @pytest.mark.asyncio
    async def test_backward_compat_defaults_kick_in(self):
        llm_response = _base_llm_response()  # No divergence fields

        async def mock_call_llm(*, model, system_prompt, user_prompt, max_tokens, temperature):
            return {"content": llm_response}

        agent = MetaAnalystAgent.__new__(MetaAnalystAgent)
        agent.name = "meta_analyst"
        agent.model = "claude-opus-4-20250514"

        with patch("src.agents.meta_analyst.call_llm", side_effect=mock_call_llm):
            result = await agent.analyze(_performance_data_without_divergence())

        assert result is not None
        assert result.divergence_assessment is None
        assert result.agent_adjustments == []
        # Existing fields still work
        assert result.total_signals == 30
        assert result.win_rate == 0.60


class TestSparseDataResilience:
    """LLM returns null/malformed fields when data is sparse — should still parse."""

    @pytest.mark.asyncio
    async def test_regime_accuracy_null(self):
        llm_response = _base_llm_response(regime_accuracy=None)

        async def mock_call_llm(*, model, system_prompt, user_prompt, max_tokens, temperature):
            return {"content": llm_response}

        agent = MetaAnalystAgent.__new__(MetaAnalystAgent)
        agent.name = "meta_analyst"
        agent.model = "claude-opus-4-20250514"

        with patch("src.agents.meta_analyst.call_llm", side_effect=mock_call_llm):
            result = await agent.analyze(_performance_data_without_divergence())

        assert result is not None
        assert result.regime_accuracy is None

    @pytest.mark.asyncio
    async def test_malformed_threshold_adjustments_filtered(self):
        """LLM writes prose instead of numeric fields — malformed entries dropped."""
        llm_response = _base_llm_response(
            threshold_adjustments=[
                # Malformed: uses "current"/"suggested" instead of "current_value"/"suggested_value"
                {"parameter": "vix_threshold", "current": 20, "suggested": 25, "reasoning": "N/A"},
                # Valid entry
                {
                    "parameter": "breakout_threshold",
                    "current_value": 50.0,
                    "suggested_value": 55.0,
                    "reasoning": "Win rate improves above 55",
                    "confidence": 65.0,
                    "evidence_sample_size": 30,
                },
            ],
        )

        async def mock_call_llm(*, model, system_prompt, user_prompt, max_tokens, temperature):
            return {"content": llm_response}

        agent = MetaAnalystAgent.__new__(MetaAnalystAgent)
        agent.name = "meta_analyst"
        agent.model = "claude-opus-4-20250514"

        with patch("src.agents.meta_analyst.call_llm", side_effect=mock_call_llm):
            result = await agent.analyze(_performance_data_without_divergence())

        assert result is not None
        # Only the valid entry survives
        assert len(result.threshold_adjustments) == 1
        assert result.threshold_adjustments[0].parameter == "breakout_threshold"

    @pytest.mark.asyncio
    async def test_all_threshold_adjustments_malformed(self):
        """All entries malformed — should parse with empty list."""
        llm_response = _base_llm_response(
            threshold_adjustments=[
                {"parameter": "x", "reasoning": "not enough data"},
            ],
        )

        async def mock_call_llm(*, model, system_prompt, user_prompt, max_tokens, temperature):
            return {"content": llm_response}

        agent = MetaAnalystAgent.__new__(MetaAnalystAgent)
        agent.name = "meta_analyst"
        agent.model = "claude-opus-4-20250514"

        with patch("src.agents.meta_analyst.call_llm", side_effect=mock_call_llm):
            result = await agent.analyze(_performance_data_without_divergence())

        assert result is not None
        assert result.threshold_adjustments == []


class TestDivergenceFetchFailureDoesNotBlockReview:
    @pytest.mark.asyncio
    async def test_fail_closed_behavior(self):
        """Simulate divergence fetch raising — meta-review should still proceed."""
        from src.main import run_weekly_meta_review

        llm_response = _base_llm_response()

        async def mock_call_llm(*, model, system_prompt, user_prompt, max_tokens, temperature):
            return {"content": llm_response}

        mock_perf = AsyncMock(return_value={
            "period_days": 30,
            "total_signals": 25,
            "overall": {"trades": 25, "win_rate": 0.60, "avg_pnl": 1.5},
        })

        mock_div = AsyncMock(side_effect=RuntimeError("DB connection failed"))

        with (
            patch("src.output.performance.get_performance_summary", mock_perf),
            patch("src.output.performance.get_divergence_stats", mock_div),
            patch("src.agents.meta_analyst.call_llm", side_effect=mock_call_llm),
            patch("src.main.get_session") as mock_session,
        ):
            mock_ctx = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

            # Should not raise
            await run_weekly_meta_review()
