"""Tests for the verifier agent."""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.agents.verifier import VerifierAgent, VerificationResult, CriterionResult


@pytest.mark.asyncio
async def test_verifier_passes_valid_output():
    """Verifier should pass when LLM says output is good."""
    verifier = VerifierAgent()

    mock_content = {
        "passed": True,
        "criteria_results": [
            {"criterion": "thesis quality", "passed": True, "note": "Good"},
        ],
        "overall_assessment": "Pipeline output meets all criteria.",
        "suggestions": [],
        "should_retry": False,
        "retry_targets": [],
    }

    with patch(
        "src.agents.verifier.call_llm",
        new_callable=AsyncMock,
        return_value={
            "content": mock_content,
            "raw": "{}",
            "tokens_in": 100,
            "tokens_out": 50,
            "latency_ms": 500,
            "model": "gpt-5.2-mini",
            "cost_usd": 0.001,
        },
    ):
        result = await verifier.verify(
            pipeline_output={"approved": [{"ticker": "AAPL"}]},
            acceptance_criteria=["At least 1 pick"],
            regime_context={"regime": "bull"},
        )

    assert result.passed is True
    assert len(result.criteria_results) == 1
    assert not result.should_retry


@pytest.mark.asyncio
async def test_verifier_flags_invalid_output():
    """Verifier should flag issues and suggest retries."""
    verifier = VerifierAgent()

    mock_content = {
        "passed": False,
        "criteria_results": [
            {"criterion": "thesis quality", "passed": False, "note": "Thesis too vague"},
        ],
        "overall_assessment": "Output quality is below standards.",
        "suggestions": ["Re-run signal interpreter with more context"],
        "should_retry": True,
        "retry_targets": ["AAPL"],
    }

    with patch(
        "src.agents.verifier.call_llm",
        new_callable=AsyncMock,
        return_value={
            "content": mock_content,
            "raw": "{}",
            "tokens_in": 100,
            "tokens_out": 50,
            "latency_ms": 500,
            "model": "gpt-5.2-mini",
            "cost_usd": 0.001,
        },
    ):
        result = await verifier.verify(
            pipeline_output={"approved": [{"ticker": "AAPL"}]},
            acceptance_criteria=["Quality threshold"],
            regime_context={"regime": "bull"},
        )

    assert result.passed is False
    assert result.should_retry is True
    assert "AAPL" in result.retry_targets


@pytest.mark.asyncio
async def test_verifier_fail_open_on_error():
    """Verifier should pass (fail-open) when LLM call fails."""
    verifier = VerifierAgent()

    with patch(
        "src.agents.verifier.call_llm",
        new_callable=AsyncMock,
        side_effect=Exception("API connection failed"),
    ):
        result = await verifier.verify(
            pipeline_output={"approved": []},
            acceptance_criteria=[],
            regime_context={"regime": "bull"},
        )

    # Fail-open: verifier failure should NOT block picks
    assert result.passed is True
    assert "fail-open" in result.overall_assessment.lower() or "error" in result.overall_assessment.lower()


@pytest.mark.asyncio
async def test_verifier_fail_open_on_non_json():
    """Verifier should pass when LLM returns non-JSON."""
    verifier = VerifierAgent()

    with patch(
        "src.agents.verifier.call_llm",
        new_callable=AsyncMock,
        return_value={
            "content": "I could not analyze the output properly.",
            "raw": "I could not analyze the output properly.",
            "tokens_in": 50,
            "tokens_out": 20,
            "latency_ms": 200,
            "model": "gpt-5.2-mini",
            "cost_usd": 0.0005,
        },
    ):
        result = await verifier.verify(
            pipeline_output={"approved": []},
            acceptance_criteria=[],
            regime_context={"regime": "bull"},
        )

    assert result.passed is True


def test_verification_result_defaults():
    result = VerificationResult(passed=True)
    assert result.passed
    assert result.criteria_results == []
    assert result.suggestions == []
    assert not result.should_retry
    assert result.retry_targets == []


def test_criterion_result():
    cr = CriterionResult(
        criterion="Position sizing",
        passed=True,
        note="Size within limits",
    )
    assert cr.passed
    assert cr.criterion == "Position sizing"
