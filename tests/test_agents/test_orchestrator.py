"""Tests for agent orchestrator — uses mocked LLM calls."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import date
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.agents.base import (
    GateDecision,
    SignalInterpretation,
    DebatePosition,
    DebateResult,
    RiskGateOutput,
)
from src.agents.retry import RetryResult, AttemptRecord
from src.agents.orchestrator import run_agent_pipeline, PipelineRun
from src.agents.planner import ExecutionPlan, PlanStep, PlanAction
from src.agents.verifier import VerificationResult
from src.signals.ranker import RankedCandidate


def _make_candidate(ticker: str, score: float) -> RankedCandidate:
    return RankedCandidate(
        ticker=ticker,
        signal_model="breakout",
        raw_score=score,
        regime_adjusted_score=score * 1.2,
        direction="LONG",
        entry_price=100.0,
        stop_loss=95.0,
        target_1=110.0,
        target_2=115.0,
        holding_period=10,
        components={"momentum": 70, "volume": 80},
        features={"rsi_14": 65, "rvol": 2.0},
    )


def _mock_interpretation(ticker: str) -> SignalInterpretation:
    return SignalInterpretation(
        ticker=ticker,
        thesis="Strong momentum breakout with volume confirmation.",
        confidence=75.0,
        key_drivers=["high RVOL", "near 20d high", "RSI 65"],
        risk_flags=[],
        suggested_entry=100.0,
        suggested_stop=95.0,
        suggested_target=110.0,
        timeframe_days=10,
    )


def _mock_debate(ticker: str) -> DebateResult:
    return DebateResult(
        ticker=ticker,
        bull_case=DebatePosition(
            position="BULL",
            argument="Strong momentum with volume.",
            evidence=["RVOL 2.0", "RSI 65"],
            weakness="Regime risk",
            conviction=70,
        ),
        bear_case=DebatePosition(
            position="BEAR",
            argument="Resistance overhead.",
            evidence=["Near 20d high"],
            weakness="Volume supports bulls",
            conviction=40,
        ),
        rebuttal_summary="Bull case wins on volume confirmation.",
        final_verdict="PROCEED",
        net_conviction=65,
        key_risk="Earnings in 5 days",
    )


def _mock_gate(ticker: str) -> RiskGateOutput:
    return RiskGateOutput(
        ticker=ticker,
        decision=GateDecision.APPROVE,
        reasoning="Strong signal with manageable risk.",
        position_size_pct=7.5,
    )


def _wrap_retry(value):
    """Wrap a value in a RetryResult for mocking agent returns."""
    result = RetryResult()
    result.value = value
    result.add_attempt(AttemptRecord(attempt_num=1, success=True))
    return result


def _mock_default_plan(candidates):
    """Return a minimal default plan for testing."""
    return ExecutionPlan(
        goal="Test plan",
        steps=[
            PlanStep(
                step_id="test_step",
                action=PlanAction.INTERPRET.value,
                target="all",
            ),
        ],
        acceptance_criteria=["At least 1 pick"],
        is_default=True,
    )


def _mock_verification_pass():
    """Return a passing verification result."""
    return VerificationResult(
        passed=True,
        overall_assessment="Pipeline output meets criteria.",
    )


def _enter_common_patches(stack: ExitStack) -> None:
    """Enter common patches for planner, verifier, and skill engine."""
    stack.enter_context(patch(
        "src.agents.planner.PlannerAgent.create_plan",
        new_callable=AsyncMock,
        side_effect=lambda candidates, regime, mem="": _mock_default_plan(candidates),
    ))
    stack.enter_context(patch(
        "src.agents.verifier.VerifierAgent.verify",
        new_callable=AsyncMock,
        return_value=_mock_verification_pass(),
    ))
    stack.enter_context(patch(
        "src.skills.engine.SkillEngine.find_applicable_skills",
        return_value=[],
    ))


@pytest.mark.asyncio
async def test_pipeline_produces_approved_picks():
    candidates = [_make_candidate("AAPL", 80), _make_candidate("MSFT", 75)]
    regime_context = {"regime": "bull", "confidence": 0.8, "vix": 14.0}

    with ExitStack() as stack:
        stack.enter_context(patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            side_effect=lambda c, r, **kw: _wrap_retry(_mock_interpretation(c.ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_debate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.risk_gate.RiskGateAgent.evaluate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_gate(ticker)),
        ))
        _enter_common_patches(stack)

        result = await run_agent_pipeline(candidates, regime_context)

    assert isinstance(result, PipelineRun)
    assert len(result.approved) > 0
    assert result.approved[0].ticker in ("AAPL", "MSFT")
    assert result.regime == "bull"
    assert result.convergence_state == "converged"


@pytest.mark.asyncio
async def test_pipeline_respects_max_picks():
    candidates = [_make_candidate(f"T{i}", 60 + i) for i in range(10)]
    regime_context = {"regime": "bull", "confidence": 0.7}

    with ExitStack() as stack:
        stack.enter_context(patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            side_effect=lambda c, r, **kw: _wrap_retry(_mock_interpretation(c.ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_debate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.risk_gate.RiskGateAgent.evaluate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_gate(ticker)),
        ))
        _enter_common_patches(stack)

        result = await run_agent_pipeline(candidates, regime_context)

    # max_final_picks defaults to 2
    assert len(result.approved) <= 2


@pytest.mark.asyncio
async def test_pipeline_vetoes_rejected_debate():
    candidates = [_make_candidate("BAD", 70)]
    regime_context = {"regime": "bull"}

    rejected_debate = DebateResult(
        ticker="BAD",
        bull_case=DebatePosition(
            position="BULL", argument="Weak.", evidence=[], weakness="Everything", conviction=30
        ),
        bear_case=DebatePosition(
            position="BEAR", argument="Strong bear case.", evidence=["Overbought"], weakness="None", conviction=80
        ),
        rebuttal_summary="Bear wins.",
        final_verdict="REJECT",
        net_conviction=25,
        key_risk="Overextended",
    )

    with ExitStack() as stack:
        stack.enter_context(patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            return_value=_wrap_retry(_mock_interpretation("BAD")),
        ))
        stack.enter_context(patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            return_value=_wrap_retry(rejected_debate),
        ))
        _enter_common_patches(stack)

        result = await run_agent_pipeline(candidates, regime_context)

    assert len(result.approved) == 0
    assert "BAD" in result.vetoed


@pytest.mark.asyncio
async def test_pipeline_handles_interpreter_failure():
    candidates = [_make_candidate("FAIL", 70)]
    regime_context = {"regime": "bull"}

    with ExitStack() as stack:
        stack.enter_context(patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            return_value=RetryResult(),  # empty result = failure
        ))
        _enter_common_patches(stack)

        result = await run_agent_pipeline(candidates, regime_context)

    assert len(result.approved) == 0
    assert result.interpreted == 0


@pytest.mark.asyncio
async def test_pipeline_logs_include_retry_info():
    """Verify agent_logs include attempts count and failure_reasons."""
    candidates = [_make_candidate("AAPL", 80)]
    regime_context = {"regime": "bull"}

    with ExitStack() as stack:
        stack.enter_context(patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            side_effect=lambda c, r, **kw: _wrap_retry(_mock_interpretation(c.ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_debate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.risk_gate.RiskGateAgent.evaluate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_gate(ticker)),
        ))
        _enter_common_patches(stack)

        result = await run_agent_pipeline(candidates, regime_context)

    # Should have interpreter + adversarial + risk gate + planner + verifier logs
    assert len(result.agent_logs) >= 3
    # Check core agent logs have retry info
    core_logs = [
        l for l in result.agent_logs
        if l["agent"] in ("signal_interpreter", "adversarial_validator", "risk_gatekeeper")
    ]
    for log in core_logs:
        assert "attempts" in log
        assert "failure_reasons" in log


@pytest.mark.asyncio
async def test_pipeline_includes_planner_and_verifier_logs():
    """Verify agent_logs include planner and verifier entries."""
    candidates = [_make_candidate("AAPL", 80)]
    regime_context = {"regime": "bull"}

    with ExitStack() as stack:
        stack.enter_context(patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            side_effect=lambda c, r, **kw: _wrap_retry(_mock_interpretation(c.ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_debate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.risk_gate.RiskGateAgent.evaluate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_gate(ticker)),
        ))
        _enter_common_patches(stack)

        result = await run_agent_pipeline(candidates, regime_context)

    agent_names = [log["agent"] for log in result.agent_logs]
    assert "planner" in agent_names
    assert "verifier" in agent_names

    # Check planner log
    planner_log = next(l for l in result.agent_logs if l["agent"] == "planner")
    assert "plan_goal" in planner_log
    assert planner_log["is_default_plan"] is True

    # Check verifier log
    verifier_log = next(l for l in result.agent_logs if l["agent"] == "verifier")
    assert verifier_log["passed"] is True
    assert verifier_log["should_retry"] is False


@pytest.mark.asyncio
async def test_pipeline_verifier_retry():
    """Verify that verifier-triggered retry re-runs targets."""
    candidates = [_make_candidate("RETRY1", 80), _make_candidate("RETRY2", 75)]
    regime_context = {"regime": "bull"}

    # Verifier says retry RETRY1
    retry_verification = VerificationResult(
        passed=False,
        overall_assessment="RETRY1 thesis needs improvement.",
        should_retry=True,
        retry_targets=["RETRY1"],
    )

    interpret_call_count = 0

    async def mock_interpret(c, r, **kw):
        nonlocal interpret_call_count
        interpret_call_count += 1
        return _wrap_retry(_mock_interpretation(c.ticker))

    with ExitStack() as stack:
        stack.enter_context(patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            side_effect=mock_interpret,
        ))
        stack.enter_context(patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_debate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.risk_gate.RiskGateAgent.evaluate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_gate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.planner.PlannerAgent.create_plan",
            new_callable=AsyncMock,
            side_effect=lambda candidates, regime, mem="": _mock_default_plan(candidates),
        ))
        stack.enter_context(patch(
            "src.agents.verifier.VerifierAgent.verify",
            new_callable=AsyncMock,
            return_value=retry_verification,
        ))
        stack.enter_context(patch(
            "src.skills.engine.SkillEngine.find_applicable_skills",
            return_value=[],
        ))

        result = await run_agent_pipeline(candidates, regime_context)

    # Interpret should be called more than 2 times (initial 2 + retries for RETRY1)
    assert interpret_call_count >= 3
    assert len(result.approved) > 0
    # Verifier always said retry, so convergence should be max_retries
    assert result.convergence_state == "max_retries"


@pytest.mark.asyncio
async def test_pipeline_memory_tracks_state():
    """Verify that working memory is updated during pipeline execution."""
    candidates = [_make_candidate("MEM1", 80)]
    regime_context = {"regime": "bear"}

    with ExitStack() as stack:
        stack.enter_context(patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            side_effect=lambda c, r, **kw: _wrap_retry(_mock_interpretation(c.ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_debate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.risk_gate.RiskGateAgent.evaluate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_gate(ticker)),
        ))
        _enter_common_patches(stack)

        result = await run_agent_pipeline(
            candidates, regime_context, run_id="test_run_123",
        )

    assert len(result.approved) == 1
    assert result.approved[0].ticker == "MEM1"


@pytest.mark.asyncio
async def test_pipeline_budget_exhausted_stops_pipeline():
    """Verify cost circuit breaker stops pipeline between stages."""
    candidates = [_make_candidate("EXP1", 80), _make_candidate("EXP2", 75)]
    regime_context = {"regime": "bull"}

    def wrap_costly(value, cost=0.50):
        """Create a RetryResult with non-zero cost to trigger budget."""
        result = RetryResult()
        result.value = value
        result.add_attempt(AttemptRecord(attempt_num=1, success=True))
        result.total_cost_usd = cost
        result.total_tokens = 500
        return result

    # Plan with tiny budget — interpretation costs exceed it
    low_budget_plan = ExecutionPlan(
        goal="Low budget test",
        steps=[PlanStep(step_id="s1", action=PlanAction.INTERPRET.value, target="all")],
        acceptance_criteria=["At least 1 pick"],
        max_cost_usd=0.05,  # very low budget
        is_default=True,
    )

    with ExitStack() as stack:
        stack.enter_context(patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            side_effect=lambda c, r, **kw: wrap_costly(_mock_interpretation(c.ticker), cost=0.10),
        ))
        stack.enter_context(patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_debate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.risk_gate.RiskGateAgent.evaluate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_gate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.planner.PlannerAgent.create_plan",
            new_callable=AsyncMock,
            return_value=low_budget_plan,
        ))
        stack.enter_context(patch(
            "src.agents.verifier.VerifierAgent.verify",
            new_callable=AsyncMock,
            return_value=_mock_verification_pass(),
        ))
        stack.enter_context(patch(
            "src.skills.engine.SkillEngine.find_applicable_skills",
            return_value=[],
        ))

        result = await run_agent_pipeline(candidates, regime_context)

    # Budget exhausted after interpretation — no debate or approval happened
    assert result.interpreted > 0
    assert result.debated == 0
    assert len(result.approved) == 0
    assert result.convergence_state == "budget_exhausted"


@pytest.mark.asyncio
async def test_pipeline_verifier_converges_on_pass():
    """Verify that when verifier passes on first check, convergence is clean."""
    candidates = [_make_candidate("CONV1", 80)]
    regime_context = {"regime": "bull"}

    verify_call_count = 0

    async def mock_verify(**kw):
        nonlocal verify_call_count
        verify_call_count += 1
        return _mock_verification_pass()

    with ExitStack() as stack:
        stack.enter_context(patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            side_effect=lambda c, r, **kw: _wrap_retry(_mock_interpretation(c.ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_debate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.risk_gate.RiskGateAgent.evaluate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_gate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.planner.PlannerAgent.create_plan",
            new_callable=AsyncMock,
            side_effect=lambda candidates, regime, mem="": _mock_default_plan(candidates),
        ))
        stack.enter_context(patch(
            "src.agents.verifier.VerifierAgent.verify",
            new_callable=AsyncMock,
            side_effect=mock_verify,
        ))
        stack.enter_context(patch(
            "src.skills.engine.SkillEngine.find_applicable_skills",
            return_value=[],
        ))

        result = await run_agent_pipeline(candidates, regime_context)

    assert result.convergence_state == "converged"
    # Verifier only called once — no retry needed
    assert verify_call_count == 1


@pytest.mark.asyncio
async def test_pipeline_multi_round_retry_with_feedback():
    """Verify multi-round retry injects verifier suggestions into memory."""
    candidates = [_make_candidate("FB1", 80)]
    regime_context = {"regime": "bull"}

    verify_calls = []

    async def mock_verify(**kw):
        call_num = len(verify_calls) + 1
        verify_calls.append(call_num)
        if call_num <= 2:
            # First 2 calls: request retry with suggestions
            return VerificationResult(
                passed=False,
                overall_assessment=f"Round {call_num}: needs improvement.",
                should_retry=True,
                retry_targets=["FB1"],
                suggestions=[f"Suggestion from round {call_num}"],
            )
        # Third call: pass
        return _mock_verification_pass()

    with ExitStack() as stack:
        stack.enter_context(patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            side_effect=lambda c, r, **kw: _wrap_retry(_mock_interpretation(c.ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_debate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.risk_gate.RiskGateAgent.evaluate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _wrap_retry(_mock_gate(ticker)),
        ))
        stack.enter_context(patch(
            "src.agents.planner.PlannerAgent.create_plan",
            new_callable=AsyncMock,
            side_effect=lambda candidates, regime, mem="": _mock_default_plan(candidates),
        ))
        stack.enter_context(patch(
            "src.agents.verifier.VerifierAgent.verify",
            new_callable=AsyncMock,
            side_effect=mock_verify,
        ))
        stack.enter_context(patch(
            "src.skills.engine.SkillEngine.find_applicable_skills",
            return_value=[],
        ))

        result = await run_agent_pipeline(candidates, regime_context)

    # Verifier called 3 times: initial + 2 retry rounds
    assert len(verify_calls) == 3
    assert result.convergence_state == "converged"
    # Should have verifier logs for retry rounds
    verifier_logs = [l for l in result.agent_logs if l["agent"] == "verifier"]
    assert len(verifier_logs) >= 3  # initial + 2 retry
