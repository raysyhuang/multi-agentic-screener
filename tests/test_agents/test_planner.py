"""Tests for the planner agent."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.planner import (
    PlannerAgent,
    ExecutionPlan,
    PlanStep,
    PlanAction,
    StepStatus,
)


def _make_candidates(n: int = 5) -> list:
    """Create mock candidate objects with ticker attributes."""
    class MockCandidate:
        def __init__(self, ticker):
            self.ticker = ticker
    return [MockCandidate(f"T{i}") for i in range(n)]


def test_default_plan_structure():
    """Default plan should have interpret → debate → risk → verify chain."""
    planner = PlannerAgent()
    candidates = _make_candidates(3)
    plan = planner._default_plan(candidates)

    assert isinstance(plan, ExecutionPlan)
    assert plan.is_default is True
    assert plan.goal == "Default linear pipeline"

    # Should have interpret steps for all 3 tickers
    interpret_steps = [s for s in plan.steps if s.action == PlanAction.INTERPRET.value]
    assert len(interpret_steps) == 3

    # Should have debate steps for up to 5 (or 3 here)
    debate_steps = [s for s in plan.steps if s.action == PlanAction.DEBATE.value]
    assert len(debate_steps) == 3

    # Should have risk steps
    risk_steps = [s for s in plan.steps if s.action == PlanAction.RISK_CHECK.value]
    assert len(risk_steps) == 3

    # Should have verify step
    verify_steps = [s for s in plan.steps if s.action == PlanAction.VERIFY.value]
    assert len(verify_steps) == 1


def test_plan_step_dependencies():
    """Debate should depend on interpret, risk should depend on debate."""
    planner = PlannerAgent()
    candidates = _make_candidates(2)
    plan = planner._default_plan(candidates)

    debate_t0 = next(s for s in plan.steps if s.step_id == "debate_T0")
    assert "interpret_T0" in debate_t0.depends_on

    risk_t0 = next(s for s in plan.steps if s.step_id == "risk_T0")
    assert "debate_T0" in risk_t0.depends_on


def test_get_ready_steps():
    """Only steps with satisfied dependencies should be ready."""
    plan = ExecutionPlan(
        goal="Test",
        steps=[
            PlanStep(step_id="s1", action="INTERPRET", target="T0"),
            PlanStep(step_id="s2", action="DEBATE", target="T0", depends_on=["s1"]),
            PlanStep(step_id="s3", action="RISK_CHECK", target="T0", depends_on=["s2"]),
        ],
    )

    ready = plan.get_ready_steps()
    assert len(ready) == 1
    assert ready[0].step_id == "s1"

    # Complete s1 — s2 should become ready
    plan.mark_completed("s1")
    ready = plan.get_ready_steps()
    assert len(ready) == 1
    assert ready[0].step_id == "s2"

    # Complete s2 — s3 should become ready
    plan.mark_completed("s2")
    ready = plan.get_ready_steps()
    assert len(ready) == 1
    assert ready[0].step_id == "s3"


def test_plan_all_completed():
    plan = ExecutionPlan(
        goal="Test",
        steps=[
            PlanStep(step_id="s1", action="INTERPRET", target="T0"),
            PlanStep(step_id="s2", action="DEBATE", target="T0", depends_on=["s1"]),
        ],
    )

    assert not plan.all_completed

    plan.mark_completed("s1")
    assert not plan.all_completed

    plan.mark_completed("s2")
    assert plan.all_completed


def test_plan_mark_failed():
    plan = ExecutionPlan(
        goal="Test",
        steps=[
            PlanStep(step_id="s1", action="INTERPRET", target="T0"),
            PlanStep(step_id="s2", action="DEBATE", target="T0", depends_on=["s1"]),
        ],
    )

    plan.mark_failed("s1")
    s1 = next(s for s in plan.steps if s.step_id == "s1")
    assert s1.status == StepStatus.FAILED

    # s2 dependencies not met (s1 failed, not completed)
    ready = plan.get_ready_steps()
    assert len(ready) == 0


@pytest.mark.asyncio
async def test_planner_falls_back_on_error():
    """Planner should return default plan if LLM call fails."""
    planner = PlannerAgent()
    candidates = _make_candidates(3)

    with patch(
        "src.agents.planner.call_llm",
        new_callable=AsyncMock,
        side_effect=Exception("LLM API error"),
    ):
        plan = await planner.create_plan(
            candidates, {"regime": "bull"}, "test memory"
        )

    assert plan.is_default is True
    assert len(plan.steps) > 0
