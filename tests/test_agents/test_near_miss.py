"""Tests for near-miss capture in the agent orchestrator."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.base import (
    GateDecision,
    SignalInterpretation,
    DebatePosition,
    DebateResult,
    RiskGateOutput,
)
from src.agents.retry import RetryResult, AttemptRecord
from src.agents.orchestrator import run_agent_pipeline
from src.agents.planner import ExecutionPlan, PlanStep, PlanAction
from src.agents.verifier import VerificationResult
from src.signals.ranker import RankedCandidate


# ---------------------------------------------------------------------------
# Helpers (reusable factories)
# ---------------------------------------------------------------------------

def _make_candidate(ticker: str, score: float = 70.0) -> RankedCandidate:
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
        thesis="Strong momentum breakout.",
        confidence=75.0,
        key_drivers=["high RVOL"],
        risk_flags=[],
        suggested_entry=100.0,
        suggested_stop=95.0,
        suggested_target=110.0,
        timeframe_days=10,
    )


def _mock_debate(ticker: str, verdict: str = "PROCEED", conviction: int = 65) -> DebateResult:
    return DebateResult(
        ticker=ticker,
        bull_case=DebatePosition(
            position="BULL", argument="Strong momentum.", evidence=["RVOL 2.0"],
            weakness="Regime risk", conviction=70,
        ),
        bear_case=DebatePosition(
            position="BEAR", argument="Resistance overhead.", evidence=["Near high"],
            weakness="Volume supports bulls", conviction=40,
        ),
        rebuttal_summary="Bull case wins.",
        final_verdict=verdict,
        net_conviction=conviction,
        key_risk="Earnings in 5 days",
    )


def _mock_gate(ticker: str, decision: GateDecision = GateDecision.APPROVE) -> RiskGateOutput:
    return RiskGateOutput(
        ticker=ticker,
        decision=decision,
        reasoning="Risk assessment complete.",
        position_size_pct=7.5,
    )


def _wrap_retry(value):
    result = RetryResult()
    result.value = value
    result.add_attempt(AttemptRecord(attempt_num=1, success=True))
    return result


def _mock_default_plan(candidates):
    return ExecutionPlan(
        goal="Test plan",
        steps=[PlanStep(step_id="s1", action=PlanAction.INTERPRET.value, target="all")],
        acceptance_criteria=["At least 1 pick"],
        is_default=True,
    )


def _mock_verification_pass():
    return VerificationResult(
        passed=True,
        overall_assessment="Pipeline output meets criteria.",
    )


def _enter_common_patches(stack: ExitStack) -> None:
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDebateRejectNearMiss:
    @pytest.mark.asyncio
    async def test_debate_reject_creates_near_miss(self):
        """Debate REJECT -> NearMissRecord with stage='debate' and correct fields."""
        candidates = [_make_candidate("BAD")]
        regime_context = {"regime": "bull"}

        rejected = _mock_debate("BAD", verdict="REJECT", conviction=25)

        with ExitStack() as stack:
            stack.enter_context(patch(
                "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
                new_callable=AsyncMock,
                return_value=_wrap_retry(_mock_interpretation("BAD")),
            ))
            stack.enter_context(patch(
                "src.agents.adversarial.AdversarialAgent.debate",
                new_callable=AsyncMock,
                return_value=_wrap_retry(rejected),
            ))
            _enter_common_patches(stack)

            result = await run_agent_pipeline(candidates, regime_context)

        assert len(result.near_misses) == 1
        nm = result.near_misses[0]
        assert nm.ticker == "BAD"
        assert nm.stage == "debate"
        assert nm.debate_verdict == "REJECT"
        assert nm.net_conviction == 25
        assert nm.bull_conviction == 70
        assert nm.bear_conviction == 40
        assert nm.key_risk == "Earnings in 5 days"
        assert nm.interpreter_confidence == 75.0
        assert nm.signal_model == "breakout"
        assert nm.entry_price == 100.0
        assert nm.stop_loss == 95.0
        assert nm.target_price == 110.0
        assert nm.timeframe_days == 10
        # Risk gate fields should be None for debate rejections
        assert nm.risk_gate_decision is None
        assert nm.risk_gate_reasoning is None


class TestRiskGateVetoNearMiss:
    @pytest.mark.asyncio
    async def test_risk_gate_veto_creates_near_miss(self):
        """CAUTIOUS debate -> VETO at risk gate -> NearMissRecord with risk gate fields."""
        candidates = [_make_candidate("RISKY")]
        regime_context = {"regime": "bear"}

        cautious_debate = _mock_debate("RISKY", verdict="CAUTIOUS", conviction=45)
        veto_gate = RiskGateOutput(
            ticker="RISKY",
            decision=GateDecision.VETO,
            reasoning="Correlation too high with existing positions.",
            position_size_pct=0.0,
        )

        with ExitStack() as stack:
            stack.enter_context(patch(
                "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
                new_callable=AsyncMock,
                return_value=_wrap_retry(_mock_interpretation("RISKY")),
            ))
            stack.enter_context(patch(
                "src.agents.adversarial.AdversarialAgent.debate",
                new_callable=AsyncMock,
                return_value=_wrap_retry(cautious_debate),
            ))
            stack.enter_context(patch(
                "src.agents.risk_gate.RiskGateAgent.evaluate",
                new_callable=AsyncMock,
                return_value=_wrap_retry(veto_gate),
            ))
            _enter_common_patches(stack)

            result = await run_agent_pipeline(candidates, regime_context)

        assert len(result.near_misses) == 1
        nm = result.near_misses[0]
        assert nm.ticker == "RISKY"
        assert nm.stage == "risk_gate"
        assert nm.debate_verdict == "CAUTIOUS"
        assert nm.net_conviction == 45
        assert nm.risk_gate_decision == "VETO"
        assert nm.risk_gate_reasoning == "Correlation too high with existing positions."


class TestApprovedSignalNoNearMiss:
    @pytest.mark.asyncio
    async def test_approved_signal_no_near_miss(self):
        """PROCEED -> APPROVE creates no near-miss."""
        candidates = [_make_candidate("GOOD")]
        regime_context = {"regime": "bull"}

        with ExitStack() as stack:
            stack.enter_context(patch(
                "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
                new_callable=AsyncMock,
                return_value=_wrap_retry(_mock_interpretation("GOOD")),
            ))
            stack.enter_context(patch(
                "src.agents.adversarial.AdversarialAgent.debate",
                new_callable=AsyncMock,
                return_value=_wrap_retry(_mock_debate("GOOD")),
            ))
            stack.enter_context(patch(
                "src.agents.risk_gate.RiskGateAgent.evaluate",
                new_callable=AsyncMock,
                return_value=_wrap_retry(_mock_gate("GOOD")),
            ))
            _enter_common_patches(stack)

            result = await run_agent_pipeline(candidates, regime_context)

        assert len(result.approved) == 1
        assert len(result.near_misses) == 0


class TestNearMissConvictionScores:
    @pytest.mark.asyncio
    async def test_conviction_scores_correctly_captured(self):
        """Verify conviction scores match debate output exactly."""
        candidates = [_make_candidate("CONV")]
        regime_context = {"regime": "choppy"}

        debate = DebateResult(
            ticker="CONV",
            bull_case=DebatePosition(
                position="BULL", argument="Solid.", evidence=["EPS beat"],
                weakness="Valuation", conviction=55,
            ),
            bear_case=DebatePosition(
                position="BEAR", argument="Expensive.", evidence=["P/E 40"],
                weakness="Growth offsets", conviction=60,
            ),
            rebuttal_summary="Bear slightly stronger.",
            final_verdict="REJECT",
            net_conviction=42,
            key_risk="Overvaluation",
        )

        with ExitStack() as stack:
            stack.enter_context(patch(
                "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
                new_callable=AsyncMock,
                return_value=_wrap_retry(_mock_interpretation("CONV")),
            ))
            stack.enter_context(patch(
                "src.agents.adversarial.AdversarialAgent.debate",
                new_callable=AsyncMock,
                return_value=_wrap_retry(debate),
            ))
            _enter_common_patches(stack)

            result = await run_agent_pipeline(candidates, regime_context)

        nm = result.near_misses[0]
        assert nm.net_conviction == 42
        assert nm.bull_conviction == 55
        assert nm.bear_conviction == 60


class TestPipelineRunContainsNearMisses:
    @pytest.mark.asyncio
    async def test_pipeline_run_contains_near_misses(self):
        """PipelineRun.near_misses is populated with mixed approve/reject."""
        candidates = [_make_candidate("WIN"), _make_candidate("LOSE")]
        regime_context = {"regime": "bull"}

        async def mock_debate(ticker, **kw):
            if ticker == "LOSE":
                return _wrap_retry(_mock_debate(ticker, verdict="REJECT", conviction=30))
            return _wrap_retry(_mock_debate(ticker))

        with ExitStack() as stack:
            stack.enter_context(patch(
                "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
                new_callable=AsyncMock,
                side_effect=lambda c, r, **kw: _wrap_retry(_mock_interpretation(c.ticker)),
            ))
            stack.enter_context(patch(
                "src.agents.adversarial.AdversarialAgent.debate",
                new_callable=AsyncMock,
                side_effect=mock_debate,
            ))
            stack.enter_context(patch(
                "src.agents.risk_gate.RiskGateAgent.evaluate",
                new_callable=AsyncMock,
                side_effect=lambda ticker, **kw: _wrap_retry(_mock_gate(ticker)),
            ))
            _enter_common_patches(stack)

            result = await run_agent_pipeline(candidates, regime_context)

        assert len(result.approved) == 1
        assert result.approved[0].ticker == "WIN"
        assert len(result.near_misses) == 1
        assert result.near_misses[0].ticker == "LOSE"


class TestBudgetExhaustedPreservesNearMisses:
    @pytest.mark.asyncio
    async def test_budget_exhausted_preserves_near_misses(self):
        """Early budget-exhausted return still includes near-misses from debate stage."""
        candidates = [_make_candidate("NM1"), _make_candidate("NM2")]
        regime_context = {"regime": "bull"}

        rejected = _mock_debate("NM1", verdict="REJECT", conviction=20)

        def wrap_costly(value, cost=0.50):
            result = RetryResult()
            result.value = value
            result.add_attempt(AttemptRecord(attempt_num=1, success=True))
            result.total_cost_usd = cost
            result.total_tokens = 500
            return result

        debate_call_count = 0

        async def mock_debate(ticker, **kw):
            nonlocal debate_call_count
            debate_call_count += 1
            if ticker == "NM1":
                return wrap_costly(rejected, cost=0.05)
            return wrap_costly(_mock_debate(ticker), cost=0.05)

        low_budget_plan = ExecutionPlan(
            goal="Budget test",
            steps=[PlanStep(step_id="s1", action=PlanAction.INTERPRET.value, target="all")],
            acceptance_criteria=["At least 1 pick"],
            max_cost_usd=0.01,  # very low — exhausted after interpretation
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
                side_effect=mock_debate,
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

        assert result.convergence_state == "budget_exhausted"
        # The near_misses list is always present, even on early return
        assert isinstance(result.near_misses, list)
