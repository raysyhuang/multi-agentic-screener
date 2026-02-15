"""Tests for agent orchestrator — uses mocked LLM calls."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.base import (
    GateDecision,
    SignalInterpretation,
    DebatePosition,
    DebateResult,
    RiskGateOutput,
)
from src.agents.orchestrator import run_agent_pipeline, PipelineRun
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


@pytest.mark.asyncio
async def test_pipeline_produces_approved_picks():
    candidates = [_make_candidate("AAPL", 80), _make_candidate("MSFT", 75)]
    regime_context = {"regime": "bull", "confidence": 0.8, "vix": 14.0}

    with (
        patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            side_effect=lambda c, r: _mock_interpretation(c.ticker),
        ),
        patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _mock_debate(ticker),
        ),
        patch(
            "src.agents.risk_gate.RiskGateAgent.evaluate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _mock_gate(ticker),
        ),
    ):
        result = await run_agent_pipeline(candidates, regime_context)

    assert isinstance(result, PipelineRun)
    assert len(result.approved) > 0
    assert result.approved[0].ticker in ("AAPL", "MSFT")
    assert result.regime == "bull"


@pytest.mark.asyncio
async def test_pipeline_respects_max_picks():
    candidates = [_make_candidate(f"T{i}", 60 + i) for i in range(10)]
    regime_context = {"regime": "bull", "confidence": 0.7}

    with (
        patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            side_effect=lambda c, r: _mock_interpretation(c.ticker),
        ),
        patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _mock_debate(ticker),
        ),
        patch(
            "src.agents.risk_gate.RiskGateAgent.evaluate",
            new_callable=AsyncMock,
            side_effect=lambda ticker, **kw: _mock_gate(ticker),
        ),
    ):
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

    with (
        patch(
            "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
            new_callable=AsyncMock,
            return_value=_mock_interpretation("BAD"),
        ),
        patch(
            "src.agents.adversarial.AdversarialAgent.debate",
            new_callable=AsyncMock,
            return_value=rejected_debate,
        ),
    ):
        result = await run_agent_pipeline(candidates, regime_context)

    assert len(result.approved) == 0
    assert "BAD" in result.vetoed


@pytest.mark.asyncio
async def test_pipeline_handles_interpreter_failure():
    candidates = [_make_candidate("FAIL", 70)]
    regime_context = {"regime": "bull"}

    with patch(
        "src.agents.signal_interpreter.SignalInterpreterAgent.interpret",
        new_callable=AsyncMock,
        return_value=None,  # interpreter fails
    ):
        result = await run_agent_pipeline(candidates, regime_context)

    assert len(result.approved) == 0
    assert result.interpreted == 0
