"""Tests for agent base schemas (Pydantic models)."""

import pytest
from pydantic import ValidationError

from src.agents.base import (
    AgentAdjustment,
    SignalInterpretation,
    DebatePosition,
    DebateResult,
    RiskGateOutput,
    GateDecision,
    MetaAnalysis,
)


def test_signal_interpretation_valid():
    si = SignalInterpretation(
        ticker="AAPL",
        thesis="Strong breakout with volume confirmation and momentum alignment.",
        confidence=75,
        key_drivers=["high RVOL", "near 20d high", "RSI 65"],
        risk_flags=[],
        suggested_entry=150.0,
        suggested_stop=145.0,
        suggested_target=160.0,
        timeframe_days=10,
    )
    assert si.confidence == 75
    assert si.ticker == "AAPL"


def test_signal_interpretation_confidence_bounds():
    with pytest.raises(ValidationError):
        SignalInterpretation(
            ticker="BAD", thesis="x", confidence=150,
            key_drivers=[], suggested_entry=100,
            suggested_stop=95, suggested_target=110, timeframe_days=5,
        )


def test_debate_result_valid():
    bull = DebatePosition(
        position="BULL", argument="Strong momentum",
        evidence=["RSI 65", "RVOL 2.5"], weakness="Regime risk",
        conviction=70,
    )
    bear = DebatePosition(
        position="BEAR", argument="Overbought conditions",
        evidence=["Near resistance"], weakness="Missing catalyst",
        conviction=40,
    )
    result = DebateResult(
        ticker="AAPL", bull_case=bull, bear_case=bear,
        rebuttal_summary="Bull case stronger due to volume",
        final_verdict="PROCEED", net_conviction=65,
        key_risk="Earnings in 3 days",
    )
    assert result.final_verdict == "PROCEED"


def test_risk_gate_output_valid():
    rgo = RiskGateOutput(
        ticker="AAPL", decision=GateDecision.APPROVE,
        reasoning="Strong signal with manageable risk.",
        position_size_pct=7.5,
    )
    assert rgo.decision == GateDecision.APPROVE
    assert rgo.position_size_pct == 7.5


def test_risk_gate_veto():
    rgo = RiskGateOutput(
        ticker="BAD", decision=GateDecision.VETO,
        reasoning="Regime mismatch and high correlation.",
        position_size_pct=0,
    )
    assert rgo.decision == GateDecision.VETO


def test_meta_analysis_valid():
    ma = MetaAnalysis(
        analysis_period="2025-03-01 to 2025-03-31",
        total_signals=25,
        win_rate=0.64,
        avg_pnl_pct=1.8,
        best_model="breakout",
        worst_model="catalyst",
        regime_accuracy=0.75,
        biases_detected=["Overconfident in choppy regime"],
        summary="System performed well overall.",
    )
    assert ma.total_signals == 25


def test_meta_analysis_with_divergence_fields():
    """New divergence fields parse correctly."""
    ma = MetaAnalysis(
        analysis_period="2025-03-01 to 2025-03-31",
        total_signals=25,
        win_rate=0.64,
        avg_pnl_pct=1.8,
        best_model="breakout",
        worst_model="catalyst",
        regime_accuracy=0.75,
        biases_detected=[],
        summary="Summary.",
        divergence_assessment="LLM overlay is net positive.",
        agent_adjustments=[
            AgentAdjustment(
                agent="risk_gate",
                condition="bull regime",
                adjustment="reduce veto aggressiveness",
                reasoning="VETO win rate below 40%",
            ),
        ],
    )
    assert ma.divergence_assessment == "LLM overlay is net positive."
    assert len(ma.agent_adjustments) == 1
    assert ma.agent_adjustments[0].agent == "risk_gate"


def test_meta_analysis_backward_compatible():
    """Old data without new fields still deserializes."""
    old_data = {
        "analysis_period": "2025-03-01 to 2025-03-31",
        "total_signals": 25,
        "win_rate": 0.64,
        "avg_pnl_pct": 1.8,
        "best_model": "breakout",
        "worst_model": "catalyst",
        "regime_accuracy": 0.75,
        "biases_detected": [],
        "summary": "Summary.",
    }
    ma = MetaAnalysis(**old_data)
    assert ma.divergence_assessment is None
    assert ma.agent_adjustments == []


def test_meta_analysis_regime_accuracy_optional():
    """regime_accuracy can be None when data is insufficient."""
    ma = MetaAnalysis(
        analysis_period="2025-03-01 to 2025-03-31",
        total_signals=2,
        win_rate=0.50,
        avg_pnl_pct=0.5,
        best_model="breakout",
        worst_model="catalyst",
        regime_accuracy=None,
        biases_detected=[],
        summary="Insufficient data for regime accuracy.",
    )
    assert ma.regime_accuracy is None


def test_meta_analysis_regime_accuracy_defaults_to_none():
    """regime_accuracy defaults to None when omitted."""
    ma = MetaAnalysis(
        analysis_period="2025-03-01 to 2025-03-31",
        total_signals=2,
        win_rate=0.50,
        avg_pnl_pct=0.5,
        best_model="breakout",
        worst_model="catalyst",
        biases_detected=[],
        summary="Minimal data.",
    )
    assert ma.regime_accuracy is None


def test_agent_adjustment_valid():
    """AgentAdjustment model validates correctly."""
    adj = AgentAdjustment(
        agent="debate",
        condition="choppy regime with VIX > 25",
        adjustment="increase bear weight by 10%",
        reasoning="Debate bull bias detected in volatile markets",
    )
    assert adj.agent == "debate"
    assert "choppy" in adj.condition
