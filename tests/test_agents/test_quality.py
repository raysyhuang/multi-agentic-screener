"""Tests for output quality checkers."""

from src.agents.base import (
    SignalInterpretation,
    DebateResult,
    DebatePosition,
    RiskGateOutput,
    GateDecision,
)
from src.agents.quality import (
    check_interpretation_quality,
    check_debate_quality,
    check_risk_gate_quality,
)


def test_valid_interpretation_passes():
    interp = SignalInterpretation(
        ticker="AAPL",
        thesis="Strong breakout with volume confirmation and momentum alignment across timeframes.",
        confidence=75,
        key_drivers=["high RVOL", "near 20d high", "RSI 65"],
        risk_flags=[],
        suggested_entry=150.0,
        suggested_stop=145.0,
        suggested_target=160.0,
        timeframe_days=10,
    )
    result = check_interpretation_quality(interp)
    assert result.passed
    assert len(result.issues) == 0


def test_interpretation_short_thesis_fails():
    interp = SignalInterpretation(
        ticker="AAPL",
        thesis="Buy it.",
        confidence=75,
        key_drivers=["RVOL", "RSI"],
        risk_flags=[],
        suggested_entry=150.0,
        suggested_stop=145.0,
        suggested_target=160.0,
        timeframe_days=10,
    )
    result = check_interpretation_quality(interp)
    assert not result.passed
    assert any("too short" in i.lower() for i in result.issues)


def test_interpretation_too_few_drivers_fails():
    interp = SignalInterpretation(
        ticker="AAPL",
        thesis="Strong breakout with volume confirmation and momentum alignment across timeframes.",
        confidence=75,
        key_drivers=["only_one"],
        risk_flags=[],
        suggested_entry=150.0,
        suggested_stop=145.0,
        suggested_target=160.0,
        timeframe_days=10,
    )
    result = check_interpretation_quality(interp)
    assert not result.passed
    assert any("key_drivers" in i for i in result.issues)


def test_interpretation_bad_trade_structure_fails():
    interp = SignalInterpretation(
        ticker="AAPL",
        thesis="Strong breakout with volume confirmation and momentum alignment across timeframes.",
        confidence=75,
        key_drivers=["RVOL", "RSI", "momentum"],
        risk_flags=[],
        suggested_entry=150.0,
        suggested_stop=155.0,  # stop > entry
        suggested_target=160.0,
        timeframe_days=10,
    )
    result = check_interpretation_quality(interp)
    assert not result.passed
    assert any("stop" in i.lower() for i in result.issues)


def test_interpretation_target_below_entry_fails():
    interp = SignalInterpretation(
        ticker="AAPL",
        thesis="Strong breakout with volume confirmation and momentum alignment across timeframes.",
        confidence=75,
        key_drivers=["RVOL", "RSI", "momentum"],
        risk_flags=[],
        suggested_entry=150.0,
        suggested_stop=145.0,
        suggested_target=140.0,  # target < entry
        timeframe_days=10,
    )
    result = check_interpretation_quality(interp)
    assert not result.passed
    assert any("target" in i.lower() for i in result.issues)


def test_valid_debate_passes():
    debate = DebateResult(
        ticker="AAPL",
        bull_case=DebatePosition(
            position="BULL", argument="Strong", evidence=["RSI"],
            weakness="Risk", conviction=70,
        ),
        bear_case=DebatePosition(
            position="BEAR", argument="Weak", evidence=["Overbought"],
            weakness="Missing catalyst", conviction=40,
        ),
        rebuttal_summary="Bull case stronger due to volume confirmation and trend alignment.",
        final_verdict="PROCEED",
        net_conviction=65,
        key_risk="Earnings in 3 days",
    )
    result = check_debate_quality(debate)
    assert result.passed


def test_debate_empty_rebuttal_fails():
    debate = DebateResult(
        ticker="AAPL",
        bull_case=DebatePosition(
            position="BULL", argument="Strong", evidence=["RSI"],
            weakness="Risk", conviction=70,
        ),
        bear_case=DebatePosition(
            position="BEAR", argument="Weak", evidence=["Overbought"],
            weakness="Missing catalyst", conviction=40,
        ),
        rebuttal_summary="",
        final_verdict="PROCEED",
        net_conviction=65,
        key_risk="Earnings",
    )
    result = check_debate_quality(debate)
    assert not result.passed
    assert any("rebuttal" in i.lower() for i in result.issues)


def test_debate_invalid_verdict_fails():
    debate = DebateResult(
        ticker="AAPL",
        bull_case=DebatePosition(
            position="BULL", argument="Strong", evidence=["RSI"],
            weakness="Risk", conviction=70,
        ),
        bear_case=DebatePosition(
            position="BEAR", argument="Weak", evidence=["Overbought"],
            weakness="Missing catalyst", conviction=40,
        ),
        rebuttal_summary="A thorough analysis of the competing cases.",
        final_verdict="MAYBE",
        net_conviction=65,
        key_risk="Earnings",
    )
    result = check_debate_quality(debate)
    assert not result.passed
    assert any("verdict" in i.lower() for i in result.issues)


def test_debate_zero_conviction_fails():
    debate = DebateResult(
        ticker="AAPL",
        bull_case=DebatePosition(
            position="BULL", argument="Strong", evidence=["RSI"],
            weakness="Risk", conviction=70,
        ),
        bear_case=DebatePosition(
            position="BEAR", argument="Weak", evidence=["Overbought"],
            weakness="Missing catalyst", conviction=40,
        ),
        rebuttal_summary="A thorough analysis of the competing cases.",
        final_verdict="PROCEED",
        net_conviction=0,
        key_risk="Earnings",
    )
    result = check_debate_quality(debate)
    assert not result.passed
    assert any("conviction" in i.lower() for i in result.issues)


def test_valid_risk_gate_passes():
    gate = RiskGateOutput(
        ticker="AAPL",
        decision=GateDecision.APPROVE,
        reasoning="Strong signal with manageable risk profile and good risk/reward ratio.",
        position_size_pct=7.5,
    )
    result = check_risk_gate_quality(gate)
    assert result.passed


def test_risk_gate_empty_reasoning_fails():
    gate = RiskGateOutput(
        ticker="AAPL",
        decision=GateDecision.APPROVE,
        reasoning="",
        position_size_pct=7.5,
    )
    result = check_risk_gate_quality(gate)
    assert not result.passed
    assert any("reasoning" in i.lower() for i in result.issues)


def test_quality_check_summary():
    from src.agents.quality import QualityCheck

    passed = QualityCheck(passed=True, issues=[])
    assert "passed" in passed.summary.lower()

    failed = QualityCheck(passed=False, issues=["too short", "missing data"])
    assert "too short" in failed.summary
    assert "missing data" in failed.summary
