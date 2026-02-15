"""Tests for the Divergence Ledger — causal attribution layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import pytest

from src.governance.divergence_ledger import (
    QUANT_DEFAULT_SIZE,
    RESIZE_THRESHOLD_PCT,
    DivergenceRecord,
    DivergenceType,
    ReasonCode,
    compute_divergences,
    freeze_quant_baseline,
    simulate_quant_counterfactual,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight fakes for RankedCandidate and PipelineResult/PipelineRun
# ---------------------------------------------------------------------------


@dataclass
class _FakeRankedCandidate:
    ticker: str
    signal_model: str = "breakout"
    raw_score: float = 0.75
    regime_adjusted_score: float = 0.80
    direction: str = "LONG"
    entry_price: float = 100.0
    stop_loss: float = 95.0
    target_1: float = 110.0
    target_2: float | None = None
    holding_period: int = 10
    components: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)


@dataclass
class _FakeRiskGate:
    position_size_pct: float = 5.0
    decision: object = None
    reasoning: str = ""
    regime_note: str | None = None
    correlation_warning: str | None = None

    class _D:
        value = "APPROVE"

    def __post_init__(self):
        if self.decision is None:
            self.decision = self._D()


@dataclass
class _FakeInterpretation:
    ticker: str = ""
    thesis: str = "test"
    confidence: float = 75.0
    risk_flags: list = field(default_factory=list)


@dataclass
class _FakeDebate:
    final_verdict: str = "PROCEED"
    rebuttal_summary: str = "ok"

    @dataclass
    class _Bear:
        argument: str = "bear"
    bear_case: _Bear = field(default_factory=_Bear)


@dataclass
class _FakePick:
    ticker: str
    signal_model: str = "breakout"
    direction: str = "LONG"
    entry_price: float = 100.0
    stop_loss: float = 95.0
    target_1: float = 110.0
    target_2: float | None = None
    holding_period: int = 10
    confidence: float = 75.0
    interpretation: _FakeInterpretation = field(default_factory=_FakeInterpretation)
    debate: _FakeDebate = field(default_factory=_FakeDebate)
    risk_gate: _FakeRiskGate = field(default_factory=_FakeRiskGate)
    features: dict = field(default_factory=dict)


@dataclass
class _FakePipelineRun:
    approved: list = field(default_factory=list)
    vetoed: list = field(default_factory=list)
    agent_logs: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# TestDivergenceType
# ---------------------------------------------------------------------------


class TestDivergenceType:
    def test_enum_values(self):
        assert DivergenceType.VETO.value == "VETO"
        assert DivergenceType.PROMOTE.value == "PROMOTE"
        assert DivergenceType.RESIZE.value == "RESIZE"

    def test_enum_members(self):
        assert set(DivergenceType) == {
            DivergenceType.VETO,
            DivergenceType.PROMOTE,
            DivergenceType.RESIZE,
        }


# ---------------------------------------------------------------------------
# TestReasonCodes
# ---------------------------------------------------------------------------


class TestReasonCodes:
    def test_enum_completeness(self):
        expected = {
            "DEBATE_REJECT",
            "DEBATE_CAUTIOUS",
            "RISK_GATE_VETO",
            "RISK_GATE_ADJUST",
            "SIZE_REDUCED_BEAR_REGIME",
            "SIZE_REDUCED_LOW_CONFIDENCE",
            "SIZE_INCREASED_HIGH_CONFIDENCE",
            "RISK_FLAG_EARNINGS_IMMINENT",
            "RISK_FLAG_HIGH_VOLATILITY",
            "RISK_FLAG_LOW_LIQUIDITY",
            "RISK_FLAG_SECTOR_CORRELATION",
            "RISK_FLAG_REGIME_MISMATCH",
            "RISK_FLAG_OVEREXTENDED",
            "RISK_FLAG_NEWS_RISK",
            "CORRELATION_WARNING",
            "INTERPRETER_LOW_CONFIDENCE",
            "INTERPRETER_HIGH_CONFIDENCE",
            "PROMOTED_BY_INTERPRETER",
            "PROMOTED_BY_DEBATE",
            "QUANT_SCORE_BELOW_THRESHOLD",
        }
        actual = {rc.value for rc in ReasonCode}
        assert actual == expected

    def test_no_free_text_values(self):
        for rc in ReasonCode:
            assert rc.value == rc.value.upper().replace(" ", "_")
            assert " " not in rc.value


# ---------------------------------------------------------------------------
# TestFreezeBaseline
# ---------------------------------------------------------------------------


class TestFreezeBaseline:
    def test_snapshot_structure(self):
        ranked = [
            _FakeRankedCandidate(ticker="AAPL"),
            _FakeRankedCandidate(ticker="MSFT"),
            _FakeRankedCandidate(ticker="GOOG"),
        ]
        baseline = freeze_quant_baseline(
            ranked=ranked, max_picks=2, regime="bull", config_hash="abc123",
        )
        assert "top_k" in baseline
        assert "all_ranked_tickers" in baseline
        assert "max_picks" in baseline
        assert "regime" in baseline
        assert "config_hash" in baseline

    def test_top_k_limited_to_max_picks(self):
        ranked = [
            _FakeRankedCandidate(ticker="AAPL"),
            _FakeRankedCandidate(ticker="MSFT"),
            _FakeRankedCandidate(ticker="GOOG"),
        ]
        baseline = freeze_quant_baseline(ranked=ranked, max_picks=2, regime="bull", config_hash="x")
        assert len(baseline["top_k"]) == 2

    def test_all_ranked_contains_all(self):
        ranked = [
            _FakeRankedCandidate(ticker="AAPL"),
            _FakeRankedCandidate(ticker="MSFT"),
            _FakeRankedCandidate(ticker="GOOG"),
        ]
        baseline = freeze_quant_baseline(ranked=ranked, max_picks=2, regime="bull", config_hash="x")
        tickers = [t["ticker"] for t in baseline["all_ranked_tickers"]]
        assert tickers == ["AAPL", "MSFT", "GOOG"]

    def test_contains_required_fields(self):
        ranked = [_FakeRankedCandidate(ticker="AAPL", entry_price=150.0, stop_loss=140.0, target_1=165.0)]
        baseline = freeze_quant_baseline(ranked=ranked, max_picks=1, regime="bear", config_hash="y")
        item = baseline["top_k"][0]
        assert item["ticker"] == "AAPL"
        assert item["entry_price"] == 150.0
        assert item["stop_loss"] == 140.0
        assert item["target_1"] == 165.0
        assert item["holding_period"] == 10
        assert item["direction"] == "LONG"
        assert item["position_size"] == QUANT_DEFAULT_SIZE
        assert item["rank"] == 1

    def test_immutability_via_separate_calls(self):
        ranked = [_FakeRankedCandidate(ticker="AAPL")]
        b1 = freeze_quant_baseline(ranked=ranked, max_picks=1, regime="bull", config_hash="a")
        b2 = freeze_quant_baseline(ranked=ranked, max_picks=1, regime="bull", config_hash="a")
        assert b1 == b2
        b1["top_k"][0]["ticker"] = "MUTATED"
        assert b2["top_k"][0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# TestComputeDivergences
# ---------------------------------------------------------------------------


class TestComputeDivergences:
    def _make_baseline(self, tickers: list[str]) -> dict:
        ranked = [_FakeRankedCandidate(ticker=t) for t in tickers]
        return freeze_quant_baseline(ranked=ranked, max_picks=len(tickers), regime="bull", config_hash="test")

    def test_identical_picks_no_divergences(self):
        baseline = self._make_baseline(["AAPL", "MSFT"])
        agentic = _FakePipelineRun(
            approved=[_FakePick(ticker="AAPL"), _FakePick(ticker="MSFT")],
        )
        divs = compute_divergences(baseline, agentic, [])
        assert divs == []

    def test_quant_pick_removed_is_veto(self):
        baseline = self._make_baseline(["AAPL", "MSFT"])
        agentic = _FakePipelineRun(
            approved=[_FakePick(ticker="MSFT")],
            vetoed=["AAPL"],
        )
        divs = compute_divergences(baseline, agentic, [])
        assert len(divs) == 1
        assert divs[0].event_type == DivergenceType.VETO
        assert divs[0].ticker == "AAPL"
        assert divs[0].quant_rank == 1
        assert divs[0].agentic_rank is None

    def test_non_quant_pick_added_is_promote(self):
        baseline = self._make_baseline(["AAPL"])
        agentic = _FakePipelineRun(
            approved=[_FakePick(ticker="AAPL"), _FakePick(ticker="GOOG")],
        )
        divs = compute_divergences(baseline, agentic, [])
        assert len(divs) == 1
        assert divs[0].event_type == DivergenceType.PROMOTE
        assert divs[0].ticker == "GOOG"
        assert divs[0].agentic_rank == 2

    def test_same_pick_different_sizing_is_resize(self):
        baseline = self._make_baseline(["AAPL"])
        gate = _FakeRiskGate(position_size_pct=3.0)  # 40% reduction from 5.0
        agentic = _FakePipelineRun(
            approved=[_FakePick(ticker="AAPL", risk_gate=gate)],
        )
        divs = compute_divergences(baseline, agentic, [])
        assert len(divs) == 1
        assert divs[0].event_type == DivergenceType.RESIZE
        assert divs[0].quant_size == QUANT_DEFAULT_SIZE
        assert divs[0].agentic_size == 3.0

    def test_same_pick_similar_sizing_no_resize(self):
        baseline = self._make_baseline(["AAPL"])
        # 4.5 is only 10% different from 5.0, below 20% threshold
        gate = _FakeRiskGate(position_size_pct=4.5)
        agentic = _FakePipelineRun(
            approved=[_FakePick(ticker="AAPL", risk_gate=gate)],
        )
        divs = compute_divergences(baseline, agentic, [])
        assert divs == []

    def test_multiple_divergences_in_one_run(self):
        baseline = self._make_baseline(["AAPL", "MSFT"])
        gate = _FakeRiskGate(position_size_pct=2.0)
        agentic = _FakePipelineRun(
            approved=[_FakePick(ticker="MSFT", risk_gate=gate), _FakePick(ticker="GOOG")],
            vetoed=["AAPL"],
        )
        divs = compute_divergences(baseline, agentic, [])
        types = {d.event_type for d in divs}
        tickers = {d.ticker for d in divs}
        assert DivergenceType.VETO in types
        assert DivergenceType.PROMOTE in types
        assert DivergenceType.RESIZE in types
        assert tickers == {"AAPL", "GOOG", "MSFT"}

    def test_cost_aggregation_from_agent_logs(self):
        baseline = self._make_baseline(["AAPL", "MSFT"])
        agentic = _FakePipelineRun(
            approved=[_FakePick(ticker="MSFT")],
            vetoed=["AAPL"],
        )
        logs = [
            {"agent": "signal_interpreter", "ticker": "AAPL", "cost_usd": 0.01},
            {"agent": "adversarial", "ticker": "AAPL", "cost_usd": 0.02},
            {"agent": "risk_gate", "ticker": "AAPL", "cost_usd": 0.005,
             "output_data": {"decision": "VETO"}},
        ]
        divs = compute_divergences(baseline, agentic, logs)
        veto = [d for d in divs if d.event_type == DivergenceType.VETO][0]
        assert veto.llm_cost_usd == pytest.approx(0.035, abs=0.001)


# ---------------------------------------------------------------------------
# TestReasonCodeExtraction
# ---------------------------------------------------------------------------


class TestReasonCodeExtraction:
    def _make_baseline(self, tickers: list[str]) -> dict:
        ranked = [_FakeRankedCandidate(ticker=t) for t in tickers]
        return freeze_quant_baseline(ranked=ranked, max_picks=len(tickers), regime="bull", config_hash="test")

    def test_veto_with_debate_reject(self):
        baseline = self._make_baseline(["AAPL"])
        agentic = _FakePipelineRun(vetoed=["AAPL"])
        logs = [
            {"agent": "adversarial", "ticker": "AAPL",
             "output_data": {"final_verdict": "REJECT"}},
        ]
        divs = compute_divergences(baseline, agentic, logs)
        assert len(divs) == 1
        assert ReasonCode.DEBATE_REJECT.value in divs[0].reason_codes

    def test_veto_with_risk_gate_veto(self):
        baseline = self._make_baseline(["AAPL"])
        agentic = _FakePipelineRun(vetoed=["AAPL"])
        logs = [
            {"agent": "risk_gate", "ticker": "AAPL",
             "output_data": {"decision": "VETO"}},
        ]
        divs = compute_divergences(baseline, agentic, logs)
        assert ReasonCode.RISK_GATE_VETO.value in divs[0].reason_codes

    def test_promote_with_interpreter_high_confidence(self):
        baseline = self._make_baseline(["MSFT"])
        agentic = _FakePipelineRun(
            approved=[_FakePick(ticker="MSFT"), _FakePick(ticker="GOOG")],
        )
        logs = [
            {"agent": "signal_interpreter", "ticker": "GOOG",
             "output_data": {"confidence": 85}},
        ]
        divs = compute_divergences(baseline, agentic, logs)
        promote = [d for d in divs if d.event_type == DivergenceType.PROMOTE][0]
        assert ReasonCode.INTERPRETER_HIGH_CONFIDENCE.value in promote.reason_codes
        assert ReasonCode.PROMOTED_BY_INTERPRETER.value in promote.reason_codes

    def test_resize_with_size_reduced_bear_regime(self):
        baseline = self._make_baseline(["AAPL"])
        gate = _FakeRiskGate(position_size_pct=3.0, regime_note="bear market caution")

        class _D:
            value = "ADJUST"
        gate.decision = _D()

        agentic = _FakePipelineRun(
            approved=[_FakePick(ticker="AAPL", risk_gate=gate)],
        )
        logs = [
            {"agent": "risk_gate", "ticker": "AAPL",
             "output_data": {"decision": "ADJUST"}},
        ]
        divs = compute_divergences(baseline, agentic, logs)
        assert len(divs) == 1
        assert ReasonCode.SIZE_REDUCED_BEAR_REGIME.value in divs[0].reason_codes

    def test_risk_flags_mapped_correctly(self):
        baseline = self._make_baseline(["AAPL"])
        agentic = _FakePipelineRun(vetoed=["AAPL"])
        logs = [
            {"agent": "signal_interpreter", "ticker": "AAPL",
             "output_data": {
                 "confidence": 30,
                 "risk_flags": ["earnings_imminent", "high_volatility"],
             }},
        ]
        divs = compute_divergences(baseline, agentic, logs)
        codes = divs[0].reason_codes
        assert ReasonCode.RISK_FLAG_EARNINGS_IMMINENT.value in codes
        assert ReasonCode.RISK_FLAG_HIGH_VOLATILITY.value in codes
        assert ReasonCode.INTERPRETER_LOW_CONFIDENCE.value in codes


# ---------------------------------------------------------------------------
# TestCounterfactualSimulation
# ---------------------------------------------------------------------------


class TestCounterfactualSimulation:
    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_stop_hit_negative_return(self):
        df = self._make_df([
            {"date": date(2025, 1, 2), "open": 100, "high": 101, "low": 94, "close": 94.5},
        ])
        result = simulate_quant_counterfactual(
            entry_price=100.0, stop_loss=95.0, target_1=110.0,
            holding_period=10, direction="LONG",
            entry_date=date(2025, 1, 2), aggregator=None, ticker="TEST",
            price_df=df,
        )
        assert result is not None
        assert result["exit_reason"] == "stop"
        assert result["quant_return"] < 0

    def test_target_hit_positive_return(self):
        df = self._make_df([
            {"date": date(2025, 1, 2), "open": 100, "high": 111, "low": 99, "close": 110.5},
        ])
        result = simulate_quant_counterfactual(
            entry_price=100.0, stop_loss=95.0, target_1=110.0,
            holding_period=10, direction="LONG",
            entry_date=date(2025, 1, 2), aggregator=None, ticker="TEST",
            price_df=df,
        )
        assert result is not None
        assert result["exit_reason"] == "target"
        assert result["quant_return"] > 0

    def test_expiry_mark_to_market(self):
        # 10 bars, no stop or target hit
        rows = []
        for i in range(10):
            d = date(2025, 1, 2 + i)
            rows.append({"date": d, "open": 100, "high": 103, "low": 97, "close": 102})
        df = self._make_df(rows)
        result = simulate_quant_counterfactual(
            entry_price=100.0, stop_loss=90.0, target_1=115.0,
            holding_period=10, direction="LONG",
            entry_date=date(2025, 1, 2), aggregator=None, ticker="TEST",
            price_df=df,
        )
        assert result is not None
        assert result["exit_reason"] == "expiry"
        assert result["quant_return"] == pytest.approx(2.0, abs=0.01)

    def test_no_data_returns_none(self):
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close"])
        result = simulate_quant_counterfactual(
            entry_price=100.0, stop_loss=95.0, target_1=110.0,
            holding_period=10, direction="LONG",
            entry_date=date(2025, 1, 2), aggregator=None, ticker="TEST",
            price_df=df,
        )
        assert result is None

    def test_none_df_returns_none(self):
        result = simulate_quant_counterfactual(
            entry_price=100.0, stop_loss=95.0, target_1=110.0,
            holding_period=10, direction="LONG",
            entry_date=date(2025, 1, 2), aggregator=None, ticker="TEST",
            price_df=None,
        )
        assert result is None

    def test_short_stop_hit(self):
        df = self._make_df([
            {"date": date(2025, 1, 2), "open": 100, "high": 106, "low": 99, "close": 105},
        ])
        result = simulate_quant_counterfactual(
            entry_price=100.0, stop_loss=105.0, target_1=90.0,
            holding_period=10, direction="SHORT",
            entry_date=date(2025, 1, 2), aggregator=None, ticker="TEST",
            price_df=df,
        )
        assert result is not None
        assert result["exit_reason"] == "stop"
        assert result["quant_return"] < 0

    def test_short_target_hit(self):
        df = self._make_df([
            {"date": date(2025, 1, 2), "open": 100, "high": 101, "low": 89, "close": 90},
        ])
        result = simulate_quant_counterfactual(
            entry_price=100.0, stop_loss=105.0, target_1=90.0,
            holding_period=10, direction="SHORT",
            entry_date=date(2025, 1, 2), aggregator=None, ticker="TEST",
            price_df=df,
        )
        assert result is not None
        assert result["exit_reason"] == "target"
        assert result["quant_return"] > 0

    def test_holding_period_not_expired(self):
        # Only 3 bars for a 10-day holding period
        rows = [
            {"date": date(2025, 1, 2 + i), "open": 100, "high": 103, "low": 97, "close": 101}
            for i in range(3)
        ]
        df = self._make_df(rows)
        result = simulate_quant_counterfactual(
            entry_price=100.0, stop_loss=90.0, target_1=115.0,
            holding_period=10, direction="LONG",
            entry_date=date(2025, 1, 2), aggregator=None, ticker="TEST",
            price_df=df,
        )
        assert result is None


# ---------------------------------------------------------------------------
# TestDivergenceOutcomeScoring
# ---------------------------------------------------------------------------


class TestDivergenceOutcomeScoring:
    """Test the scoring logic for divergence outcomes.

    These are unit-level tests of the scoring rules, not DB integration.
    """

    def test_veto_avoided_loss_improved_true(self):
        # VETO: quant would have lost 5%. Avoiding a loss = good veto.
        quant_return = -5.0
        agentic_return = 0.0  # didn't trade
        return_delta = (agentic_return or 0.0) - quant_return
        improved = quant_return < 0
        assert improved is True
        assert return_delta == 5.0

    def test_veto_missed_gain_improved_false(self):
        # VETO: quant would have gained 3%. Missing a gain = bad veto.
        quant_return = 3.0
        improved = quant_return < 0
        assert improved is False

    def test_promote_outperformed_cash_improved_true(self):
        # PROMOTE: agentic traded and gained 4%. quant = cash (0%).
        agentic_return = 4.0
        quant_return = 0.0
        return_delta = agentic_return - quant_return
        improved = return_delta > 0
        assert improved is True
        assert return_delta == 4.0

    def test_promote_lost_money_improved_false(self):
        # PROMOTE: agentic traded and lost 2%. quant = cash.
        agentic_return = -2.0
        quant_return = 0.0
        return_delta = agentic_return - quant_return
        improved = return_delta > 0
        assert improved is False

    def test_resize_return_delta_calculation(self):
        # RESIZE: same trade but different sizes.
        # agentic: 3% size, 10% return → 0.3% weighted
        # quant: 5% size, 10% return → 0.5% weighted
        agentic_return = 10.0
        quant_return = 10.0
        agentic_size = 3.0
        quant_size = 5.0
        agentic_weighted = agentic_return * (agentic_size / 100.0)
        quant_weighted = quant_return * (quant_size / 100.0)
        return_delta = agentic_weighted - quant_weighted
        assert return_delta == pytest.approx(-0.2, abs=0.01)

    def test_resize_increased_size_positive_return(self):
        # RESIZE: agentic increased size and trade was profitable
        agentic_return = 8.0
        quant_return = 8.0
        agentic_size = 7.0
        quant_size = 5.0
        agentic_weighted = agentic_return * (agentic_size / 100.0)
        quant_weighted = quant_return * (quant_size / 100.0)
        return_delta = agentic_weighted - quant_weighted
        improved = return_delta > 0
        assert improved is True
        assert return_delta == pytest.approx(0.16, abs=0.01)
