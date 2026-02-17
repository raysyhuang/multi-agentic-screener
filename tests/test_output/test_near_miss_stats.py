"""Tests for get_near_miss_stats() in performance.py."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.output.performance import get_near_miss_stats


# ---------------------------------------------------------------------------
# Helpers — lightweight fakes for NearMiss rows
# ---------------------------------------------------------------------------

def _make_near_miss(
    id: int = 1,
    run_date: date | None = None,
    ticker: str = "AAPL",
    stage: str = "debate",
    debate_verdict: str = "REJECT",
    net_conviction: float = 40.0,
    bull_conviction: float = 60.0,
    bear_conviction: float = 70.0,
    key_risk: str = "Overextended",
    risk_gate_decision: str | None = None,
    risk_gate_reasoning: str | None = None,
    interpreter_confidence: float = 75.0,
    signal_model: str = "breakout",
    regime: str = "bull",
    entry_price: float = 100.0,
    stop_loss: float = 95.0,
    target_price: float = 110.0,
    timeframe_days: int = 10,
):
    nm = MagicMock()
    nm.id = id
    nm.run_date = run_date or date.today()
    nm.ticker = ticker
    nm.stage = stage
    nm.debate_verdict = debate_verdict
    nm.net_conviction = net_conviction
    nm.bull_conviction = bull_conviction
    nm.bear_conviction = bear_conviction
    nm.key_risk = key_risk
    nm.risk_gate_decision = risk_gate_decision
    nm.risk_gate_reasoning = risk_gate_reasoning
    nm.interpreter_confidence = interpreter_confidence
    nm.signal_model = signal_model
    nm.regime = regime
    nm.entry_price = entry_price
    nm.stop_loss = stop_loss
    nm.target_price = target_price
    nm.timeframe_days = timeframe_days
    nm.outcome_resolved = False
    nm.counterfactual_return = None
    nm.counterfactual_exit_reason = None
    return nm


class _FakeScalarsResult:
    """Mimics SQLAlchemy result.scalars().all()."""

    def __init__(self, rows: list):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeResult:
    def __init__(self, rows: list):
        self._rows = rows

    def scalars(self):
        return _FakeScalarsResult(self._rows)


class _FakeSession:
    """Async context manager returning a fake session with preset rows."""

    def __init__(self, rows: list):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, stmt):
        return _FakeResult(self._rows)


def _patch_session(rows: list):
    return patch(
        "src.output.performance.get_session",
        return_value=_FakeSession(rows),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoDataReturnsNone:
    @pytest.mark.asyncio
    async def test_no_data_returns_none(self):
        with _patch_session([]):
            result = await get_near_miss_stats(days=30)
        assert result is None


class TestBasicStats:
    @pytest.mark.asyncio
    async def test_single_near_miss(self):
        nm = _make_near_miss(id=1, net_conviction=40.0, stage="debate")
        with _patch_session([nm]):
            result = await get_near_miss_stats(days=30)

        assert result is not None
        assert result["total_near_misses"] == 1
        assert result["period_days"] == 30
        assert result["by_stage"]["debate"]["count"] == 1
        assert result["by_stage"]["debate"]["avg_conviction"] == 40.0

    @pytest.mark.asyncio
    async def test_multiple_near_misses(self):
        nms = [
            _make_near_miss(id=1, ticker="A", net_conviction=30.0),
            _make_near_miss(id=2, ticker="B", net_conviction=60.0),
        ]
        with _patch_session(nms):
            result = await get_near_miss_stats(days=30)

        assert result["total_near_misses"] == 2
        assert result["by_stage"]["debate"]["count"] == 2
        assert result["by_stage"]["debate"]["avg_conviction"] == 45.0


class TestConvictionDistribution:
    @pytest.mark.asyncio
    async def test_conviction_bucketing(self):
        nms = [
            _make_near_miss(id=1, net_conviction=10.0),   # 0-25
            _make_near_miss(id=2, net_conviction=35.0),   # 25-50
            _make_near_miss(id=3, net_conviction=55.0),   # 50-75
            _make_near_miss(id=4, net_conviction=80.0),   # 75-100
            _make_near_miss(id=5, net_conviction=60.0),   # 50-75
        ]
        with _patch_session(nms):
            result = await get_near_miss_stats(days=30)

        dist = result["conviction_distribution"]
        assert dist["0_25"] == 1
        assert dist["25_50"] == 1
        assert dist["50_75"] == 2
        assert dist["75_100"] == 1

    @pytest.mark.asyncio
    async def test_boundary_values(self):
        """Test exact boundary values: 0, 25, 50, 75."""
        nms = [
            _make_near_miss(id=1, net_conviction=0.0),    # 0-25
            _make_near_miss(id=2, net_conviction=25.0),   # 25-50
            _make_near_miss(id=3, net_conviction=50.0),   # 50-75
            _make_near_miss(id=4, net_conviction=75.0),   # 75-100
        ]
        with _patch_session(nms):
            result = await get_near_miss_stats(days=30)

        dist = result["conviction_distribution"]
        assert dist["0_25"] == 1
        assert dist["25_50"] == 1
        assert dist["50_75"] == 1
        assert dist["75_100"] == 1


class TestByStageBreakdown:
    @pytest.mark.asyncio
    async def test_debate_vs_risk_gate(self):
        nms = [
            _make_near_miss(id=1, stage="debate", net_conviction=30.0),
            _make_near_miss(id=2, stage="debate", net_conviction=40.0),
            _make_near_miss(id=3, stage="risk_gate", net_conviction=55.0,
                           risk_gate_decision="VETO"),
        ]
        with _patch_session(nms):
            result = await get_near_miss_stats(days=30)

        assert result["by_stage"]["debate"]["count"] == 2
        assert result["by_stage"]["debate"]["avg_conviction"] == 35.0
        assert result["by_stage"]["risk_gate"]["count"] == 1
        assert result["by_stage"]["risk_gate"]["avg_conviction"] == 55.0


class TestByRegime:
    @pytest.mark.asyncio
    async def test_regime_grouping(self):
        nms = [
            _make_near_miss(id=1, regime="bull", net_conviction=40.0),
            _make_near_miss(id=2, regime="bear", net_conviction=50.0),
            _make_near_miss(id=3, regime="bear", net_conviction=60.0),
        ]
        with _patch_session(nms):
            result = await get_near_miss_stats(days=30)

        assert result["by_regime"]["bull"]["count"] == 1
        assert result["by_regime"]["bull"]["avg_conviction"] == 40.0
        assert result["by_regime"]["bear"]["count"] == 2
        assert result["by_regime"]["bear"]["avg_conviction"] == 55.0

    @pytest.mark.asyncio
    async def test_none_regime_becomes_unknown(self):
        nm = _make_near_miss(id=1, regime=None, net_conviction=30.0)
        with _patch_session([nm]):
            result = await get_near_miss_stats(days=30)

        assert "unknown" in result["by_regime"]


class TestBySignalModel:
    @pytest.mark.asyncio
    async def test_signal_model_grouping(self):
        nms = [
            _make_near_miss(id=1, signal_model="breakout", net_conviction=40.0),
            _make_near_miss(id=2, signal_model="mean_reversion", net_conviction=55.0),
            _make_near_miss(id=3, signal_model="breakout", net_conviction=60.0),
        ]
        with _patch_session(nms):
            result = await get_near_miss_stats(days=30)

        assert result["by_signal_model"]["breakout"]["count"] == 2
        assert result["by_signal_model"]["breakout"]["avg_conviction"] == 50.0
        assert result["by_signal_model"]["mean_reversion"]["count"] == 1


class TestClosestMissesOrdering:
    @pytest.mark.asyncio
    async def test_top_3_by_conviction_descending(self):
        nms = [
            _make_near_miss(id=1, ticker="LOW", net_conviction=20.0),
            _make_near_miss(id=2, ticker="MID", net_conviction=50.0),
            _make_near_miss(id=3, ticker="HIGH", net_conviction=70.0),
            _make_near_miss(id=4, ticker="TOP", net_conviction=80.0),
        ]
        with _patch_session(nms):
            result = await get_near_miss_stats(days=30)

        closest = result["closest_misses"]
        assert len(closest) == 3
        assert closest[0]["ticker"] == "TOP"
        assert closest[0]["conviction"] == 80.0
        assert closest[1]["ticker"] == "HIGH"
        assert closest[2]["ticker"] == "MID"

    @pytest.mark.asyncio
    async def test_fewer_than_3_returns_all(self):
        nms = [
            _make_near_miss(id=1, ticker="ONLY", net_conviction=45.0),
        ]
        with _patch_session(nms):
            result = await get_near_miss_stats(days=30)

        assert len(result["closest_misses"]) == 1
        assert result["closest_misses"][0]["ticker"] == "ONLY"

    @pytest.mark.asyncio
    async def test_closest_misses_include_key_risk_and_date(self):
        d = date(2026, 2, 15)
        nm = _make_near_miss(id=1, ticker="TST", net_conviction=60.0,
                            key_risk="High P/E", run_date=d, stage="risk_gate")
        with _patch_session([nm]):
            result = await get_near_miss_stats(days=30)

        miss = result["closest_misses"][0]
        assert miss["key_risk"] == "High P/E"
        assert miss["run_date"] == "2026-02-15"
        assert miss["stage"] == "risk_gate"
