"""Tests for get_divergence_stats() in performance.py."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.output.performance import get_divergence_stats


# ---------------------------------------------------------------------------
# Helpers — lightweight fakes for DivergenceEvent / DivergenceOutcome rows
# ---------------------------------------------------------------------------

def _make_event(
    id: int = 1,
    run_date: date | None = None,
    event_type: str = "VETO",
    reason_codes: list[str] | None = None,
    llm_cost_usd: float = 0.05,
    regime: str = "bull",
):
    ev = MagicMock()
    ev.id = id
    ev.run_date = run_date or date.today()
    ev.event_type = event_type
    ev.reason_codes = reason_codes or ["RISK_GATE_VETO"]
    ev.llm_cost_usd = llm_cost_usd
    ev.regime = regime
    return ev


def _make_outcome(
    divergence_id: int = 1,
    return_delta: float | None = 1.5,
    improved_vs_quant: bool | None = True,
):
    out = MagicMock()
    out.divergence_id = divergence_id
    out.return_delta = return_delta
    out.improved_vs_quant = improved_vs_quant
    return out


class _FakeResult:
    """Mimics SQLAlchemy async result.all()."""

    def __init__(self, rows: list):
        self._rows = rows

    def all(self):
        return self._rows


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
    """Return a patch that replaces get_session with a fake returning *rows*."""
    return patch(
        "src.output.performance.get_session",
        return_value=_FakeSession(rows),
    )


# ---------------------------------------------------------------------------
# TestGetDivergenceStatsEmpty
# ---------------------------------------------------------------------------

class TestGetDivergenceStatsEmpty:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_rows(self):
        with _patch_session([]):
            result = await get_divergence_stats(days=30)
        assert result is None


# ---------------------------------------------------------------------------
# TestGetDivergenceStatsBasic
# ---------------------------------------------------------------------------

class TestGetDivergenceStatsBasic:
    @pytest.mark.asyncio
    async def test_single_resolved_veto(self):
        ev = _make_event(id=1, event_type="VETO")
        out = _make_outcome(divergence_id=1, return_delta=2.0, improved_vs_quant=True)
        with _patch_session([(ev, out)]):
            result = await get_divergence_stats(days=30)

        assert result is not None
        assert result["total_events"] == 1
        assert result["total_resolved"] == 1
        assert result["overall_improvement_rate"] == 1.0
        assert result["net_portfolio_delta"] == 2.0
        assert result["by_event_type"]["VETO"]["win_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_unresolved_event_not_counted_as_resolved(self):
        ev = _make_event(id=2, event_type="PROMOTE")
        with _patch_session([(ev, None)]):
            result = await get_divergence_stats(days=30)

        assert result is not None
        assert result["total_events"] == 1
        assert result["total_resolved"] == 0
        assert result["overall_improvement_rate"] is None
        assert result["net_portfolio_delta"] is None

    @pytest.mark.asyncio
    async def test_mixed_event_types(self):
        ev1 = _make_event(id=1, event_type="VETO")
        out1 = _make_outcome(divergence_id=1, return_delta=2.0, improved_vs_quant=True)
        ev2 = _make_event(id=2, event_type="PROMOTE")
        out2 = _make_outcome(divergence_id=2, return_delta=-1.0, improved_vs_quant=False)
        ev3 = _make_event(id=3, event_type="RESIZE")
        # ev3 unresolved

        with _patch_session([(ev1, out1), (ev2, out2), (ev3, None)]):
            result = await get_divergence_stats(days=30)

        assert result["total_events"] == 3
        assert result["total_resolved"] == 2
        assert result["overall_improvement_rate"] == 0.5
        assert result["net_portfolio_delta"] == 1.0  # 2.0 + (-1.0)
        assert "VETO" in result["by_event_type"]
        assert "PROMOTE" in result["by_event_type"]
        assert "RESIZE" in result["by_event_type"]


# ---------------------------------------------------------------------------
# TestRunLevelDeltas
# ---------------------------------------------------------------------------

class TestRunLevelDeltas:
    @pytest.mark.asyncio
    async def test_two_events_same_date_summed(self):
        run_date = date(2025, 3, 1)
        ev1 = _make_event(id=1, run_date=run_date)
        out1 = _make_outcome(divergence_id=1, return_delta=1.0, improved_vs_quant=True)
        ev2 = _make_event(id=2, run_date=run_date)
        out2 = _make_outcome(divergence_id=2, return_delta=0.5, improved_vs_quant=True)

        with _patch_session([(ev1, out1), (ev2, out2)]):
            result = await get_divergence_stats(days=365)

        deltas = result["run_level_deltas"]
        assert len(deltas) == 1
        assert deltas[0]["divergence_count"] == 2
        assert deltas[0]["net_delta"] == 1.5
        assert deltas[0]["positive"] is True

    @pytest.mark.asyncio
    async def test_trend_none_when_fewer_than_4_runs(self):
        dates = [date(2025, 3, i) for i in range(1, 4)]
        rows = []
        for i, d in enumerate(dates):
            ev = _make_event(id=i + 1, run_date=d)
            out = _make_outcome(divergence_id=i + 1, return_delta=1.0, improved_vs_quant=True)
            rows.append((ev, out))

        with _patch_session(rows):
            result = await get_divergence_stats(days=365)

        assert result["run_level_trend"] is None

    @pytest.mark.asyncio
    async def test_trend_improving(self):
        # 4 runs: early negative, recent positive
        dates = [date(2025, 3, i) for i in range(1, 5)]
        deltas = [-1.0, -0.5, 1.0, 2.0]
        rows = []
        for i, (d, delta) in enumerate(zip(dates, deltas)):
            ev = _make_event(id=i + 1, run_date=d)
            out = _make_outcome(divergence_id=i + 1, return_delta=delta, improved_vs_quant=delta > 0)
            rows.append((ev, out))

        with _patch_session(rows):
            result = await get_divergence_stats(days=365)

        trend = result["run_level_trend"]
        assert trend is not None
        assert trend["improving"] is True
        assert trend["recent_4_avg_delta"] > trend["early_4_avg_delta"]

    @pytest.mark.asyncio
    async def test_trend_not_improving(self):
        dates = [date(2025, 3, i) for i in range(1, 5)]
        deltas = [2.0, 1.0, -0.5, -1.0]
        rows = []
        for i, (d, delta) in enumerate(zip(dates, deltas)):
            ev = _make_event(id=i + 1, run_date=d)
            out = _make_outcome(divergence_id=i + 1, return_delta=delta, improved_vs_quant=delta > 0)
            rows.append((ev, out))

        with _patch_session(rows):
            result = await get_divergence_stats(days=365)

        trend = result["run_level_trend"]
        assert trend is not None
        assert trend["improving"] is False


# ---------------------------------------------------------------------------
# TestByReasonCode
# ---------------------------------------------------------------------------

class TestByReasonCode:
    @pytest.mark.asyncio
    async def test_reason_code_stats_correct(self):
        ev = _make_event(id=1, reason_codes=["RISK_GATE_VETO"])
        out = _make_outcome(divergence_id=1, return_delta=1.5, improved_vs_quant=True)

        with _patch_session([(ev, out)]):
            result = await get_divergence_stats(days=30)

        rc = result["by_reason_code"]["RISK_GATE_VETO"]
        assert rc["events"] == 1
        assert rc["with_outcome"] == 1
        assert rc["win_rate"] == 1.0
        assert rc["avg_return_delta"] == 1.5

    @pytest.mark.asyncio
    async def test_multiple_reason_codes_per_event(self):
        ev = _make_event(id=1, reason_codes=["RISK_GATE_VETO", "REGIME_MISMATCH"])
        out = _make_outcome(divergence_id=1, return_delta=1.0, improved_vs_quant=True)

        with _patch_session([(ev, out)]):
            result = await get_divergence_stats(days=30)

        assert "RISK_GATE_VETO" in result["by_reason_code"]
        assert "REGIME_MISMATCH" in result["by_reason_code"]
        # Each code counted once
        assert result["by_reason_code"]["RISK_GATE_VETO"]["events"] == 1
        assert result["by_reason_code"]["REGIME_MISMATCH"]["events"] == 1


# ---------------------------------------------------------------------------
# TestCostEfficiency
# ---------------------------------------------------------------------------

class TestCostEfficiency:
    @pytest.mark.asyncio
    async def test_cost_per_positive_divergence(self):
        ev = _make_event(id=1, llm_cost_usd=0.10)
        out = _make_outcome(divergence_id=1, return_delta=2.0, improved_vs_quant=True)

        with _patch_session([(ev, out)]):
            result = await get_divergence_stats(days=30)

        ce = result["cost_efficiency"]
        assert ce["total_llm_cost"] == 0.10
        assert ce["cost_per_positive_divergence"] == 0.10
        assert ce["net_delta_per_dollar"] == 20.0

    @pytest.mark.asyncio
    async def test_zero_cost_returns_none(self):
        ev = _make_event(id=1, llm_cost_usd=0.0)
        out = _make_outcome(divergence_id=1, return_delta=1.0, improved_vs_quant=True)

        with _patch_session([(ev, out)]):
            result = await get_divergence_stats(days=30)

        ce = result["cost_efficiency"]
        assert ce["cost_per_positive_divergence"] is None
        assert ce["net_delta_per_dollar"] is None


# ---------------------------------------------------------------------------
# TestNetPortfolioDelta
# ---------------------------------------------------------------------------

class TestNetPortfolioDelta:
    @pytest.mark.asyncio
    async def test_positive_net(self):
        ev1 = _make_event(id=1)
        out1 = _make_outcome(divergence_id=1, return_delta=3.0, improved_vs_quant=True)
        ev2 = _make_event(id=2)
        out2 = _make_outcome(divergence_id=2, return_delta=-1.0, improved_vs_quant=False)

        with _patch_session([(ev1, out1), (ev2, out2)]):
            result = await get_divergence_stats(days=30)

        assert result["net_portfolio_delta"] == 2.0

    @pytest.mark.asyncio
    async def test_negative_net(self):
        ev1 = _make_event(id=1)
        out1 = _make_outcome(divergence_id=1, return_delta=-3.0, improved_vs_quant=False)
        ev2 = _make_event(id=2)
        out2 = _make_outcome(divergence_id=2, return_delta=1.0, improved_vs_quant=True)

        with _patch_session([(ev1, out1), (ev2, out2)]):
            result = await get_divergence_stats(days=30)

        assert result["net_portfolio_delta"] == -2.0
