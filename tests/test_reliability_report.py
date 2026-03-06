"""Tests for engine reliability report."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.engines.reliability_report import (
    EngineCoverageStats,
    EngineHitRateBreakdown,
    EngineRunStats,
    ReliabilityReport,
    _trading_days_in_range,
    compute_reliability_report,
    format_reliability_report,
)


class TestTradingDays:
    def test_weekdays_only(self):
        # Mon Mar 2 to Sun Mar 8, 2026
        days = _trading_days_in_range(date(2026, 3, 2), date(2026, 3, 8))
        assert len(days) == 5  # Mon-Fri
        assert all(d.weekday() < 5 for d in days)

    def test_single_weekend_day(self):
        days = _trading_days_in_range(date(2026, 3, 7), date(2026, 3, 7))  # Saturday
        assert len(days) == 0

    def test_single_weekday(self):
        days = _trading_days_in_range(date(2026, 3, 2), date(2026, 3, 2))  # Monday
        assert len(days) == 1


def _mock_session_with(rows):
    """Create a mock get_session context manager returning rows."""
    mock_gs = MagicMock()
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = rows
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_gs


def _make_engine_run(
    engine_name: str,
    run_date: date,
    status: str = "success",
    fetch_duration_ms: int | None = 5000,
    picks_count: int | None = 3,
    candidates_screened: int | None = 150,
    error_message: str | None = None,
) -> MagicMock:
    row = MagicMock()
    row.engine_name = engine_name
    row.run_date = run_date
    row.status = status
    row.fetch_duration_ms = fetch_duration_ms
    row.picks_count = picks_count
    row.candidates_screened = candidates_screened
    row.error_message = error_message
    return row


def _make_engine_result(
    engine_name: str,
    run_date: date,
    picks_count: int = 3,
) -> MagicMock:
    row = MagicMock()
    row.engine_name = engine_name
    row.run_date = run_date
    row.picks_count = picks_count
    return row


def _make_pick_outcome(
    engine_name: str,
    run_date: date,
    ticker: str,
    strategy: str,
    hit_target: bool,
    actual_return_pct: float,
) -> MagicMock:
    row = MagicMock()
    row.engine_name = engine_name
    row.run_date = run_date
    row.ticker = ticker
    row.strategy = strategy
    row.outcome_resolved = True
    row.hit_target = hit_target
    row.actual_return_pct = actual_return_pct
    return row


class TestComputeReliabilityReport:
    @pytest.mark.asyncio
    async def test_empty_data(self):
        """All sections return empty with no data."""
        mock_gs = _mock_session_with([])
        with patch("src.engines.reliability_report.get_session", mock_gs):
            report = await compute_reliability_report(lookback_days=30)

        assert report.run_stats == []
        assert len(report.coverage_stats) == 3  # KNOWN_ENGINES
        assert report.hit_rate_breakdown == []

    @pytest.mark.asyncio
    async def test_run_stats_success_and_failure(self):
        """Mixes of success/failure compute correctly."""
        runs = [
            _make_engine_run("koocore_d", date(2026, 3, 1), "success"),
            _make_engine_run("koocore_d", date(2026, 3, 2), "success"),
            _make_engine_run("koocore_d", date(2026, 3, 3), "failed",
                             fetch_duration_ms=None, picks_count=None,
                             candidates_screened=None, error_message="timeout"),
        ]
        mock_gs = _mock_session_with(runs)
        with patch("src.engines.reliability_report.get_session", mock_gs):
            from src.engines.reliability_report import _compute_run_stats
            stats = await _compute_run_stats(date(2026, 2, 1))

        assert len(stats) == 1
        s = stats[0]
        assert s.engine_name == "koocore_d"
        assert s.total_runs == 3
        assert s.successes == 2
        assert s.failures == 1
        assert s.success_rate == pytest.approx(2 / 3)
        assert s.failure_breakdown == {"failed": 1}
        assert s.avg_fetch_duration_ms == 5000.0
        assert s.last_error == "timeout"

    @pytest.mark.asyncio
    async def test_hit_rate_by_strategy(self):
        """Per-strategy breakdown works."""
        picks = [
            _make_pick_outcome("gemini_stst", date(2026, 3, 1), "AAPL", "momentum", True, 5.0),
            _make_pick_outcome("gemini_stst", date(2026, 3, 1), "MSFT", "momentum", False, -2.0),
            _make_pick_outcome("gemini_stst", date(2026, 3, 2), "TSLA", "mean_reversion", True, 3.0),
        ]
        mock_gs = _mock_session_with(picks)
        with patch("src.engines.reliability_report.get_session", mock_gs):
            from src.engines.reliability_report import _compute_hit_rate_breakdown
            breakdowns = await _compute_hit_rate_breakdown(date(2026, 2, 1))

        assert len(breakdowns) == 1
        b = breakdowns[0]
        assert b.engine_name == "gemini_stst"
        assert b.total_resolved == 3
        assert b.total_hits == 2
        assert b.overall_hit_rate == pytest.approx(2 / 3)
        assert "momentum" in b.by_strategy
        assert b.by_strategy["momentum"]["resolved"] == 2
        assert b.by_strategy["momentum"]["hits"] == 1
        assert "mean_reversion" in b.by_strategy
        assert b.by_strategy["mean_reversion"]["hit_rate"] == 1.0


class TestFormatReliabilityReport:
    def test_empty_report(self):
        report = ReliabilityReport(
            generated_at=datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc),
            lookback_days=90,
        )
        text = format_reliability_report(report)
        assert "ENGINE RELIABILITY REPORT" in text
        assert "No engine_runs data" in text

    def test_full_report_formatting(self):
        report = ReliabilityReport(
            generated_at=datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc),
            lookback_days=90,
            run_stats=[
                EngineRunStats(
                    engine_name="koocore_d", total_runs=10, successes=8,
                    failures=2, success_rate=0.8,
                    failure_breakdown={"failed": 1, "no_response": 1},
                    avg_fetch_duration_ms=4500.0, avg_picks_count=3.2,
                    avg_candidates_screened=180.0,
                    last_success_date=date(2026, 3, 5),
                    last_failure_date=date(2026, 3, 3),
                    last_error="Connection refused",
                ),
            ],
            coverage_stats=[
                EngineCoverageStats(
                    engine_name="koocore_d", total_days_reported=15,
                    first_report_date=date(2026, 2, 10),
                    last_report_date=date(2026, 3, 5),
                    avg_picks_per_day=3.5,
                    missing_dates=[date(2026, 3, 4)],
                ),
            ],
            hit_rate_breakdown=[
                EngineHitRateBreakdown(
                    engine_name="koocore_d", total_resolved=20,
                    total_hits=6, overall_hit_rate=0.3,
                    avg_return_pct=-0.5,
                    by_strategy={
                        "momentum": {"resolved": 12, "hits": 4, "hit_rate": 0.333, "avg_return": 1.0},
                    },
                ),
            ],
        )
        text = format_reliability_report(report)
        assert "koocore_d" in text
        assert "80%" in text  # success rate
        assert "4500ms" in text
        assert "momentum" in text
        assert "Connection refused" in text
        assert "2026-03-04" in text  # missing date
