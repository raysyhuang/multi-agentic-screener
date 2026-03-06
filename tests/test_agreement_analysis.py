"""Tests for engine agreement/convergence analysis."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.engines.agreement_analysis import (
    AgreementReport,
    ConvergenceBucket,
    EnginePairStats,
    compute_agreement_report,
    format_agreement_report,
)


def _make_pick(
    engine_name: str,
    run_date: date,
    ticker: str,
    hit_target: bool,
    actual_return_pct: float,
    max_favorable_pct: float = 3.0,
    max_adverse_pct: float = -1.5,
) -> MagicMock:
    """Create a mock EnginePickOutcome row."""
    pick = MagicMock()
    pick.engine_name = engine_name
    pick.run_date = run_date
    pick.ticker = ticker
    pick.outcome_resolved = True
    pick.hit_target = hit_target
    pick.actual_return_pct = actual_return_pct
    pick.max_favorable_pct = max_favorable_pct
    pick.max_adverse_pct = max_adverse_pct
    return pick


class TestComputeAgreementReport:
    """Test agreement report computation logic."""

    @pytest.mark.asyncio
    async def test_empty_outcomes(self):
        """No resolved picks returns empty report."""
        with patch("src.engines.agreement_analysis.get_session") as mock_gs:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            report = await compute_agreement_report(lookback_days=90)

        assert report.total_resolved_picks == 0
        assert report.by_convergence == []
        assert report.engine_pairs == []

    @pytest.mark.asyncio
    async def test_single_engine_picks(self):
        """All picks from one engine -> convergence bucket of 1."""
        d = date(2026, 3, 1)
        picks = [
            _make_pick("koocore_d", d, "AAPL", True, 5.0),
            _make_pick("koocore_d", d, "MSFT", False, -2.0),
        ]
        with patch("src.engines.agreement_analysis.get_session") as mock_gs:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = picks
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            report = await compute_agreement_report(lookback_days=90)

        assert report.total_resolved_picks == 2
        assert len(report.by_convergence) == 1
        assert report.by_convergence[0].engine_count == 1
        assert report.by_convergence[0].total_picks == 2
        assert report.by_convergence[0].hits == 1
        assert report.by_convergence[0].hit_rate == 0.5

    @pytest.mark.asyncio
    async def test_convergent_picks(self):
        """Two engines picking same ticker on same date -> convergence=2."""
        d = date(2026, 3, 1)
        picks = [
            _make_pick("koocore_d", d, "AAPL", True, 5.0),
            _make_pick("gemini_stst", d, "AAPL", True, 4.0),
            _make_pick("koocore_d", d, "MSFT", False, -2.0),
        ]
        with patch("src.engines.agreement_analysis.get_session") as mock_gs:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = picks
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            report = await compute_agreement_report(lookback_days=90)

        assert report.total_resolved_picks == 3
        # Two buckets: convergence=1 (MSFT) and convergence=2 (AAPL)
        assert len(report.by_convergence) == 2
        bucket_1 = next(b for b in report.by_convergence if b.engine_count == 1)
        bucket_2 = next(b for b in report.by_convergence if b.engine_count == 2)
        assert bucket_1.total_picks == 1
        assert bucket_2.total_picks == 2  # both engines' AAPL picks
        assert bucket_2.hit_rate == 1.0

    @pytest.mark.asyncio
    async def test_engine_pair_stats(self):
        """Engine pair stats correctly identify agreements."""
        d = date(2026, 3, 1)
        picks = [
            _make_pick("koocore_d", d, "AAPL", True, 5.0),
            _make_pick("gemini_stst", d, "AAPL", True, 4.0),
            _make_pick("top3_7d", d, "TSLA", False, -3.0),
        ]
        with patch("src.engines.agreement_analysis.get_session") as mock_gs:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = picks
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            report = await compute_agreement_report(lookback_days=90)

        # 3 pairs: koocore+gemini, koocore+top3, gemini+top3
        assert len(report.engine_pairs) == 3
        kg_pair = next(
            p for p in report.engine_pairs
            if {p.engine_a, p.engine_b} == {"gemini_stst", "koocore_d"}
        )
        assert kg_pair.agreement_count == 1  # AAPL
        assert kg_pair.agreement_hit_rate == 1.0


class TestFormatAgreementReport:
    """Test report formatting."""

    def test_empty_report(self):
        report = AgreementReport(
            generated_at=datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc),
            lookback_days=90,
            total_resolved_picks=0,
        )
        text = format_agreement_report(report)
        assert "No resolved picks" in text
        assert "ENGINE AGREEMENT ANALYSIS" in text

    def test_report_with_data(self):
        report = AgreementReport(
            generated_at=datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc),
            lookback_days=90,
            total_resolved_picks=50,
            by_convergence=[
                ConvergenceBucket(
                    engine_count=1, total_picks=30, hits=10,
                    hit_rate=0.333, avg_return_pct=-0.5,
                    avg_mfe_pct=2.0, avg_mae_pct=-1.5,
                ),
                ConvergenceBucket(
                    engine_count=2, total_picks=20, hits=12,
                    hit_rate=0.6, avg_return_pct=3.0,
                    avg_mfe_pct=5.0, avg_mae_pct=-1.0,
                ),
            ],
            engine_pairs=[
                EnginePairStats(
                    engine_a="koocore_d", engine_b="gemini_stst",
                    agreement_count=10, agreement_hit_rate=0.6,
                    agreement_avg_return=3.0,
                ),
            ],
        )
        text = format_agreement_report(report)
        assert "CONVERGENCE BUCKETS" in text
        assert "ENGINE PAIR AGREEMENT" in text
        assert "koocore_d + gemini_stst" in text
        assert "50" in text  # total resolved picks
