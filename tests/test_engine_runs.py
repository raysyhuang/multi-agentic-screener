"""Tests for EngineRun model and persistence logic."""

from datetime import date, datetime, timezone


from src.db.models import EngineRun


class TestEngineRunModel:
    """Test EngineRun ORM model structure."""

    def test_success_row_fields(self):
        """EngineRun can represent a successful engine collection."""
        now = datetime.now(timezone.utc)
        run = EngineRun(
            engine_name="koocore_d",
            run_date=date(2026, 3, 6),
            attempt=1,
            status="success",
            fetch_started_at=now,
            fetch_finished_at=now,
            fetch_duration_ms=1234,
            picks_count=5,
            candidates_screened=200,
            payload_hash="abc123",
            engine_result_id=42,
        )
        assert run.engine_name == "koocore_d"
        assert run.status == "success"
        assert run.picks_count == 5
        assert run.candidates_screened == 200
        assert run.error_message is None

    def test_failure_row_fields(self):
        """EngineRun can represent a failed engine collection."""
        run = EngineRun(
            engine_name="gemini_stst",
            run_date=date(2026, 3, 6),
            attempt=1,
            status="failed",
            error_message="Connection timeout after 30s",
        )
        assert run.engine_name == "gemini_stst"
        assert run.status == "failed"
        assert run.error_message == "Connection timeout after 30s"
        assert run.picks_count is None
        assert run.engine_result_id is None

    def test_all_status_values(self):
        """All planned status values can be set."""
        statuses = ["success", "failed", "quality_rejected", "no_response", "no_output", "timeout"]
        for status in statuses:
            run = EngineRun(
                engine_name="test",
                run_date=date(2026, 3, 6),
                attempt=1,
                status=status,
            )
            assert run.status == status

    def test_tablename(self):
        assert EngineRun.__tablename__ == "engine_runs"

    def test_unique_constraint_exists(self):
        """Unique constraint on (engine_name, run_date, attempt) is defined."""
        constraint_names = [
            c.name for c in EngineRun.__table_args__ if hasattr(c, "name")
        ]
        assert "uq_engine_run_name_date_attempt" in constraint_names


class TestFailureStatusMapping:
    """Test the failure kind -> status mapping used in main.py."""

    def test_mapping_covers_all_failure_kinds(self):
        """All EngineFailureKind values map to a status."""
        from src.engines.collector import EngineFailureKind
        from typing import get_args

        failure_kinds = get_args(EngineFailureKind)
        status_map = {
            "exception": "failed",
            "no_output": "no_output",
            "no_response": "no_response",
            "quality_rejected": "quality_rejected",
        }
        for kind in failure_kinds:
            assert kind in status_map, f"EngineFailureKind '{kind}' not in status map"
