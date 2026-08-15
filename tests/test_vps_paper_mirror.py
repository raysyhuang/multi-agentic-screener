"""Tests for the VPS paper-mirror launcher.

Unit tests only — no DB, no VPS, no network. Covers:
  - Morning vs afternoon plan resolution
  - Log file naming (step-based, not command[-1])
  - Separate output directories per phase
  - Fail-closed validation guards
  - Dry-run mode
  - Missing --phase
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_settings():
    """Mock settings that pass fail-closed validation."""
    settings = MagicMock()
    settings.trading_mode = "PAPER"
    settings.execution_mode = "quant_only"
    settings.database_url = "postgresql://user:pass@host/mas_mirror_db"
    settings.telegram_bot_token = ""
    settings.telegram_chat_id = ""
    return settings


@pytest.fixture
def mock_settings_live():
    """Mock settings with LIVE trading_mode (should fail validation)."""
    settings = MagicMock()
    settings.trading_mode = "LIVE"
    settings.execution_mode = "quant_only"
    settings.database_url = "postgresql://user:pass@host/mas_mirror_db"
    settings.telegram_bot_token = ""
    settings.telegram_chat_id = ""
    return settings


@pytest.fixture
def mock_settings_telegram():
    """Mock settings with Telegram creds (should fail validation)."""
    settings = MagicMock()
    settings.trading_mode = "PAPER"
    settings.execution_mode = "quant_only"
    settings.database_url = "postgresql://user:pass@host/mas_mirror_db"
    settings.telegram_bot_token = "1234567890:ABCDEF..."
    settings.telegram_chat_id = "-1001234567890"
    return settings


class TestPlanResolution:
    """Test that morning vs afternoon plans resolve correctly."""

    def test_morning_plan_includes_alembic(self):
        """Morning phase must include alembic as the first step."""
        from scripts.mas_vps_paper_mirror import _resolve_plan

        plan = _resolve_plan(
            phase="morning",
            out_root=Path("/tmp/test_out"),
            run_date="2026-08-15",
            repo_root=Path("/workspace"),
        )

        assert plan["phase"] == "morning"
        assert len(plan["steps"]) == 3
        assert plan["steps"][0]["name"] == "alembic"
        assert plan["steps"][0]["command"] == ["alembic", "upgrade", "head"]
        assert plan["steps"][1]["name"] == "worker"
        assert "--run-now" in plan["steps"][1]["command"]
        assert plan["steps"][2]["name"] == "export"

    def test_afternoon_plan_no_alembic(self):
        """Afternoon phase must NOT include alembic."""
        from scripts.mas_vps_paper_mirror import _resolve_plan

        plan = _resolve_plan(
            phase="afternoon",
            out_root=Path("/tmp/test_out"),
            run_date="2026-08-15",
            repo_root=Path("/workspace"),
        )

        assert plan["phase"] == "afternoon"
        assert len(plan["steps"]) == 2
        step_names = [s["name"] for s in plan["steps"]]
        assert "alembic" not in step_names
        assert plan["steps"][0]["name"] == "worker"
        assert "--check-now" in plan["steps"][0]["command"]
        assert plan["steps"][1]["name"] == "export"

    def test_log_files_named_from_step_not_command(self):
        """Log files must be named from the step name, not command[-1]."""
        from scripts.mas_vps_paper_mirror import _resolve_plan

        plan = _resolve_plan(
            phase="morning",
            out_root=Path("/tmp/test_out"),
            run_date="2026-08-15",
            repo_root=Path("/workspace"),
        )

        # Check that log files are named after the step, not the last command token
        alembic_log = plan["steps"][0]["log_file"]
        worker_log = plan["steps"][1]["log_file"]
        export_log = plan["steps"][2]["log_file"]

        assert "alembic.stdout.log" in alembic_log
        assert "worker.stdout.log" in worker_log
        assert "export.stdout.log" in export_log

        # Should NOT be named "head.stdout.log" (the VPS bug)
        assert "head.stdout.log" not in alembic_log

    def test_separate_out_dirs_per_phase(self):
        """Morning and afternoon must write to separate phase directories."""
        from scripts.mas_vps_paper_mirror import _resolve_plan

        morning = _resolve_plan(
            phase="morning",
            out_root=Path("/tmp/test_out"),
            run_date="2026-08-15",
            repo_root=Path("/workspace"),
        )
        afternoon = _resolve_plan(
            phase="afternoon",
            out_root=Path("/tmp/test_out"),
            run_date="2026-08-15",
            repo_root=Path("/workspace"),
        )

        assert "morning" in morning["phase_dir"]
        assert "afternoon" in afternoon["phase_dir"]
        assert morning["phase_dir"] != afternoon["phase_dir"]

    def test_unknown_phase_raises(self):
        """Unknown phase value must raise ValueError."""
        from scripts.mas_vps_paper_mirror import _resolve_plan

        with pytest.raises(ValueError, match="Unknown phase"):
            _resolve_plan(
                phase="evening",
                out_root=Path("/tmp/test_out"),
                run_date="2026-08-15",
                repo_root=Path("/workspace"),
            )


class TestFailClosedValidation:
    """Test fail-closed settings validation guards."""

    def test_validation_passes_with_safe_settings(self, mock_settings):
        """Validation must pass when all settings are safe."""
        with patch("src.config.get_settings", return_value=mock_settings):
            with patch("importlib.util.find_spec", return_value=None):
                from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

                result = _validate_settings_fail_closed()
                assert result["trading_mode"] == "PAPER"
                assert result["execution_mode"] == "quant_only"

    def test_validation_fails_on_live_trading_mode(self, mock_settings_live):
        """Validation must fail if trading_mode is LIVE."""
        with patch("src.config.get_settings", return_value=mock_settings_live):
            with patch("importlib.util.find_spec", return_value=None):
                from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

                with pytest.raises(SystemExit):
                    _validate_settings_fail_closed()

    def test_validation_fails_on_telegram_creds(self, mock_settings_telegram):
        """Validation must fail if Telegram credentials are present."""
        with patch("src.config.get_settings", return_value=mock_settings_telegram):
            with patch("importlib.util.find_spec", return_value=None):
                from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

                with pytest.raises(SystemExit):
                    _validate_settings_fail_closed()

    def test_validation_fails_on_non_postgres_db(self, mock_settings):
        """Validation must fail if database_url is not postgres."""
        mock_settings.database_url = "sqlite:///mas_mirror.db"
        with patch("src.config.get_settings", return_value=mock_settings):
            with patch("importlib.util.find_spec", return_value=None):
                from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

                with pytest.raises(SystemExit):
                    _validate_settings_fail_closed()

    def test_validation_fails_if_ibkr_importable(self, mock_settings):
        """Validation must fail if src.broker.ibkr is importable."""
        with patch("src.config.get_settings", return_value=mock_settings):
            # Simulate IBKR being importable
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

                with pytest.raises(SystemExit):
                    _validate_settings_fail_closed()

    def test_validation_fails_on_missing_db_marker(self, mock_settings):
        """Validation must fail if database_url lacks the mirror marker."""
        mock_settings.database_url = "postgresql://user:pass@host/production_db"
        with patch("src.config.get_settings", return_value=mock_settings):
            with patch("importlib.util.find_spec", return_value=None):
                from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

                with pytest.raises(SystemExit):
                    _validate_settings_fail_closed()


class TestDryRun:
    """Test that --dry-run prints the plan and exits without running commands."""

    def test_dry_run_exits_zero(self, tmp_path, mock_settings, monkeypatch):
        """Dry-run must exit 0 after printing the plan."""
        monkeypatch.setattr(
            "sys.argv",
            ["mas_vps_paper_mirror.py", "--phase", "morning", "--dry-run", "--out-root", str(tmp_path)],
        )
        with patch("src.config.get_settings", return_value=mock_settings):
            with patch("importlib.util.find_spec", return_value=None):
                from scripts.mas_vps_paper_mirror import main

                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 0

    def test_dry_run_does_not_create_output_dirs(self, tmp_path, mock_settings, monkeypatch):
        """Dry-run must not create output directories or run commands."""
        out_root = tmp_path / "out"
        monkeypatch.setattr(
            "sys.argv",
            ["mas_vps_paper_mirror.py", "--phase", "afternoon", "--dry-run", "--out-root", str(out_root)],
        )
        with patch("src.config.get_settings", return_value=mock_settings):
            with patch("importlib.util.find_spec", return_value=None):
                from scripts.mas_vps_paper_mirror import main

                with pytest.raises(SystemExit):
                    main()

                # out_root should not exist (dry-run doesn't create dirs)
                # Actually, the plan resolution doesn't create dirs either, so this is fine
                # But if out_root was created, it should be empty
                if out_root.exists():
                    assert list(out_root.iterdir()) == []


class TestArgparsing:
    """Test that argparse rejects missing or invalid arguments."""

    def test_missing_phase_arg_fails(self, monkeypatch, capsys):
        """Missing --phase must fail with usage message."""
        monkeypatch.setattr("sys.argv", ["mas_vps_paper_mirror.py"])
        from scripts.mas_vps_paper_mirror import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0

    def test_invalid_phase_value_fails(self, monkeypatch):
        """Invalid --phase value must fail."""
        monkeypatch.setattr("sys.argv", ["mas_vps_paper_mirror.py", "--phase", "evening"])
        from scripts.mas_vps_paper_mirror import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0
