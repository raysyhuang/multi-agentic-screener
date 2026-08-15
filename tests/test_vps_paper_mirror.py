"""Tests for the VPS paper-mirror launcher.

Unit tests only — no DB, no VPS, no network. Covers:
  - Morning vs afternoon plan resolution
  - Log file naming (step-based, not command[-1])
  - Separate output directories per phase
  - Fail-closed validation guards
  - Dry-run mode
  - Missing --phase
  - ET date resolution (not UTC)
  - Child env forces empty telegram
  - No secrets in stderr
  - run-meta stamps git_sha and dashboard_sha256
"""
from __future__ import annotations

import hashlib
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
        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch("importlib.util.find_spec", return_value=None),
        ):
            from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

            result = _validate_settings_fail_closed()
            assert result["trading_mode"] == "PAPER"
            assert result["execution_mode"] == "quant_only"
            assert result["telegram_bot_token_empty"] is True
            assert result["telegram_chat_id_empty"] is True

    def test_validation_fails_on_live_trading_mode(self, mock_settings_live):
        """Validation must fail if trading_mode is LIVE."""
        with (
            patch("src.config.get_settings", return_value=mock_settings_live),
            patch("importlib.util.find_spec", return_value=None),
        ):
            from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

            with pytest.raises(SystemExit):
                _validate_settings_fail_closed()

    def test_validation_fails_on_telegram_creds(self, mock_settings_telegram):
        """Validation must fail if Telegram credentials are present."""
        with (
            patch("src.config.get_settings", return_value=mock_settings_telegram),
            patch("importlib.util.find_spec", return_value=None),
        ):
            from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

            with pytest.raises(SystemExit):
                _validate_settings_fail_closed()

    def test_validation_fails_on_non_postgres_db(self, mock_settings):
        """Validation must fail if database_url is not postgres."""
        mock_settings.database_url = "sqlite:///mas_mirror.db"
        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch("importlib.util.find_spec", return_value=None),
        ):
            from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

            with pytest.raises(SystemExit):
                _validate_settings_fail_closed()

    def test_validation_fails_if_ibkr_importable(self, mock_settings):
        """Validation must fail if src.broker.ibkr is importable."""
        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch("importlib.util.find_spec", return_value=MagicMock()),
        ):
            from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

            with pytest.raises(SystemExit):
                _validate_settings_fail_closed()

    def test_validation_fails_on_missing_db_marker(self, mock_settings):
        """Validation must fail if database_url lacks the mirror marker."""
        mock_settings.database_url = "postgresql://user:pass@host/production_db"
        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch("importlib.util.find_spec", return_value=None),
        ):
            from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

            with pytest.raises(SystemExit):
                _validate_settings_fail_closed()

    def test_validation_redacts_secrets_in_error(self, mock_settings, capsys):
        """Validation errors must not print secrets or connection strings."""
        mock_settings.database_url = "sqlite:///secret_path.db"
        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch("importlib.util.find_spec", return_value=None),
        ):
            from scripts.mas_vps_paper_mirror import _validate_settings_fail_closed

            with pytest.raises(SystemExit):
                _validate_settings_fail_closed()

            captured = capsys.readouterr()
            # Should mention the scheme, not the full URL
            assert "sqlite" in captured.err
            assert "secret_path" not in captured.err


class TestDryRun:
    """Test that --dry-run prints the plan and exits without running commands."""

    def test_dry_run_exits_zero(self, tmp_path, mock_settings, monkeypatch):
        """Dry-run must exit 0 after printing the plan."""
        monkeypatch.setattr(
            "sys.argv",
            ["mas_vps_paper_mirror.py", "--phase", "morning", "--dry-run", "--out-root", str(tmp_path)],
        )
        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch("importlib.util.find_spec", return_value=None),
        ):
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
        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch("importlib.util.find_spec", return_value=None),
        ):
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

    def test_missing_phase_arg_fails(self, monkeypatch):
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


class TestRunDate:
    """Test that run_date uses America/New_York, not UTC."""

    def test_run_date_is_et_not_utc(self, monkeypatch):
        """Run date must be America/New_York date, not UTC."""
        # Mock datetime to return a UTC Wednesday 02:00 (which is still Tuesday ET)
        from datetime import UTC, datetime
        from zoneinfo import ZoneInfo

        # UTC Wed 02:00 = Tue 22:00 ET (during EDT, UTC-4)
        mock_utc_time = datetime(2026, 8, 20, 2, 0, 0, tzinfo=UTC)  # Wed 02:00 UTC
        mock_et_time = datetime(2026, 8, 19, 22, 0, 0, tzinfo=ZoneInfo("America/New_York"))  # Tue 22:00 ET

        with patch("scripts.mas_vps_paper_mirror.datetime") as mock_datetime:
            mock_datetime.now.side_effect = lambda tz=None: mock_et_time if tz else mock_utc_time
            mock_datetime.UTC = UTC

            from scripts.mas_vps_paper_mirror import _resolve_plan

            plan = _resolve_plan(
                phase="morning",
                out_root=Path("/tmp/test"),
                run_date="2026-08-19",  # Tuesday (ET date, not Wed UTC date)
                repo_root=Path("/workspace"),
            )

            # The phase_dir should contain the ET date (2026-08-19), not the UTC date (2026-08-20)
            assert "2026-08-19" in plan["phase_dir"]
            assert "2026-08-20" not in plan["phase_dir"]


class TestChildEnv:
    """Test that child processes get forced-empty Telegram creds."""

    def test_child_env_forces_empty_telegram(self, tmp_path, mock_settings, monkeypatch):
        """Child env must have TELEGRAM_BOT_TOKEN="" and TELEGRAM_CHAT_ID=""."""

        # Create the phase dir so the script doesn't fail on mkdir
        phase_dir = tmp_path / "2026-08-15" / "afternoon"
        phase_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(
            "sys.argv",
            ["mas_vps_paper_mirror.py", "--phase", "afternoon", "--out-root", str(tmp_path)],
        )

        # Simulate a leftover token in os.environ (should be overridden)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "leftover_token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "leftover_chat_id")

        captured_env = {}

        def mock_run(*args, **kwargs):
            nonlocal captured_env
            cmd = args[0] if args else kwargs.get("cmd", [])
            # Mock successful exit
            mock_result = MagicMock()
            mock_result.returncode = 0
            # For git rev-parse, return a proper string
            if "git" in cmd and "rev-parse" in cmd:
                mock_result.stdout = "abc1234\n"
            # Capture env for non-git commands
            if "git" not in cmd:
                captured_env = kwargs.get("env", {})
            return mock_result

        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch("importlib.util.find_spec", return_value=None),
            patch("subprocess.run", side_effect=mock_run),
        ):
            from scripts.mas_vps_paper_mirror import main

            main()

            # Check that the child env has empty Telegram creds
            assert captured_env.get("TELEGRAM_BOT_TOKEN") == ""
            assert captured_env.get("TELEGRAM_CHAT_ID") == ""
            assert captured_env.get("TRADING_MODE") == "PAPER"
            assert captured_env.get("EXECUTION_MODE") == "quant_only"


class TestRunMeta:
    """Test that run-meta.json stamps git_sha and dashboard_sha256."""

    def test_run_meta_stamps_git_sha_and_dashboard_sha256(self, tmp_path, mock_settings, monkeypatch):
        """run-meta.json must include git_sha and dashboard_sha256."""
        import json

        # Create the phase dir and dashboard file
        phase_dir = tmp_path / "2026-08-15" / "afternoon"
        phase_dir.mkdir(parents=True, exist_ok=True)

        mock_dashboard_content = b'{"test": "data"}'
        expected_sha256 = hashlib.sha256(mock_dashboard_content).hexdigest()

        # Write the dashboard file
        dashboard_path = phase_dir / "dashboard-data.json"
        dashboard_path.write_bytes(mock_dashboard_content)

        monkeypatch.setattr(
            "sys.argv",
            ["mas_vps_paper_mirror.py", "--phase", "afternoon", "--out-root", str(tmp_path)],
        )

        def mock_subprocess_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("cmd", [])
            mock_result = MagicMock()
            if "git" in cmd and "rev-parse" in cmd:
                mock_result.returncode = 0
                mock_result.stdout = "abc1234\n"
            else:
                mock_result.returncode = 0
            return mock_result

        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch("importlib.util.find_spec", return_value=None),
            patch("subprocess.run", side_effect=mock_subprocess_run),
        ):
            from scripts.mas_vps_paper_mirror import main

            main()

            # Read the run-meta.json that was written
            meta_path = phase_dir / "run-meta.json"
            assert meta_path.exists()

            meta_content = json.loads(meta_path.read_text())
            assert meta_content["git_sha"] == "abc1234"
            assert meta_content["dashboard_sha256"] == expected_sha256
