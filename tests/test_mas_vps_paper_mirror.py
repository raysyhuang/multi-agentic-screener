"""Behaviour tests for the VPS paper-mirror launcher.

This file produces the measurement lane's artifacts. Until 2026-08-20 it lived
only on one host, outside every control built for this repository. These tests
exist so that never silently regresses again.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mas_vps_paper_mirror as launcher

SCRIPT = REPO / "scripts" / "mas_vps_paper_mirror.py"
GOOD_DB = "postgresql://u:p@127.0.0.1:5432/mas_mirror_main"


@pytest.fixture
def deploy(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"; (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").write_text("")
    out_root = tmp_path / "out"
    envfile = tmp_path / "mas.env"
    envfile.write_text(f"DATABASE_URL={GOOD_DB}\nPOLYGON_API_KEY=x\n")
    monkeypatch.setenv("MAS_MIRROR_REPO", str(repo))
    monkeypatch.setenv("MAS_MIRROR_OUT_ROOT", str(out_root))
    monkeypatch.setenv("MAS_MIRROR_ENV_FILES", str(envfile))
    return repo, out_root, envfile


# ── configuration: no host defaults, fail closed ─────────────────────────

@pytest.mark.parametrize("missing", ["MAS_MIRROR_REPO", "MAS_MIRROR_OUT_ROOT", "MAS_MIRROR_ENV_FILES"])
def test_every_required_variable_fails_closed(deploy, monkeypatch, missing):
    monkeypatch.delenv(missing, raising=False)
    with pytest.raises(launcher.ConfigError, match=missing):
        launcher.resolve_config()


def test_no_host_paths_are_embedded_in_the_file():
    """A default would publish the deployment layout of the host that runs it."""
    text = SCRIPT.read_text()
    for leak in ("/srv/", ".hermes", "/home/agent"):
        assert leak not in text, f"host path {leak!r} must not appear in a public repo"


def test_python_defaults_to_the_checkout_venv(deploy):
    repo, _, _ = deploy
    _, python, _, _ = launcher.resolve_config()
    assert python == repo / ".venv/bin/python"


def test_env_files_preserve_precedence_order(deploy, monkeypatch, tmp_path):
    a, b = tmp_path / "a.env", tmp_path / "b.env"
    a.write_text("X=1\n"); b.write_text("X=2\n")
    import os
    monkeypatch.setenv("MAS_MIRROR_ENV_FILES", os.pathsep.join([str(a), str(b)]))
    _, _, _, env_files = launcher.resolve_config()
    assert env_files == (a, b)
    merged: dict[str, str] = {}
    for p in env_files:
        launcher.load_env_file(p, merged)
    assert merged["X"] == "2", "later overlays must win"


def test_missing_env_overlay_is_an_error_not_a_skip(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        launcher.load_env_file(tmp_path / "nope.env", {})


# ── isolation guarantees on the RAW environment ──────────────────────────

def test_mirror_env_forces_paper_quant_only_and_empty_telegram():
    env = launcher.build_mirror_env({"TELEGRAM_BOT_TOKEN": "leaked", "TELEGRAM_CHAT_ID": "123"})
    assert env["TRADING_MODE"] == "PAPER"
    assert env["EXECUTION_MODE"] == "quant_only"
    assert env["TELEGRAM_BOT_TOKEN"] == "" and env["TELEGRAM_CHAT_ID"] == ""


@pytest.mark.parametrize("url", [
    "postgresql://u:p@db.example.com/mas_mirror_main",   # not local
    "postgresql://u:p@127.0.0.1/production",             # not a mirror db
    "mysql://u:p@127.0.0.1/mas_mirror_main",             # wrong scheme
    "",                                                   # absent
])
def test_non_isolated_database_is_refused(url):
    env = launcher.build_mirror_env({"DATABASE_URL": url})
    with pytest.raises(ValueError):
        launcher.validate_mirror_env(env)


def test_isolated_database_is_accepted():
    launcher.validate_mirror_env(launcher.build_mirror_env({"DATABASE_URL": GOOD_DB}))


# ── isolation guarantees on the RESOLVED settings, per review ────────────
# Checking the raw environment is not enough: pydantic-settings resolves a
# dotenv file in the checkout, so an unset variable can be re-supplied there.

def test_resolved_settings_reject_telegram_credentials():
    with pytest.raises(ValueError, match="Telegram"):
        launcher.validate_resolved_settings({
            "telegram_bot_token": "re-supplied-by-dotenv", "telegram_chat_id": "",
            "database_url": GOOD_DB, "trading_mode": "PAPER", "execution_mode": "quant_only"})


@pytest.mark.parametrize("field,value", [
    ("database_url", "postgresql://u:p@db.example.com/mas_mirror_main"),
    ("trading_mode", "LIVE"),
    ("execution_mode", "agentic_full"),
])
def test_resolved_settings_reject_unsafe_configuration(field, value):
    settings = {"telegram_bot_token": "", "telegram_chat_id": "",
                "database_url": GOOD_DB, "trading_mode": "PAPER", "execution_mode": "quant_only"}
    settings[field] = value
    with pytest.raises(ValueError):
        launcher.validate_resolved_settings(settings)


def test_settings_resolution_never_leaks_credentials_on_bad_output(monkeypatch, tmp_path):
    """stdout carries database_url and telegram tokens; a JSONDecodeError
    embeds a snippet of what it failed on, and main() prints the exception."""
    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout='{"database_url": "postgres://u:SECRET@h/db"', stderr="")
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as excinfo:
        launcher.resolved_settings({}, tmp_path / "py", tmp_path)
    message = str(excinfo.value)
    assert "SECRET" not in message and "postgres://" not in message
    assert "stdout withheld" in message


# ── the exchange-date partition key ──────────────────────────────────────

def test_run_date_uses_the_exchange_date_not_utc():
    """00:00-04:00 UTC is still the previous trading day in New York. #78 was
    this same defect in a smoke test."""
    moment = datetime(2026, 8, 16, 2, 30, tzinfo=UTC)
    assert moment.astimezone(ZoneInfo("America/New_York")).date().isoformat() == "2026-08-15"
    assert moment.date().isoformat() == "2026-08-16"


# ── the artifact contract ────────────────────────────────────────────────

def _smoke(monkeypatch, out_root: Path, phase: str) -> dict:
    """One controlled run with the worker and export stubbed out."""
    monkeypatch.setattr(launcher, "resolved_settings", lambda env, python, repo: {
        "telegram_bot_token": "", "telegram_chat_id": "",
        "database_url": GOOD_DB, "trading_mode": "PAPER", "execution_mode": "quant_only"})

    def fake_run(command, env, out, repo):
        assert env["TRADING_MODE"] == "PAPER"
        assert env["EXECUTION_MODE"] == "quant_only"
        assert env["TELEGRAM_BOT_TOKEN"] == ""
        if "--out" in command:
            Path(command[command.index("--out") + 1]).write_text(json.dumps({"today_picks": []}))
    monkeypatch.setattr(launcher, "run", fake_run)
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **k: "deadbeef\n")
    monkeypatch.setattr(sys, "argv", ["mirror", "--phase", phase])
    assert launcher.main() == 0
    meta = next(out_root.rglob("run-meta.json"))
    return json.loads(meta.read_text())


def test_manifest_stamps_the_hash_of_the_executing_launcher(deploy, monkeypatch):
    """launcher_sha256 must be the content hash of THIS file, so a run record
    can be compared against a committed revision.

    It proves byte-equivalence with a commit, NOT that the scheduler invoked
    the repository copy — an identical host copy passes the same check. The
    scheduler repoint is a separate, still-outstanding step.
    """
    _, out_root, _ = deploy
    manifest = _smoke(monkeypatch, out_root, "afternoon")
    assert manifest["launcher_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def test_manifest_records_both_dates_and_neither_is_mislabelled(deploy, monkeypatch):
    _, out_root, _ = deploy
    manifest = _smoke(monkeypatch, out_root, "morning")
    assert manifest["run_date_tz"] == "America/New_York"
    assert manifest["run_date"] == datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    assert manifest["run_date_utc"] == datetime.now(UTC).date().isoformat()


def test_artifacts_are_phase_separated(deploy, monkeypatch):
    """run() names log files after the last argv token, so two phases sharing a
    directory would overwrite each other's dashboard and leave run-meta's
    dashboard_sha256 pointing at bytes that no longer exist."""
    _, out_root, _ = deploy
    _smoke(monkeypatch, out_root, "morning")
    _smoke(monkeypatch, out_root, "afternoon")
    phases = {p.parent.name for p in out_root.rglob("run-meta.json")}
    assert phases == {"morning", "afternoon"}


def test_manifest_asserts_the_isolation_it_ran_under(deploy, monkeypatch):
    _, out_root, _ = deploy
    manifest = _smoke(monkeypatch, out_root, "afternoon")
    assert manifest["trading_mode"] == "PAPER"
    assert manifest["execution_mode"] == "quant_only"
    assert manifest["telegram_disabled"] is True
    assert manifest["dashboard_sha256"]
