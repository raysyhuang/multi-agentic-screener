"""Regression tests — KooCore-D engine endpoint fallback ordering.

Bug context (2026-02):
    After Eco dyno restart, the fallback chain checked local filesystem
    before GitHub API, returning stale ``outputs/`` data.  The fix
    reordered priority: disk cache → GitHub API → local filesystem.

These tests monkeypatch the three data sources and verify:
  1. Disk cache is the primary source
  2. GitHub overwrites disk when it has a newer run_date
  3. Local filesystem is the last resort (dev only)
  4. Graceful degradation when sources fail
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import sys

import pytest

# ── Import KooCore-D's engine_endpoint via absolute path ─────────────────
# We can't use ``from src.api.engine_endpoint import ...`` because MAS's own
# ``src/`` package shadows KooCore-D's during full-suite collection.
_KOOCORE_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / ".." / "KooCore-D" / "src" / "api" / "engine_endpoint.py"
)
_spec = importlib.util.spec_from_file_location("koocore_engine_endpoint", _KOOCORE_MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["koocore_engine_endpoint"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

_is_newer_run_date = _mod._is_newer_run_date
_load_latest_from_disk = _mod._load_latest_from_disk
_load_latest_from_github = _mod._load_latest_from_github
_parse_run_date = _mod._parse_run_date
get_engine_results = _mod.get_engine_results

# Module path for monkeypatching
_MODULE = "koocore_engine_endpoint"


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_payload(run_date: str) -> dict:
    """Minimal valid engine payload for a given run_date."""
    return {
        "engine_name": "koocore_d",
        "engine_version": "2.0",
        "run_date": run_date,
        "run_timestamp": f"{run_date}T12:00:00",
        "regime": None,
        "picks": [
            {
                "ticker": "AAPL",
                "strategy": "swing",
                "entry_price": 150.0,
                "stop_loss": 142.5,
                "target_price": 165.0,
                "confidence": 45.0,
                "holding_period_days": 14,
                "thesis": "test",
                "risk_factors": [],
                "raw_score": 4.5,
                "metadata": {},
            }
        ],
        "candidates_screened": 35,
        "pipeline_duration_s": 10.0,
        "status": "success",
    }


def _make_hybrid(run_date: str) -> dict:
    """Minimal hybrid_analysis JSON that ``_map_hybrid_to_payload`` can process."""
    return {
        "hybrid_top3": [
            {
                "ticker": "AAPL",
                "composite_score": 4.5,
                "sources": ["weekly"],
                "current_price": 150.0,
                "target": {},
                "rank": 1,
            }
        ],
        "weighted_picks": [],
        "summary": {
            "weekly_top5_count": 5,
            "pro30_candidates_count": 30,
            "movers_count": 0,
        },
    }


# ── _is_newer_run_date tests ────────────────────────────────────────────

class TestIsNewerRunDate:
    def test_newer_date_returns_true(self):
        assert _is_newer_run_date("2026-02-20", "2026-02-19") is True

    def test_older_date_returns_false(self):
        assert _is_newer_run_date("2026-02-18", "2026-02-19") is False

    def test_same_date_returns_false(self):
        assert _is_newer_run_date("2026-02-19", "2026-02-19") is False

    def test_candidate_none_returns_false(self):
        assert _is_newer_run_date(None, "2026-02-19") is False

    def test_current_none_returns_true(self):
        assert _is_newer_run_date("2026-02-19", None) is True

    def test_both_none_returns_false(self):
        assert _is_newer_run_date(None, None) is False

    def test_iso_timestamp_parsed(self):
        """ISO timestamps with T-suffix should be parsed correctly."""
        assert _is_newer_run_date("2026-02-20T14:30:00Z", "2026-02-19") is True


# ── Fallback chain tests ────────────────────────────────────────────────

def _reset_latest_result():
    """Clear the module-level _latest_result so tests start with clean state."""
    _mod._latest_result = None


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset module-level state between tests."""
    _reset_latest_result()
    yield
    _reset_latest_result()


@pytest.mark.asyncio
async def test_github_promoted_over_stale_disk():
    """When GitHub has a newer date than disk, result should use GitHub data."""
    disk_payload = _make_payload("2026-02-18")
    gh_payload = _make_payload("2026-02-20")
    # GitHub returns a mapped payload from _load_latest_from_github
    gh_payload["picks"][0]["ticker"] = "MSFT"  # distinguish from disk

    with (
        patch(f"{_MODULE}._load_latest_from_disk", return_value=disk_payload),
        patch(f"{_MODULE}._load_latest_from_github", return_value=gh_payload),
        patch(f"{_MODULE}._save_latest_to_disk") as mock_save,
    ):
        result = await get_engine_results()

    assert result["run_date"] == "2026-02-20"
    assert result["picks"][0]["ticker"] == "MSFT"
    mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_disk_kept_when_github_older():
    """When disk has fresher data than GitHub, result should keep disk data."""
    disk_payload = _make_payload("2026-02-20")
    gh_payload = _make_payload("2026-02-18")

    with (
        patch(f"{_MODULE}._load_latest_from_disk", return_value=disk_payload),
        patch(f"{_MODULE}._load_latest_from_github", return_value=gh_payload),
        patch(f"{_MODULE}._save_latest_to_disk") as mock_save,
    ):
        result = await get_engine_results()

    assert result["run_date"] == "2026-02-20"
    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_github_fallback_when_disk_empty():
    """When disk cache is empty, GitHub result should be used."""
    gh_payload = _make_payload("2026-02-19")

    with (
        patch(f"{_MODULE}._load_latest_from_disk", return_value=None),
        patch(f"{_MODULE}._load_latest_from_github", return_value=gh_payload),
        patch(f"{_MODULE}._save_latest_to_disk") as mock_save,
    ):
        result = await get_engine_results()

    assert result["run_date"] == "2026-02-19"
    mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_filesystem_fallback_when_both_empty(tmp_path: Path):
    """When disk and GitHub are empty, local filesystem outputs/ is used."""
    run_date = "2026-02-17"
    outputs_dir = tmp_path / "outputs" / run_date
    outputs_dir.mkdir(parents=True)
    hybrid_path = outputs_dir / f"hybrid_analysis_{run_date}.json"
    hybrid_path.write_text(json.dumps(_make_hybrid(run_date)))

    with (
        patch(f"{_MODULE}._load_latest_from_disk", return_value=None),
        patch(f"{_MODULE}._load_latest_from_github", return_value=None),
        patch(f"{_MODULE}.Path", side_effect=lambda p: tmp_path / p if p == "outputs" else Path(p)),
    ):
        result = await get_engine_results()

    assert result["run_date"] == run_date
    assert result["engine_name"] == "koocore_d"


@pytest.mark.asyncio
async def test_404_when_all_sources_empty(tmp_path: Path):
    """When no data source has results, raise 404."""
    from fastapi import HTTPException

    empty_dir = tmp_path / "outputs"
    empty_dir.mkdir()

    with (
        patch(f"{_MODULE}._load_latest_from_disk", return_value=None),
        patch(f"{_MODULE}._load_latest_from_github", return_value=None),
        patch(f"{_MODULE}.Path", side_effect=lambda p: tmp_path / p if p == "outputs" else Path(p)),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await get_engine_results()
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_github_failure_does_not_crash():
    """GitHub API returning error should not crash — gracefully use disk data."""
    disk_payload = _make_payload("2026-02-18")

    with (
        patch(f"{_MODULE}._load_latest_from_disk", return_value=disk_payload),
        patch(f"{_MODULE}._load_latest_from_github", return_value=None),
    ):
        result = await get_engine_results()

    assert result["run_date"] == "2026-02-18"
