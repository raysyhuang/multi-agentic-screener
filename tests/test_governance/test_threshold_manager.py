"""Tests for the Threshold Manager (PR6)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.base import ThresholdAdjustment
from src.governance.threshold_manager import (
    ADJUSTABLE_PARAMS,
    MAX_CHANGE_PCT,
    MIN_SAMPLE_SIZE,
    validate_adjustment,
    process_adjustments,
    get_current_thresholds,
    load_snapshot,
    get_snapshot_history,
)


def _make_adj(**overrides) -> ThresholdAdjustment:
    defaults = {
        "parameter": "vix_high_threshold",
        "current_value": 25.0,
        "suggested_value": 27.0,
        "reasoning": "Higher VIX threshold improves win rate",
        "confidence": 75.0,
        "evidence_sample_size": 30,
    }
    defaults.update(overrides)
    return ThresholdAdjustment(**defaults)


class TestValidateAdjustment:
    def test_valid_adjustment_passes(self):
        adj = _make_adj()
        current = get_current_thresholds()
        result = validate_adjustment(adj, current)
        assert result is None  # None = valid

    def test_rejects_non_whitelisted_param(self):
        adj = _make_adj(parameter="secret_key")
        result = validate_adjustment(adj, get_current_thresholds())
        assert result is not None
        assert "whitelist" in result.lower()

    def test_rejects_insufficient_sample(self):
        adj = _make_adj(evidence_sample_size=5)
        result = validate_adjustment(adj, get_current_thresholds())
        assert result is not None
        assert "evidence" in result.lower() or "sample" in result.lower()

    def test_rejects_excessive_change(self):
        # 25.0 → 40.0 = 60% change, exceeds 20% max
        adj = _make_adj(current_value=25.0, suggested_value=40.0)
        result = validate_adjustment(adj, get_current_thresholds())
        assert result is not None
        assert "large" in result.lower() or "change" in result.lower()

    def test_allows_change_within_limit(self):
        # 25.0 → 28.0 = 12% change, within 20% max
        adj = _make_adj(current_value=25.0, suggested_value=28.0)
        result = validate_adjustment(adj, get_current_thresholds())
        assert result is None

    def test_rejects_unknown_param(self):
        adj = _make_adj(parameter="nonexistent_param")
        result = validate_adjustment(adj, {"other_param": 1.0})
        assert result is not None


class TestProcessAdjustments:
    def test_dry_run_does_not_apply(self):
        adj = _make_adj()
        with patch("src.governance.threshold_manager.SNAPSHOT_DIR", Path(tempfile.mkdtemp())):
            result = process_adjustments([adj], run_date="2025-01-01", dry_run=True)

        assert result.dry_run is True
        assert len(result.applied) == 1
        assert len(result.rejected) == 0
        assert result.snapshot.dry_run is True

    def test_rejects_bad_adjustments(self):
        good = _make_adj()
        bad = _make_adj(parameter="not_whitelisted")

        with patch("src.governance.threshold_manager.SNAPSHOT_DIR", Path(tempfile.mkdtemp())):
            result = process_adjustments([good, bad], run_date="2025-01-01")

        assert len(result.applied) == 1
        assert len(result.rejected) == 1
        assert result.rejected[0][0].parameter == "not_whitelisted"

    def test_snapshot_saved(self):
        adj = _make_adj()
        tmp_dir = Path(tempfile.mkdtemp())
        with patch("src.governance.threshold_manager.SNAPSHOT_DIR", tmp_dir):
            process_adjustments([adj], run_date="2025-03-15")

        snapshot_file = tmp_dir / "snapshot_2025-03-15.json"
        assert snapshot_file.is_file()
        data = json.loads(snapshot_file.read_text())
        assert data["run_date"] == "2025-03-15"
        assert len(data["adjustments_applied"]) == 1


class TestSnapshotManagement:
    def test_load_snapshot(self):
        tmp_dir = Path(tempfile.mkdtemp())
        adj = _make_adj()
        with patch("src.governance.threshold_manager.SNAPSHOT_DIR", tmp_dir):
            process_adjustments([adj], run_date="2025-06-01")
            snapshot = load_snapshot("2025-06-01")

        assert snapshot is not None
        assert snapshot.run_date == "2025-06-01"

    def test_load_nonexistent_returns_none(self):
        tmp_dir = Path(tempfile.mkdtemp())
        with patch("src.governance.threshold_manager.SNAPSHOT_DIR", tmp_dir):
            assert load_snapshot("1999-01-01") is None

    def test_snapshot_history(self):
        tmp_dir = Path(tempfile.mkdtemp())
        with patch("src.governance.threshold_manager.SNAPSHOT_DIR", tmp_dir):
            process_adjustments([_make_adj()], run_date="2025-01-01")
            process_adjustments([_make_adj()], run_date="2025-01-02")
            history = get_snapshot_history()

        assert len(history) == 2
        assert history[0]["run_date"] == "2025-01-02"  # Most recent first


class TestAdjustableParams:
    def test_whitelist_contains_expected(self):
        assert "vix_high_threshold" in ADJUSTABLE_PARAMS
        assert "vix_low_threshold" in ADJUSTABLE_PARAMS
        assert "breadth_bullish_threshold" in ADJUSTABLE_PARAMS
        assert "min_price" in ADJUSTABLE_PARAMS
        assert "min_avg_daily_volume" in ADJUSTABLE_PARAMS

    def test_whitelist_excludes_dangerous(self):
        assert "anthropic_api_key" not in ADJUSTABLE_PARAMS
        assert "database_url" not in ADJUSTABLE_PARAMS
        assert "telegram_bot_token" not in ADJUSTABLE_PARAMS
        assert "max_run_cost_usd" not in ADJUSTABLE_PARAMS
