"""Tests for validation history loading in the main pipeline."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import src.main as main
import src.output.performance as performance


@pytest.mark.asyncio
async def test_load_validation_history_cards_forwards_execution_mode(monkeypatch):
    """Validation history must be scoped to the current execution mode."""

    captured: dict[str, object] = {}

    async def _fake_build_validation_card_from_history(*, days, execution_mode):
        captured["days"] = days
        captured["execution_mode"] = execution_mode
        return {}

    monkeypatch.setattr(
        performance,
        "build_validation_card_from_history",
        _fake_build_validation_card_from_history,
    )

    result = await main._load_validation_history_cards("hybrid")

    assert result == {}
    assert captured == {"days": 90, "execution_mode": "hybrid"}
