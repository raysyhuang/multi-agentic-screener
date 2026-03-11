"""Tests for pipeline memory cleanup — aggregator teardown on exception paths."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_aggregator_closed_on_pipeline_exception():
    """If _run_pipeline_core() crashes, run_morning_pipeline() must still close the aggregator."""
    mock_aggregator = MagicMock()
    mock_aggregator.close = MagicMock()

    # Patch _run_pipeline_core to stash a mock aggregator then raise
    original_state = {}

    async def fake_pipeline_core(today, settings, run_id, start_time, _state=None):
        if _state is not None:
            _state["aggregator"] = mock_aggregator
            original_state.update(_state)
        raise RuntimeError("simulated pipeline crash")

    with (
        patch("src.main._run_pipeline_core", side_effect=fake_pipeline_core),
        patch("src.main._trading_date_et", return_value="2026-03-11"),
        patch("src.main.get_settings") as mock_settings,
        patch("src.main.get_session"),
        patch("src.main.send_alert", new_callable=AsyncMock),
        patch("src.main.select"),
        patch("src.main._log_memory"),
    ):
        mock_settings.return_value = MagicMock(
            trading_mode="PAPER",
            execution_mode="quant_only",
        )
        from src.main import run_morning_pipeline
        await run_morning_pipeline()

    # The aggregator must have been closed by the finally block
    mock_aggregator.close.assert_called_once()


@pytest.mark.asyncio
async def test_aggregator_not_double_closed_on_success():
    """If success path already closed the aggregator, finally must not fail."""
    mock_aggregator = MagicMock()
    mock_aggregator.close = MagicMock()

    async def fake_pipeline_core(today, settings, run_id, start_time, _state=None):
        if _state is not None:
            # Simulate success-path cleanup: stash then clear
            _state["aggregator"] = mock_aggregator
            mock_aggregator.close()
            _state["aggregator"] = None

    with (
        patch("src.main._run_pipeline_core", side_effect=fake_pipeline_core),
        patch("src.main._trading_date_et", return_value="2026-03-11"),
        patch("src.main.get_settings") as mock_settings,
        patch("src.main._log_memory"),
    ):
        mock_settings.return_value = MagicMock(
            trading_mode="PAPER",
            execution_mode="quant_only",
        )
        from src.main import run_morning_pipeline
        await run_morning_pipeline()

    # close() called once by success path, NOT again by finally
    mock_aggregator.close.assert_called_once()
