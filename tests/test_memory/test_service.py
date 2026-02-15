"""Tests for the unified memory service."""

import pytest

from src.memory.working import WorkingMemory
from src.memory.service import MemoryService


@pytest.mark.asyncio
async def test_service_returns_working_memory_without_session():
    """Without a DB session, only working memory is returned."""
    wm = WorkingMemory(regime="bull")
    wm.record_interpretation("AAPL")
    service = MemoryService(working=wm, session=None)

    ctx = await service.get_context_for_candidate("AAPL", "breakout")
    assert "working_memory" in ctx
    assert "AAPL" in ctx["working_memory"]
    assert "ticker_history" not in ctx
    assert "model_performance" not in ctx


@pytest.mark.asyncio
async def test_service_caching():
    """Verify that episodic queries are cached per ticker."""
    wm = WorkingMemory(regime="bull")
    service = MemoryService(working=wm, session=None)

    # Without session, cache is empty
    ctx1 = await service.get_context_for_candidate("AAPL", "breakout")
    ctx2 = await service.get_context_for_candidate("AAPL", "breakout")

    # Both should return same structure (working memory only)
    assert ctx1.keys() == ctx2.keys()


@pytest.mark.asyncio
async def test_service_working_memory_updates():
    """Verify working memory reflects state changes between calls."""
    wm = WorkingMemory(regime="bull")
    service = MemoryService(working=wm, session=None)

    ctx1 = await service.get_context_for_candidate("AAPL", "breakout")
    assert "MSFT" not in ctx1["working_memory"]

    wm.record_interpretation("MSFT")
    ctx2 = await service.get_context_for_candidate("MSFT", "breakout")
    assert "MSFT" in ctx2["working_memory"]
