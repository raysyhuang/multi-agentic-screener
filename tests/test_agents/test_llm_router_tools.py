"""Tests for LLM router tool-use integration (mocked API calls)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.tools import ToolRegistry, ToolDefinition


async def _echo_tool(**kwargs):
    return {"result": kwargs}


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters={
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
        handler=_echo_tool,
    ))
    return registry


@pytest.mark.asyncio
async def test_call_llm_with_tools_anthropic_no_tool_use():
    """Test Anthropic path when LLM responds with text (no tool use)."""
    from src.agents.llm_router import call_llm_with_tools

    # Mock the Anthropic client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text='{"decision": "APPROVE"}')]
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client

        result = await call_llm_with_tools(
            model="claude-sonnet-4-5-20250929",
            system_prompt="You are a test agent.",
            user_prompt="Test prompt",
            tools=_make_registry(),
        )

    assert result["content"] == {"decision": "APPROVE"}
    assert result["tool_calls"] == []
    assert result["tokens_in"] == 100
    assert result["tokens_out"] == 50


@pytest.mark.asyncio
async def test_call_llm_with_tools_openai_no_tool_use():
    """Test OpenAI path when LLM responds with text (no tool use)."""
    from src.agents.llm_router import call_llm_with_tools

    mock_message = MagicMock()
    mock_message.content = '{"decision": "VETO"}'
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=80, completion_tokens=40)

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client

        result = await call_llm_with_tools(
            model="gpt-5.2",
            system_prompt="You are a test agent.",
            user_prompt="Test prompt",
            tools=_make_registry(),
        )

    assert result["content"] == {"decision": "VETO"}
    assert result["tool_calls"] == []
