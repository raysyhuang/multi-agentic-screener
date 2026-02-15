"""Tests for tool definitions and registry."""

import pytest

from src.agents.tools import (
    ToolDefinition,
    ToolRegistry,
    build_default_registry,
)


async def _echo_handler(**kwargs):
    return {"echo": kwargs}


async def _failing_handler(**kwargs):
    raise ValueError("intentional test failure")


def _make_tool(name: str = "test_tool", handler=None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Test tool: {name}",
        parameters={
            "type": "object",
            "properties": {"arg1": {"type": "string"}},
            "required": ["arg1"],
        },
        handler=handler or _echo_handler,
    )


def test_registry_register_and_get():
    registry = ToolRegistry()
    tool = _make_tool("my_tool")
    registry.register(tool)

    assert registry.get("my_tool") is tool
    assert registry.get("nonexistent") is None


def test_registry_list_names():
    registry = ToolRegistry()
    registry.register(_make_tool("tool_a"))
    registry.register(_make_tool("tool_b"))

    names = registry.list_names()
    assert "tool_a" in names
    assert "tool_b" in names


@pytest.mark.asyncio
async def test_registry_execute_success():
    registry = ToolRegistry()
    registry.register(_make_tool("echo"))

    result = await registry.execute("echo", {"arg1": "hello"})
    assert result == {"echo": {"arg1": "hello"}}


@pytest.mark.asyncio
async def test_registry_execute_unknown_tool():
    registry = ToolRegistry()
    result = await registry.execute("unknown", {})
    assert "error" in result


@pytest.mark.asyncio
async def test_registry_execute_handler_error():
    registry = ToolRegistry()
    registry.register(_make_tool("failing", handler=_failing_handler))

    result = await registry.execute("failing", {"arg1": "x"})
    assert "error" in result
    assert "intentional" in result["error"]


def test_list_schemas_anthropic():
    registry = ToolRegistry()
    registry.register(_make_tool("my_tool"))

    schemas = registry.list_schemas_anthropic()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "my_tool"
    assert "input_schema" in schemas[0]
    assert schemas[0]["input_schema"]["type"] == "object"


def test_list_schemas_openai():
    registry = ToolRegistry()
    registry.register(_make_tool("my_tool"))

    schemas = registry.list_schemas_openai()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "my_tool"
    assert "parameters" in schemas[0]["function"]


def test_default_registry_has_all_tools():
    registry = build_default_registry()
    names = registry.list_names()

    assert "lookup_price_history" in names
    assert "lookup_ticker_outcomes" in names
    assert "get_model_stats" in names
    assert "check_earnings_date" in names
    assert "get_sector_exposure" in names
    assert len(names) == 5


@pytest.mark.asyncio
async def test_default_tools_return_placeholder():
    """Built-in tools return placeholder data without a DB session."""
    registry = build_default_registry()

    result = await registry.execute("lookup_price_history", {"ticker": "AAPL"})
    assert result["ticker"] == "AAPL"

    result = await registry.execute("check_earnings_date", {"ticker": "MSFT"})
    assert result["ticker"] == "MSFT"
