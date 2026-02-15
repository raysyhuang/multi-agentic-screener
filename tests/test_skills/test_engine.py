"""Tests for skill engine — YAML loading, preconditions, execution."""

import pytest

from src.agents.tools import ToolRegistry, ToolDefinition, build_default_registry
from src.skills.engine import (
    SkillEngine,
    SkillDefinition,
    SkillStep,
    SkillValidation,
    SkillResult,
    _render_template,
    _render_dict,
)


async def _mock_handler(**kwargs):
    return {"status": "ok", "args": kwargs}


def _make_registry() -> ToolRegistry:
    """Create a registry with mock tools."""
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="check_earnings_date",
        description="Check earnings",
        parameters={"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
        handler=_mock_handler,
    ))
    registry.register(ToolDefinition(
        name="lookup_price_history",
        description="Price history",
        parameters={"type": "object", "properties": {"ticker": {"type": "string"}, "days": {"type": "integer"}}, "required": ["ticker"]},
        handler=_mock_handler,
    ))
    registry.register(ToolDefinition(
        name="get_sector_exposure",
        description="Sector exposure",
        parameters={"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
        handler=_mock_handler,
    ))
    return registry


def test_yaml_loading():
    """Verify skill definitions load from YAML files."""
    registry = build_default_registry()
    engine = SkillEngine(registry)
    assert len(engine.skills) == 3
    names = [s.name for s in engine.skills]
    assert "pre_earnings" in names
    assert "high_volatility" in names
    assert "sector_rotation" in names


def test_precondition_matching_bool():
    """Test boolean precondition matching."""
    registry = _make_registry()
    engine = SkillEngine(registry)

    # Pre-earnings matches when has_earnings_soon is True
    context = {"has_earnings_soon": True}
    applicable = engine.find_applicable_skills(context)
    assert any(s.name == "pre_earnings" for s in applicable)

    # Does not match when False
    context_no = {"has_earnings_soon": False}
    applicable_no = engine.find_applicable_skills(context_no)
    assert not any(s.name == "pre_earnings" for s in applicable_no)


def test_precondition_matching_in_list():
    """Test 'in' list precondition matching."""
    registry = _make_registry()
    engine = SkillEngine(registry)

    # High volatility matches bear or choppy
    context_bear = {"regime": "bear"}
    applicable = engine.find_applicable_skills(context_bear)
    assert any(s.name == "high_volatility" for s in applicable)

    context_bull = {"regime": "bull"}
    applicable_bull = engine.find_applicable_skills(context_bull)
    assert not any(s.name == "high_volatility" for s in applicable_bull)


def test_precondition_missing_key():
    """Missing keys in context should not match."""
    registry = _make_registry()
    engine = SkillEngine(registry)

    context_empty = {}
    applicable = engine.find_applicable_skills(context_empty)
    assert len(applicable) == 0


def test_template_rendering():
    assert _render_template("Hello {{name}}", {"name": "World"}) == "Hello World"
    assert _render_template("{{ticker}} is {{status}}", {"ticker": "AAPL", "status": "bullish"}) == "AAPL is bullish"
    # Unknown variables left as-is
    assert _render_template("{{unknown}}", {}) == "{{unknown}}"


def test_render_dict():
    result = _render_dict(
        {"ticker": "{{ticker}}", "days": 20},
        {"ticker": "AAPL"},
    )
    assert result["ticker"] == "AAPL"
    assert result["days"] == 20


@pytest.mark.asyncio
async def test_skill_execution():
    """Test executing a skill with tool calls and prompt addons."""
    registry = _make_registry()
    skill = SkillDefinition(
        name="test_skill",
        description="A test skill",
        preconditions={"has_earnings_soon": True},
        steps=[
            SkillStep(
                action="tool_call",
                tool="check_earnings_date",
                arguments={"ticker": "{{ticker}}"},
            ),
            SkillStep(
                action="prompt_addon",
                prompt_text="Warning: {{ticker}} has earnings soon.",
            ),
        ],
        validations=[
            SkillValidation(check="has_tool_results", description="Tools executed"),
        ],
        prompt_addons=["Cap position at 5%."],
    )

    engine = SkillEngine(registry)
    result = await engine.execute_skill(skill, {"ticker": "AAPL"})

    assert result.executed
    assert len(result.tool_results) == 1
    assert result.tool_results[0]["tool"] == "check_earnings_date"
    assert result.tool_results[0]["arguments"]["ticker"] == "AAPL"
    assert len(result.prompt_addons) == 2  # step addon + skill addon
    assert "AAPL" in result.prompt_addons[0]
    assert "Cap position" in result.prompt_addons[1]
    assert len(result.validation_results) == 1
    assert result.validation_results[0]["passed"] is True


@pytest.mark.asyncio
async def test_skill_execution_with_yaml():
    """Test executing a real YAML-loaded skill."""
    registry = _make_registry()
    engine = SkillEngine(registry)

    pre_earnings = next(s for s in engine.skills if s.name == "pre_earnings")
    result = await engine.execute_skill(pre_earnings, {"ticker": "TSLA"})

    assert result.executed
    assert result.skill_name == "pre_earnings"
    assert len(result.tool_results) >= 1
    assert any("TSLA" in str(addon) for addon in result.prompt_addons)
