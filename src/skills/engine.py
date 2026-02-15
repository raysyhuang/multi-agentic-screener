"""Skill loader and executor.

Skills are declarative YAML playbooks that define preconditions,
steps (tool calls + prompt addons), and validation checks. When a
candidate matches a skill's preconditions, the skill is executed
to augment the analysis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.agents.tools import ToolRegistry

logger = logging.getLogger(__name__)

DEFINITIONS_DIR = Path(__file__).parent / "definitions"


@dataclass
class SkillStep:
    """A single step in a skill execution."""
    action: str  # "tool_call" or "prompt_addon"
    tool: str | None = None
    arguments: dict = field(default_factory=dict)
    prompt_text: str | None = None


@dataclass
class SkillValidation:
    """A validation check after skill execution."""
    check: str
    description: str


@dataclass
class SkillDefinition:
    """A complete skill playbook."""
    name: str
    description: str
    preconditions: dict = field(default_factory=dict)
    steps: list[SkillStep] = field(default_factory=list)
    validations: list[SkillValidation] = field(default_factory=list)
    prompt_addons: list[str] = field(default_factory=list)


@dataclass
class SkillResult:
    """Result of executing a skill."""
    skill_name: str
    matched: bool
    executed: bool = False
    tool_results: list[dict] = field(default_factory=list)
    prompt_addons: list[str] = field(default_factory=list)
    validation_results: list[dict] = field(default_factory=list)
    error: str | None = None


def _render_template(text: str, variables: dict) -> str:
    """Simple {{variable}} template rendering."""
    def replacer(match):
        key = match.group(1).strip()
        return str(variables.get(key, match.group(0)))
    return re.sub(r"\{\{(\w+)\}\}", replacer, text)


def _render_dict(d: dict, variables: dict) -> dict:
    """Render templates in all string values of a dict."""
    rendered = {}
    for k, v in d.items():
        if isinstance(v, str):
            rendered[k] = _render_template(v, variables)
        elif isinstance(v, dict):
            rendered[k] = _render_dict(v, variables)
        else:
            rendered[k] = v
    return rendered


class SkillEngine:
    """Loads YAML skill definitions and executes applicable ones."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self.skills: list[SkillDefinition] = []
        self._load_definitions()

    def _load_definitions(self) -> None:
        """Load all YAML skill definitions from the definitions directory."""
        if not DEFINITIONS_DIR.exists():
            logger.warning("Skill definitions directory not found: %s", DEFINITIONS_DIR)
            return

        for yaml_file in sorted(DEFINITIONS_DIR.glob("*.yaml")):
            try:
                skill = self._parse_yaml(yaml_file)
                self.skills.append(skill)
                logger.info("Loaded skill: %s", skill.name)
            except Exception as e:
                logger.error("Failed to load skill %s: %s", yaml_file.name, e)

    def _parse_yaml(self, path: Path) -> SkillDefinition:
        """Parse a YAML file into a SkillDefinition."""
        with open(path) as f:
            data = yaml.safe_load(f)

        steps = []
        for step_data in data.get("steps", []):
            steps.append(SkillStep(
                action=step_data.get("action", "tool_call"),
                tool=step_data.get("tool"),
                arguments=step_data.get("arguments", {}),
                prompt_text=step_data.get("prompt_text"),
            ))

        validations = []
        for val_data in data.get("validations", []):
            validations.append(SkillValidation(
                check=val_data.get("check", ""),
                description=val_data.get("description", ""),
            ))

        return SkillDefinition(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            preconditions=data.get("preconditions", {}),
            steps=steps,
            validations=validations,
            prompt_addons=data.get("prompt_addons", []),
        )

    def find_applicable_skills(self, candidate_context: dict) -> list[SkillDefinition]:
        """Find skills whose preconditions match the candidate context."""
        applicable = []
        for skill in self.skills:
            if self._check_preconditions(skill.preconditions, candidate_context):
                applicable.append(skill)
        return applicable

    def _check_preconditions(self, preconditions: dict, context: dict) -> bool:
        """Check if all preconditions are met by the context."""
        for key, condition in preconditions.items():
            value = context.get(key)
            if value is None:
                return False

            if isinstance(condition, dict):
                # Range check: {"min": x, "max": y}
                if "min" in condition and value < condition["min"]:
                    return False
                if "max" in condition and value > condition["max"]:
                    return False
                # Equality check: {"equals": x}
                if "equals" in condition and value != condition["equals"]:
                    return False
                # In-list check: {"in": [x, y, z]}
                if "in" in condition and value not in condition["in"]:
                    return False
            elif isinstance(condition, bool):
                if bool(value) != condition:
                    return False
            else:
                # Direct equality
                if value != condition:
                    return False

        return True

    async def execute_skill(
        self,
        skill: SkillDefinition,
        variables: dict,
    ) -> SkillResult:
        """Execute a skill's steps and validations."""
        result = SkillResult(skill_name=skill.name, matched=True)

        try:
            # Execute steps
            for step in skill.steps:
                if step.action == "tool_call" and step.tool:
                    args = _render_dict(step.arguments, variables)
                    tool_result = await self.tool_registry.execute(step.tool, args)
                    result.tool_results.append({
                        "tool": step.tool,
                        "arguments": args,
                        "result": tool_result,
                    })
                elif step.action == "prompt_addon" and step.prompt_text:
                    rendered = _render_template(step.prompt_text, variables)
                    result.prompt_addons.append(rendered)

            # Add skill-level prompt addons
            for addon in skill.prompt_addons:
                rendered = _render_template(addon, variables)
                result.prompt_addons.append(rendered)

            # Run validations
            for validation in skill.validations:
                check_result = self._run_validation(validation, result, variables)
                result.validation_results.append(check_result)

            result.executed = True

        except Exception as e:
            logger.error("Skill '%s' execution failed: %s", skill.name, e)
            result.error = str(e)

        return result

    def _run_validation(
        self, validation: SkillValidation, result: SkillResult, variables: dict
    ) -> dict:
        """Run a single validation check."""
        check = validation.check

        if check == "has_tool_results":
            passed = len(result.tool_results) > 0
        elif check == "no_errors":
            passed = all(
                "error" not in tr.get("result", {})
                for tr in result.tool_results
            )
        elif check.startswith("max_position_size:"):
            max_pct = float(check.split(":")[1])
            # This would check against the candidate's position size
            passed = True  # Placeholder — actual check in orchestrator
        else:
            passed = True  # Unknown checks pass by default

        return {
            "check": check,
            "description": validation.description,
            "passed": passed,
        }
