"""Planner agent — decomposes daily goals into execution plans.

Uses GPT-4.1-mini (cheap) to create a task graph. Falls back to the
default linear pipeline plan if LLM fails — guarantees no regression.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum

from src.agents.base import BaseAgent
from src.agents.llm_router import call_llm
from src.config import get_settings

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a trading pipeline planner. Given a set of candidates and market context, create an execution plan.

The plan should specify which candidates to process and in what order. You can:
- INTERPRET: Run signal interpretation on a candidate
- DEBATE: Run adversarial debate on an interpreted candidate
- RISK_CHECK: Run risk gate evaluation on a debated candidate
- SKILL: Run a specialized skill (e.g., pre-earnings analysis)
- VERIFY: Verify the final output

Respond with JSON:
{
  "goal": "Brief description of today's goal",
  "acceptance_criteria": ["criterion 1", "criterion 2"],
  "max_cost_usd": float,
  "steps": [
    {
      "step_id": "step_1",
      "action": "INTERPRET | DEBATE | RISK_CHECK | SKILL | VERIFY",
      "target": "ticker or 'all'",
      "depends_on": [],
      "notes": "optional notes"
    }
  ]
}"""


class PlanAction(str, Enum):
    INTERPRET = "INTERPRET"
    DEBATE = "DEBATE"
    RISK_CHECK = "RISK_CHECK"
    SKILL = "SKILL"
    VERIFY = "VERIFY"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in the execution plan."""
    step_id: str
    action: str  # PlanAction value
    target: str
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    notes: str = ""
    result: dict | None = None


@dataclass
class ExecutionPlan:
    """A complete execution plan for the pipeline run."""
    goal: str
    steps: list[PlanStep]
    acceptance_criteria: list[str] = field(default_factory=list)
    max_cost_usd: float = 0.50
    is_default: bool = False

    def get_ready_steps(self) -> list[PlanStep]:
        """Get steps that are ready to execute (dependencies met)."""
        completed_ids = {s.step_id for s in self.steps if s.status == StepStatus.COMPLETED}
        ready = []
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            deps_met = all(dep in completed_ids for dep in step.depends_on)
            if deps_met:
                ready.append(step)
        return ready

    def mark_completed(self, step_id: str, result: dict | None = None) -> None:
        for step in self.steps:
            if step.step_id == step_id:
                step.status = StepStatus.COMPLETED
                step.result = result
                return

    def mark_failed(self, step_id: str) -> None:
        for step in self.steps:
            if step.step_id == step_id:
                step.status = StepStatus.FAILED
                return

    @property
    def all_completed(self) -> bool:
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED)
            for s in self.steps
        )


class PlannerAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        model = getattr(settings, "planner_model", "gpt-5.2")
        super().__init__("planner", model)

    async def create_plan(
        self,
        candidates: list,
        regime_context: dict,
        working_memory_summary: str = "",
    ) -> ExecutionPlan:
        """Create an execution plan for the pipeline run.

        Falls back to default linear plan on any failure.
        """
        try:
            return await self._llm_plan(candidates, regime_context, working_memory_summary)
        except Exception as e:
            logger.warning("Planner LLM failed, using default plan: %s", e)
            return self._default_plan(candidates)

    async def _llm_plan(
        self,
        candidates: list,
        regime_context: dict,
        working_memory_summary: str,
    ) -> ExecutionPlan:
        """Generate plan via LLM."""
        tickers = [getattr(c, "ticker", str(c)) for c in candidates[:10]]
        user_prompt = json.dumps({
            "candidates": tickers,
            "regime": regime_context,
            "working_memory": working_memory_summary,
            "num_candidates": len(candidates),
        }, default=str)

        result = await call_llm(
            model=self.model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=f"Create an execution plan for today's pipeline.\n\n{user_prompt}",
            max_tokens=1500,
            temperature=0.3,
        )

        self._store_meta(result)
        content = result["content"]

        if isinstance(content, str):
            logger.warning("Planner returned non-JSON, using default plan")
            return self._default_plan(candidates)

        return self._parse_plan(content, candidates)

    def _parse_plan(self, content: dict, candidates: list) -> ExecutionPlan:
        """Parse LLM output into an ExecutionPlan."""
        steps = []
        for step_data in content.get("steps", []):
            steps.append(PlanStep(
                step_id=step_data.get("step_id", f"step_{len(steps)}"),
                action=step_data.get("action", "INTERPRET"),
                target=step_data.get("target", "all"),
                depends_on=step_data.get("depends_on", []),
                notes=step_data.get("notes", ""),
            ))

        if not steps:
            return self._default_plan(candidates)

        return ExecutionPlan(
            goal=content.get("goal", "Daily signal pipeline"),
            steps=steps,
            acceptance_criteria=content.get("acceptance_criteria", []),
            max_cost_usd=content.get("max_cost_usd", 0.50),
        )

    def _default_plan(self, candidates: list) -> ExecutionPlan:
        """Create the default linear pipeline plan (mirrors current behavior)."""
        tickers = [getattr(c, "ticker", str(c)) for c in candidates[:10]]
        steps = []

        # Step 1: Interpret all candidates
        for i, ticker in enumerate(tickers):
            steps.append(PlanStep(
                step_id=f"interpret_{ticker}",
                action=PlanAction.INTERPRET.value,
                target=ticker,
            ))

        # Step 2: Debate top candidates (depends on all interpretations)
        interpret_ids = [s.step_id for s in steps]
        for ticker in tickers[:5]:
            steps.append(PlanStep(
                step_id=f"debate_{ticker}",
                action=PlanAction.DEBATE.value,
                target=ticker,
                depends_on=[f"interpret_{ticker}"],
            ))

        # Step 3: Risk check survivors (depends on debates)
        for ticker in tickers[:5]:
            steps.append(PlanStep(
                step_id=f"risk_{ticker}",
                action=PlanAction.RISK_CHECK.value,
                target=ticker,
                depends_on=[f"debate_{ticker}"],
            ))

        # Step 4: Verify
        risk_ids = [f"risk_{t}" for t in tickers[:5]]
        steps.append(PlanStep(
            step_id="verify",
            action=PlanAction.VERIFY.value,
            target="all",
            depends_on=risk_ids,
        ))

        return ExecutionPlan(
            goal="Default linear pipeline",
            steps=steps,
            acceptance_criteria=["At least 1 candidate processed", "No unhandled errors"],
            max_cost_usd=0.50,
            is_default=True,
        )

    def _build_system_prompt(self) -> str:
        return PLANNER_SYSTEM_PROMPT

    def _build_user_prompt(self, **kwargs) -> str:
        return ""
