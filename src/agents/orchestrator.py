"""Agent orchestrator — autonomous runtime pipeline.

Flow:
  Planner → [Signal Interpreter → Skills → Adversarial Debate → Risk Gate] → Verifier
  With memory service providing context and optional retry on verifier flag.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date

from src.agents.signal_interpreter import SignalInterpreterAgent
from src.agents.adversarial import AdversarialAgent
from src.agents.risk_gate import RiskGateAgent
from src.agents.planner import PlannerAgent
from src.agents.verifier import VerifierAgent
from src.agents.base import (
    GateDecision,
    SignalInterpretation,
    DebateResult,
    RiskGateOutput,
    RiskFlag,
)
from src.agents.retry import RetryPolicy
from src.agents.tools import build_default_registry, build_live_registry
from src.config import get_settings
from src.memory.working import WorkingMemory
from src.memory.service import MemoryService
from src.skills.engine import SkillEngine
from src.signals.ranker import RankedCandidate

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    ticker: str
    signal_model: str
    direction: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float | None
    holding_period: int
    confidence: float
    interpretation: SignalInterpretation
    debate: DebateResult
    risk_gate: RiskGateOutput
    features: dict


@dataclass
class PipelineRun:
    run_date: date
    regime: str
    regime_details: dict
    candidates_scored: int
    interpreted: int
    debated: int
    approved: list[PipelineResult]
    vetoed: list[str]
    agent_logs: list[dict]
    convergence_state: str = "converged"


async def run_agent_pipeline(
    candidates: list[RankedCandidate],
    regime_context: dict,
    existing_positions: list[dict] | None = None,
    db_session=None,
    run_id: str | None = None,
) -> PipelineRun:
    """Run the autonomous agent pipeline: plan → execute → verify → retry.

    Args:
        candidates: Top N candidates from the ranker (typically 10)
        regime_context: Current regime assessment as dict
        existing_positions: Current portfolio positions for correlation check
        db_session: Optional async DB session for episodic memory queries
        run_id: Optional run ID for tracing
    """
    settings = get_settings()

    # --- Initialize autonomous runtime components ---
    working_memory = WorkingMemory(
        run_id=run_id or uuid.uuid4().hex[:12],
        regime=regime_context.get("regime", "unknown"),
    )
    memory_service = MemoryService(working_memory, session=db_session)
    if db_session:
        tool_registry = build_live_registry(db_session, existing_positions)
    else:
        tool_registry = build_default_registry()
    skill_engine = SkillEngine(tool_registry)

    # --- Step A: Planner creates execution plan ---
    planner = PlannerAgent()
    plan = await planner.create_plan(
        candidates, regime_context, working_memory.to_prompt_context()
    )
    logger.info(
        "Plan: goal='%s', steps=%d, is_default=%s",
        plan.goal, len(plan.steps), plan.is_default,
    )

    # --- Step B: Execute pipeline stages ---
    budget_usd = min(plan.max_cost_usd, settings.max_run_cost_usd)

    pipeline_run = await _execute_pipeline_stages(
        candidates=candidates,
        regime_context=regime_context,
        existing_positions=existing_positions,
        settings=settings,
        memory_service=memory_service,
        skill_engine=skill_engine,
        budget_usd=budget_usd,
    )

    # Record planner in agent logs
    pipeline_run.agent_logs.append({
        "agent": "planner",
        "plan_goal": plan.goal,
        "plan_steps": len(plan.steps),
        "is_default_plan": plan.is_default,
        **planner.last_call_meta,
    })

    # --- Step C: Verifier checks output ---
    verifier = VerifierAgent()
    pipeline_output_for_verify = {
        "approved": [
            {
                "ticker": p.ticker,
                "confidence": p.confidence,
                "signal_model": p.signal_model,
            }
            for p in pipeline_run.approved
        ],
        "vetoed": pipeline_run.vetoed,
        "num_approved": len(pipeline_run.approved),
        "num_vetoed": len(pipeline_run.vetoed),
    }
    verification = await verifier.verify(
        pipeline_output=pipeline_output_for_verify,
        acceptance_criteria=plan.acceptance_criteria,
        regime_context=regime_context,
    )

    pipeline_run.agent_logs.append({
        "agent": "verifier",
        "passed": verification.passed,
        "should_retry": verification.should_retry,
        "assessment": verification.overall_assessment,
        **verifier.last_call_meta,
    })

    # --- Step D: Verify-Redo-Converge loop ---
    # Preserve budget_exhausted from inner pipeline if it triggered
    convergence = pipeline_run.convergence_state

    for retry_round in range(settings.max_verifier_retries):
        if not verification.should_retry or not verification.retry_targets:
            break

        if working_memory.total_cost_usd >= budget_usd:
            convergence = "budget_exhausted"
            break

        # Inject verifier feedback into working memory for retry agents
        if verification.suggestions:
            working_memory.record_verifier_feedback(verification.suggestions)

        retry_tickers = set(verification.retry_targets)
        retry_candidates = [c for c in candidates if c.ticker in retry_tickers]
        if not retry_candidates:
            break

        logger.info(
            "Verifier retry round %d for: %s",
            retry_round + 1, list(retry_tickers),
        )

        retry_run = await _execute_pipeline_stages(
            candidates=retry_candidates,
            regime_context=regime_context,
            existing_positions=existing_positions,
            settings=settings,
            memory_service=memory_service,
            skill_engine=skill_engine,
            budget_usd=budget_usd,
        )

        # Merge retry results — only add new approvals
        existing_approved = {p.ticker for p in pipeline_run.approved}
        for pick in retry_run.approved:
            if (
                pick.ticker not in existing_approved
                and len(pipeline_run.approved) < settings.max_final_picks
            ):
                pipeline_run.approved.append(pick)
        pipeline_run.agent_logs.extend(retry_run.agent_logs)

        # Re-verify after retry
        pipeline_output_for_verify = {
            "approved": [
                {
                    "ticker": p.ticker,
                    "confidence": p.confidence,
                    "signal_model": p.signal_model,
                }
                for p in pipeline_run.approved
            ],
            "vetoed": pipeline_run.vetoed,
            "num_approved": len(pipeline_run.approved),
            "num_vetoed": len(pipeline_run.vetoed),
        }
        verification = await verifier.verify(
            pipeline_output=pipeline_output_for_verify,
            acceptance_criteria=plan.acceptance_criteria,
            regime_context=regime_context,
        )
        pipeline_run.agent_logs.append({
            "agent": "verifier",
            "retry_round": retry_round + 1,
            "passed": verification.passed,
            "should_retry": verification.should_retry,
            "assessment": verification.overall_assessment,
            **verifier.last_call_meta,
        })

    # Determine final convergence state (preserve budget_exhausted from inner pipeline)
    if convergence not in ("budget_exhausted",):
        if verification.should_retry:
            convergence = "max_retries"
        else:
            convergence = "converged"

    working_memory.set_convergence(convergence)
    pipeline_run.convergence_state = convergence

    logger.info(
        "Pipeline complete: %d approved, %d vetoed, verification=%s, convergence=%s",
        len(pipeline_run.approved), len(pipeline_run.vetoed),
        "PASS" if verification.passed else "FAIL",
        convergence,
    )

    return pipeline_run


async def _execute_pipeline_stages(
    candidates: list[RankedCandidate],
    regime_context: dict,
    existing_positions: list[dict] | None,
    settings,
    memory_service: MemoryService,
    skill_engine: SkillEngine,
    budget_usd: float | None = None,
) -> PipelineRun:
    """Execute the 3-stage pipeline: interpret → skills → debate → risk gate.

    Integrates memory context and skill execution at each stage.
    Checks cost budget between stages and stops early if exhausted.
    """
    interpreter = SignalInterpreterAgent()
    adversarial = AdversarialAgent()
    risk_gate = RiskGateAgent()

    agent_logs: list[dict] = []
    approved: list[PipelineResult] = []
    vetoed: list[str] = []

    # --- Stage 1: Signal Interpretation (top 10 → top 5) ---
    logger.info("Stage 1: Interpreting %d candidates", len(candidates))
    interpretations: list[tuple[RankedCandidate, SignalInterpretation]] = []

    for candidate in candidates[:settings.top_n_for_interpretation]:
        memory_ctx = await memory_service.get_context_for_candidate(
            candidate.ticker, candidate.signal_model,
        )

        retry_result = await interpreter.interpret(
            candidate, regime_context, memory_context=memory_ctx,
        )
        if retry_result.value:
            interpretations.append((candidate, retry_result.value))
            memory_service.working.record_interpretation(candidate.ticker)
            memory_service.working.add_cost(
                retry_result.total_tokens, retry_result.total_cost_usd,
            )
            agent_logs.append({
                "agent": "signal_interpreter",
                "ticker": candidate.ticker,
                "confidence": retry_result.value.confidence,
                "attempts": retry_result.attempt_count,
                "failure_reasons": [r.value for r in retry_result.failure_reasons],
                **interpreter.last_call_meta,
            })

    # Sort by confidence, take top 5
    interpretations.sort(key=lambda x: x[1].confidence, reverse=True)
    top_interpretations = interpretations[:settings.top_n_for_debate]

    logger.info(
        "Stage 1 complete: %d/%d interpreted, top %d advancing",
        len(interpretations), len(candidates), len(top_interpretations),
    )

    # Cost circuit breaker after interpretation
    if budget_usd and memory_service.working.total_cost_usd >= budget_usd:
        logger.warning(
            "Budget exhausted ($%.2f >= $%.2f) after interpretation",
            memory_service.working.total_cost_usd, budget_usd,
        )
        return PipelineRun(
            run_date=date.today(),
            regime=regime_context.get("regime", "unknown"),
            regime_details=regime_context,
            candidates_scored=len(candidates),
            interpreted=len(interpretations),
            debated=0,
            approved=[],
            vetoed=[],
            agent_logs=agent_logs,
            convergence_state="budget_exhausted",
        )

    # --- Stage 1b: Skill Execution ---
    skill_addons_by_ticker: dict[str, list[str]] = {}
    for candidate, interpretation in top_interpretations:
        candidate_context = _build_skill_context(
            candidate, interpretation, regime_context,
        )
        applicable_skills = skill_engine.find_applicable_skills(candidate_context)

        addons: list[str] = []
        for skill in applicable_skills:
            variables = {
                "ticker": candidate.ticker,
                "regime": regime_context.get("regime", "unknown"),
            }
            skill_result = await skill_engine.execute_skill(skill, variables)
            if skill_result.executed:
                addons.extend(skill_result.prompt_addons)
                agent_logs.append({
                    "agent": "skill_engine",
                    "ticker": candidate.ticker,
                    "skill": skill.name,
                    "executed": True,
                    "tool_results": len(skill_result.tool_results),
                    "prompt_addons": len(skill_result.prompt_addons),
                })
                logger.info(
                    "Skill '%s' executed for %s", skill.name, candidate.ticker,
                )

        if addons:
            skill_addons_by_ticker[candidate.ticker] = addons

    # --- Stage 2: Adversarial Debate (top 5 → survivors) ---
    logger.info("Stage 2: Debating %d candidates", len(top_interpretations))
    debate_survivors: list[
        tuple[RankedCandidate, SignalInterpretation, DebateResult]
    ] = []

    for candidate, interpretation in top_interpretations:
        signal_data = {
            "ticker": candidate.ticker,
            "signal_model": candidate.signal_model,
            "raw_score": candidate.raw_score,
            "confidence": interpretation.confidence,
            "component_scores": candidate.components,
            "features": candidate.features,
        }

        # Build memory context with skill addons
        memory_ctx = await memory_service.get_context_for_candidate(
            candidate.ticker, candidate.signal_model,
        )
        if candidate.ticker in skill_addons_by_ticker:
            memory_ctx["skill_addons"] = skill_addons_by_ticker[candidate.ticker]

        retry_result = await adversarial.debate(
            ticker=candidate.ticker,
            bull_thesis=interpretation.thesis,
            signal_data=signal_data,
            regime_context=regime_context,
            memory_context=memory_ctx,
        )

        debate = retry_result.value
        if debate:
            memory_service.working.add_cost(
                retry_result.total_tokens, retry_result.total_cost_usd,
            )
            agent_logs.append({
                "agent": "adversarial_validator",
                "ticker": candidate.ticker,
                "verdict": debate.final_verdict,
                "net_conviction": debate.net_conviction,
                "attempts": retry_result.attempt_count,
                "failure_reasons": [r.value for r in retry_result.failure_reasons],
                **adversarial.last_call_meta,
            })

            if debate.final_verdict != "REJECT":
                debate_survivors.append((candidate, interpretation, debate))
            else:
                logger.info("Debate REJECTED %s", candidate.ticker)
                vetoed.append(candidate.ticker)
                memory_service.working.record_veto(candidate.ticker)

    logger.info(
        "Stage 2 complete: %d/%d survived debate",
        len(debate_survivors), len(top_interpretations),
    )

    # Cost circuit breaker after debate
    if budget_usd and memory_service.working.total_cost_usd >= budget_usd:
        logger.warning(
            "Budget exhausted ($%.2f >= $%.2f) after debate",
            memory_service.working.total_cost_usd, budget_usd,
        )
        return PipelineRun(
            run_date=date.today(),
            regime=regime_context.get("regime", "unknown"),
            regime_details=regime_context,
            candidates_scored=len(candidates),
            interpreted=len(interpretations),
            debated=len(debate_survivors),
            approved=[],
            vetoed=vetoed,
            agent_logs=agent_logs,
            convergence_state="budget_exhausted",
        )

    # --- Stage 3: Risk Gate (survivors → final 1-2) ---
    logger.info("Stage 3: Risk gate for %d survivors", len(debate_survivors))

    for candidate, interpretation, debate in debate_survivors:
        # Build memory context with skill addons
        memory_ctx = await memory_service.get_context_for_candidate(
            candidate.ticker, candidate.signal_model,
        )
        if candidate.ticker in skill_addons_by_ticker:
            memory_ctx["skill_addons"] = skill_addons_by_ticker[candidate.ticker]

        retry_result = await risk_gate.evaluate(
            ticker=candidate.ticker,
            interpretation=interpretation.model_dump(),
            debate_result=debate.model_dump(),
            signal_data={
                "signal_model": candidate.signal_model,
                "raw_score": candidate.raw_score,
                "entry_price": candidate.entry_price,
                "stop_loss": candidate.stop_loss,
                "target_1": candidate.target_1,
                "target_2": candidate.target_2,
                "holding_period": candidate.holding_period,
            },
            regime_context=regime_context,
            existing_positions=existing_positions,
            memory_context=memory_ctx,
        )

        gate_result = retry_result.value
        if gate_result:
            memory_service.working.add_cost(
                retry_result.total_tokens, retry_result.total_cost_usd,
            )
            agent_logs.append({
                "agent": "risk_gatekeeper",
                "ticker": candidate.ticker,
                "decision": gate_result.decision.value,
                "position_size_pct": gate_result.position_size_pct,
                "attempts": retry_result.attempt_count,
                "failure_reasons": [r.value for r in retry_result.failure_reasons],
                **risk_gate.last_call_meta,
            })

            if gate_result.decision in (GateDecision.APPROVE, GateDecision.ADJUST):
                # Apply adjustments if ADJUST
                stop = gate_result.adjusted_stop or candidate.stop_loss
                target = gate_result.adjusted_target or candidate.target_1

                approved.append(PipelineResult(
                    ticker=candidate.ticker,
                    signal_model=candidate.signal_model,
                    direction=candidate.direction,
                    entry_price=candidate.entry_price,
                    stop_loss=stop,
                    target_1=target,
                    target_2=candidate.target_2,
                    holding_period=candidate.holding_period,
                    confidence=interpretation.confidence,
                    interpretation=interpretation,
                    debate=debate,
                    risk_gate=gate_result,
                    features=candidate.features,
                ))
                memory_service.working.record_approval(candidate.ticker)
            else:
                vetoed.append(candidate.ticker)
                memory_service.working.record_veto(candidate.ticker)

        # Enforce max picks
        if len(approved) >= settings.max_final_picks:
            break

    return PipelineRun(
        run_date=date.today(),
        regime=regime_context.get("regime", "unknown"),
        regime_details=regime_context,
        candidates_scored=len(candidates),
        interpreted=len(interpretations),
        debated=len(debate_survivors),
        approved=approved,
        vetoed=vetoed,
        agent_logs=agent_logs,
    )


def _build_skill_context(
    candidate: RankedCandidate,
    interpretation: SignalInterpretation,
    regime_context: dict,
) -> dict:
    """Build a context dict for skill precondition matching."""
    return {
        "ticker": candidate.ticker,
        "regime": regime_context.get("regime", "unknown"),
        "has_earnings_soon": RiskFlag.EARNINGS_IMMINENT in interpretation.risk_flags,
        "check_sector_exposure": True,  # always check sector exposure
        "confidence": interpretation.confidence,
        "signal_model": candidate.signal_model,
    }
