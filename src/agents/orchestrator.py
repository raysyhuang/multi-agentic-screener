"""Agent orchestrator — runs the 3-agent pipeline sequentially.

Flow:
  Top 10 candidates → Signal Interpreter → Top 5 by confidence →
  Adversarial Debate → Survivors → Risk Gate → Final 1-2 picks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import date

from src.agents.signal_interpreter import SignalInterpreterAgent
from src.agents.adversarial import AdversarialAgent
from src.agents.risk_gate import RiskGateAgent
from src.agents.base import (
    GateDecision,
    SignalInterpretation,
    DebateResult,
    RiskGateOutput,
)
from src.config import get_settings
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


async def run_agent_pipeline(
    candidates: list[RankedCandidate],
    regime_context: dict,
    existing_positions: list[dict] | None = None,
) -> PipelineRun:
    """Run the full 3-agent pipeline on ranked candidates.

    Args:
        candidates: Top N candidates from the ranker (typically 10)
        regime_context: Current regime assessment as dict
        existing_positions: Current portfolio positions for correlation check
    """
    settings = get_settings()
    interpreter = SignalInterpreterAgent()
    adversarial = AdversarialAgent()
    risk_gate = RiskGateAgent()

    agent_logs = []
    approved = []
    vetoed = []

    # --- Stage 1: Signal Interpretation (top 10 → top 5) ---
    logger.info("Stage 1: Interpreting %d candidates", len(candidates))
    interpretations: list[tuple[RankedCandidate, SignalInterpretation]] = []

    for candidate in candidates[:settings.top_n_for_interpretation]:
        result = await interpreter.interpret(candidate, regime_context)
        if result:
            interpretations.append((candidate, result))
            agent_logs.append({
                "agent": "signal_interpreter",
                "ticker": candidate.ticker,
                "confidence": result.confidence,
            })

    # Sort by confidence, take top 5
    interpretations.sort(key=lambda x: x[1].confidence, reverse=True)
    top_interpretations = interpretations[:settings.top_n_for_debate]

    logger.info(
        "Stage 1 complete: %d/%d interpreted, top %d advancing",
        len(interpretations), len(candidates), len(top_interpretations),
    )

    # --- Stage 2: Adversarial Debate (top 5 → survivors) ---
    logger.info("Stage 2: Debating %d candidates", len(top_interpretations))
    debate_survivors: list[tuple[RankedCandidate, SignalInterpretation, DebateResult]] = []

    for candidate, interpretation in top_interpretations:
        signal_data = {
            "ticker": candidate.ticker,
            "signal_model": candidate.signal_model,
            "raw_score": candidate.raw_score,
            "confidence": interpretation.confidence,
            "component_scores": candidate.components,
            "features": candidate.features,
        }

        debate = await adversarial.debate(
            ticker=candidate.ticker,
            bull_thesis=interpretation.thesis,
            signal_data=signal_data,
            regime_context=regime_context,
        )

        if debate:
            agent_logs.append({
                "agent": "adversarial_validator",
                "ticker": candidate.ticker,
                "verdict": debate.final_verdict,
                "net_conviction": debate.net_conviction,
            })

            if debate.final_verdict != "REJECT":
                debate_survivors.append((candidate, interpretation, debate))
            else:
                logger.info("Debate REJECTED %s", candidate.ticker)
                vetoed.append(candidate.ticker)

    logger.info(
        "Stage 2 complete: %d/%d survived debate",
        len(debate_survivors), len(top_interpretations),
    )

    # --- Stage 3: Risk Gate (survivors → final 1-2) ---
    logger.info("Stage 3: Risk gate for %d survivors", len(debate_survivors))

    for candidate, interpretation, debate in debate_survivors:
        gate_result = await risk_gate.evaluate(
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
        )

        if gate_result:
            agent_logs.append({
                "agent": "risk_gatekeeper",
                "ticker": candidate.ticker,
                "decision": gate_result.decision.value,
                "position_size_pct": gate_result.position_size_pct,
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
            else:
                vetoed.append(candidate.ticker)

        # Enforce max picks
        if len(approved) >= settings.max_final_picks:
            break

    logger.info(
        "Pipeline complete: %d approved, %d vetoed",
        len(approved), len(vetoed),
    )

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
