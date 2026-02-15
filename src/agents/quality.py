"""Output quality checkers for agent results.

Each checker validates that an agent's structured output meets minimum
quality thresholds beyond basic schema validation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.base import SignalInterpretation, DebateResult, RiskGateOutput


@dataclass
class QualityCheck:
    """Result of a quality check on agent output."""
    passed: bool
    issues: list[str]

    @property
    def summary(self) -> str:
        if self.passed:
            return "Quality check passed"
        return "; ".join(self.issues)


def check_interpretation_quality(interp: SignalInterpretation) -> QualityCheck:
    """Validate that a SignalInterpretation meets minimum quality standards.

    Checks:
    - Thesis length >= 30 chars
    - At least 2 key_drivers
    - Valid trade structure: stop < entry < target for LONG
    """
    issues: list[str] = []

    if len(interp.thesis) < 30:
        issues.append(f"Thesis too short ({len(interp.thesis)} chars, need >= 30)")

    if len(interp.key_drivers) < 2:
        issues.append(f"Too few key_drivers ({len(interp.key_drivers)}, need >= 2)")

    # Check trade structure (assuming LONG direction)
    if interp.suggested_stop >= interp.suggested_entry:
        issues.append(
            f"Stop ({interp.suggested_stop}) >= entry ({interp.suggested_entry})"
        )

    if interp.suggested_target <= interp.suggested_entry:
        issues.append(
            f"Target ({interp.suggested_target}) <= entry ({interp.suggested_entry})"
        )

    return QualityCheck(passed=len(issues) == 0, issues=issues)


def check_debate_quality(debate: DebateResult) -> QualityCheck:
    """Validate that a DebateResult meets minimum quality standards.

    Checks:
    - Rebuttal summary is non-empty
    - Valid verdict (PROCEED, CAUTIOUS, REJECT)
    - Non-zero conviction
    """
    issues: list[str] = []

    if not debate.rebuttal_summary or len(debate.rebuttal_summary.strip()) < 10:
        issues.append("Rebuttal summary is empty or too short")

    valid_verdicts = {"PROCEED", "CAUTIOUS", "REJECT"}
    if debate.final_verdict not in valid_verdicts:
        issues.append(f"Invalid verdict '{debate.final_verdict}', expected one of {valid_verdicts}")

    if debate.net_conviction == 0:
        issues.append("Net conviction is zero")

    return QualityCheck(passed=len(issues) == 0, issues=issues)


def check_risk_gate_quality(gate: RiskGateOutput) -> QualityCheck:
    """Validate that a RiskGateOutput meets minimum quality standards.

    Checks:
    - Reasoning is non-empty and substantive
    """
    issues: list[str] = []

    if not gate.reasoning or len(gate.reasoning.strip()) < 10:
        issues.append("Reasoning is empty or too short")

    return QualityCheck(passed=len(issues) == 0, issues=issues)
