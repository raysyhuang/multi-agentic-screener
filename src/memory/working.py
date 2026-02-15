"""Working memory — in-process run state.

Tracks the current pipeline run's state: which tickers have been
interpreted, vetoed, approved, and accumulates notes from agents.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class WorkingMemory:
    """In-process state for the current pipeline run."""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    regime: str = "unknown"
    interpreted_tickers: list[str] = field(default_factory=list)
    vetoed_tickers: list[str] = field(default_factory=list)
    approved_tickers: list[str] = field(default_factory=list)
    agent_notes: list[str] = field(default_factory=list)
    verifier_feedback: list[str] = field(default_factory=list)
    convergence_state: str = "running"
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    def record_interpretation(self, ticker: str) -> None:
        if ticker not in self.interpreted_tickers:
            self.interpreted_tickers.append(ticker)

    def record_veto(self, ticker: str) -> None:
        if ticker not in self.vetoed_tickers:
            self.vetoed_tickers.append(ticker)

    def record_approval(self, ticker: str) -> None:
        if ticker not in self.approved_tickers:
            self.approved_tickers.append(ticker)

    def add_note(self, note: str) -> None:
        self.agent_notes.append(note)

    def record_verifier_feedback(self, suggestions: list[str]) -> None:
        self.verifier_feedback.extend(suggestions)

    def set_convergence(self, state: str) -> None:
        self.convergence_state = state

    def add_cost(self, tokens: int, cost_usd: float) -> None:
        self.total_tokens += tokens
        self.total_cost_usd += cost_usd

    def to_prompt_context(self) -> str:
        """Format working memory state for inclusion in agent prompts."""
        lines = [
            f"Run ID: {self.run_id}",
            f"Regime: {self.regime}",
        ]

        if self.interpreted_tickers:
            lines.append(f"Interpreted so far: {', '.join(self.interpreted_tickers)}")
        if self.vetoed_tickers:
            lines.append(f"Vetoed so far: {', '.join(self.vetoed_tickers)}")
        if self.approved_tickers:
            lines.append(f"Approved so far: {', '.join(self.approved_tickers)}")
        if self.agent_notes:
            lines.append(f"Notes: {'; '.join(self.agent_notes[-5:])}")  # last 5

        if self.verifier_feedback:
            lines.append(f"Previous verifier feedback: {'; '.join(self.verifier_feedback[-3:])}")

        lines.append(f"Run cost so far: ${self.total_cost_usd:.4f}")

        return "\n".join(lines)
