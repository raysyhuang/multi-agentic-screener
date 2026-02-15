"""Retry infrastructure for agent calls.

Provides RetryPolicy, RetryResult, AttemptRecord, and retry prompt
suffix generation. Agents use these to wrap LLM calls with automatic
retry on parse errors and API failures while respecting cost caps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")


class FailureReason(str, Enum):
    """Why an agent attempt failed."""
    PARSE_ERROR = "parse_error"
    LLM_API_ERROR = "llm_api_error"
    LOW_QUALITY = "low_quality"
    TIMEOUT = "timeout"
    FUNDAMENTAL_REJECT = "fundamental_reject"  # legitimate VETO/REJECT — never retry


@dataclass
class AttemptRecord:
    """Record of a single agent call attempt."""
    attempt_num: int
    success: bool
    failure_reason: FailureReason | None = None
    error_message: str | None = None
    raw_output: str | None = None


@dataclass
class RetryPolicy:
    """Controls retry behavior for agent calls."""
    max_attempts: int = 2
    retry_on_parse_error: bool = True
    retry_on_api_error: bool = True
    retry_on_low_quality: bool = False
    max_total_cost_usd: float = 0.50

    def should_retry(self, reason: FailureReason) -> bool:
        """Check if a failure reason is retryable under this policy."""
        if reason == FailureReason.FUNDAMENTAL_REJECT:
            return False
        if reason == FailureReason.PARSE_ERROR:
            return self.retry_on_parse_error
        if reason == FailureReason.LLM_API_ERROR:
            return self.retry_on_api_error
        if reason == FailureReason.LOW_QUALITY:
            return self.retry_on_low_quality
        if reason == FailureReason.TIMEOUT:
            return self.retry_on_api_error
        return False


@dataclass
class RetryResult(Generic[T]):
    """Wraps the final result of an agent call with retry history."""
    value: T | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.value is not None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def failure_reasons(self) -> list[FailureReason]:
        return [a.failure_reason for a in self.attempts if a.failure_reason]

    def add_attempt(self, record: AttemptRecord) -> None:
        self.attempts.append(record)

    def add_cost(self, tokens: int, cost_usd: float) -> None:
        self.total_tokens += tokens
        self.total_cost_usd += cost_usd


def build_retry_prompt_suffix(
    attempt_num: int,
    failure_reason: FailureReason,
    error_message: str | None = None,
) -> str:
    """Build a correction instruction to append on retry attempts.

    Returns a string that should be appended to the user prompt on retry.
    """
    if failure_reason == FailureReason.PARSE_ERROR:
        suffix = (
            f"\n\n[RETRY — Attempt {attempt_num}] "
            "Your previous response could not be parsed as valid JSON. "
            "Please respond with ONLY a valid JSON object matching the required schema. "
            "Do not include any text before or after the JSON."
        )
        if error_message:
            suffix += f"\nParse error was: {error_message}"
        return suffix

    if failure_reason == FailureReason.LLM_API_ERROR:
        return (
            f"\n\n[RETRY — Attempt {attempt_num}] "
            "The previous call failed due to an API error. "
            "Please try again with the same analysis."
        )

    if failure_reason == FailureReason.LOW_QUALITY:
        suffix = (
            f"\n\n[RETRY — Attempt {attempt_num}] "
            "Your previous response was valid but did not meet quality standards. "
            "Please provide a more thorough analysis."
        )
        if error_message:
            suffix += f"\nQuality issue: {error_message}"
        return suffix

    return f"\n\n[RETRY — Attempt {attempt_num}] Previous attempt failed. Please try again."
