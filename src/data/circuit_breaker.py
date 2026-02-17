"""API circuit breaker — prevents cascading failures from flaky providers.

Tracks consecutive failures per provider. After a configurable number of
consecutive failures, the circuit opens and skips that provider for a
cooldown period. Resets on the first success.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes


@dataclass
class _ProviderState:
    consecutive_failures: int = 0
    open_until: float = 0.0  # monotonic timestamp when circuit re-closes
    total_failures: int = 0
    total_successes: int = 0


class APICircuitBreaker:
    """Per-provider circuit breaker with configurable threshold and cooldown."""

    def __init__(
        self,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ):
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._providers: dict[str, _ProviderState] = {}

    def _get(self, provider: str) -> _ProviderState:
        if provider not in self._providers:
            self._providers[provider] = _ProviderState()
        return self._providers[provider]

    def is_open(self, provider: str) -> bool:
        """Return True if the circuit is open (provider should be skipped)."""
        state = self._get(provider)
        if state.consecutive_failures < self._threshold:
            return False
        if time.monotonic() >= state.open_until:
            # Cooldown expired — allow a probe request (half-open)
            return False
        return True

    def record_success(self, provider: str) -> None:
        """Record a successful call — resets the failure counter."""
        state = self._get(provider)
        if state.consecutive_failures > 0:
            logger.info(
                "Circuit breaker reset for %s after %d consecutive failures",
                provider, state.consecutive_failures,
            )
        state.consecutive_failures = 0
        state.open_until = 0.0
        state.total_successes += 1

    def record_failure(self, provider: str) -> None:
        """Record a failed call — may open the circuit."""
        state = self._get(provider)
        state.consecutive_failures += 1
        state.total_failures += 1

        if state.consecutive_failures >= self._threshold:
            state.open_until = time.monotonic() + self._cooldown
            logger.warning(
                "Circuit OPEN for %s: %d consecutive failures — "
                "skipping for %.0fs",
                provider, state.consecutive_failures, self._cooldown,
            )

    def get_stats(self) -> dict[str, dict]:
        """Return stats for all tracked providers."""
        return {
            provider: {
                "consecutive_failures": s.consecutive_failures,
                "is_open": self.is_open(provider),
                "total_failures": s.total_failures,
                "total_successes": s.total_successes,
            }
            for provider, s in self._providers.items()
        }


class RateLimitError(Exception):
    """Raised when an API returns 429 Too Many Requests."""

    def __init__(self, provider: str, retry_after: float | None = None):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"Rate limited by {provider}" + (
            f" (retry after {retry_after}s)" if retry_after else ""
        ))
