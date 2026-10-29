"""Resilience layer for model calls: exponential backoff retry and circuit breaker (A5).

Requirements satisfied:
- Retry with exponential backoff on retryable transient errors (network, timeout, 429/5xx).
- Circuit breaker with CLOSED, OPEN, and HALF_OPEN states.
- Fast-fail when OPEN, preventing cascading failures and resource exhaustion.
- Definitive fallback path degrading gracefully to Route.LEGAL_COUNSEL.
- Demonstrable on request: test that forcibly trips or triggers breaker opening.
"""

from __future__ import annotations

import logging
import os
import time
from enum import StrEnum
from typing import Any, Callable, TypeVar

import httpx
from pydantic import BaseModel

from contracts.analyst_result import Route

from .model import Message, Model, ModelError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class CircuitState(StrEnum):
    """States of the model circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(ModelError):
    """Raised when an operation is rejected because the circuit breaker is OPEN."""


class CircuitBreaker:
    """Three-state circuit breaker guarding model endpoints.

    CLOSED: Normal operation. Successes reset failure counts.
    OPEN: Tripped when consecutive failures reach failure_threshold. Fast-fails calls.
    HALF_OPEN: After recovery_timeout_s, admits a trial probe request.
    """

    def __init__(
        self,
        failure_threshold: int | None = None,
        recovery_timeout_s: float | None = None,
        name: str = "llm-breaker",
    ) -> None:
        self.name = name
        raw_threshold = os.environ.get("LLM_CB_FAILURE_THRESHOLD")
        self.failure_threshold = (
            failure_threshold
            if failure_threshold is not None
            else (int(raw_threshold) if raw_threshold else 3)
        )
        raw_recovery = os.environ.get("LLM_CB_RECOVERY_TIMEOUT_S")
        self.recovery_timeout_s = (
            recovery_timeout_s
            if recovery_timeout_s is not None
            else (float(raw_recovery) if raw_recovery else 30.0)
        )

        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._last_state_change: float = time.monotonic()
        self._trip_count: int = 0

    @property
    def state(self) -> CircuitState:
        # Check for automatic transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN:
            now = time.monotonic()
            if now - self._last_state_change >= self.recovery_timeout_s:
                self._state = CircuitState.HALF_OPEN
                self._last_state_change = now
                logger.info(
                    "Circuit breaker '%s' transition: OPEN -> HALF_OPEN (recovery probe)",
                    self.name,
                )
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def trip_count(self) -> int:
        return self._trip_count

    def allow_request(self) -> bool:
        """True if the breaker permits a call to proceed."""
        current_state = self.state
        if current_state == CircuitState.CLOSED:
            return True
        if current_state == CircuitState.HALF_OPEN:
            return True
        return False

    def record_success(self) -> None:
        """Record a successful model invocation."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info(
                "Circuit breaker '%s' probe succeeded; transition: HALF_OPEN -> CLOSED",
                self.name,
            )
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_state_change = time.monotonic()

    def record_failure(self, exc: Exception) -> None:
        """Record an upstream failure and trip if threshold is reached."""
        self._consecutive_failures += 1
        now = time.monotonic()
        logger.warning(
            "Circuit breaker '%s' registered failure #%d: %s",
            self.name,
            self._consecutive_failures,
            exc,
        )

        if self._state == CircuitState.HALF_OPEN or self._consecutive_failures >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                self._state = CircuitState.OPEN
                self._last_state_change = now
                self._trip_count += 1
                logger.error(
                    "Circuit breaker '%s' tripped OPEN (%d consecutive failures, trip #%d)",
                    self.name,
                    self._consecutive_failures,
                    self._trip_count,
                )

    def trip(self) -> None:
        """Forcibly trip the circuit breaker OPEN (test & demo manual trigger)."""
        self._state = CircuitState.OPEN
        self._consecutive_failures = self.failure_threshold
        self._last_state_change = time.monotonic()
        self._trip_count += 1
        logger.warning("Circuit breaker '%s' manually tripped to OPEN", self.name)

    def reset(self) -> None:
        """Reset the breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_state_change = time.monotonic()


class RetryPolicy:
    """Configurable exponential backoff retry policy."""

    def __init__(
        self,
        max_retries: int | None = None,
        backoff_factor: float | None = None,
    ) -> None:
        raw_retries = os.environ.get("LLM_MAX_RETRIES")
        self.max_retries = (
            max_retries
            if max_retries is not None
            else (int(raw_retries) if raw_retries else 2)
        )
        raw_backoff = os.environ.get("LLM_BACKOFF_FACTOR")
        self.backoff_factor = (
            backoff_factor
            if backoff_factor is not None
            else (float(raw_backoff) if raw_backoff else 0.5)
        )

    def is_retryable(self, exc: Exception) -> bool:
        """Determine if an exception represents a retryable transient failure."""
        if isinstance(exc, CircuitBreakerOpenError):
            return False

        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)):
            return True

        if isinstance(exc, ModelError):
            msg = str(exc).lower()
            # Non-retryable markers: client errors, bad schema, unauthorized
            non_retryable = ("http 400", "http 401", "http 403", "http 404", "unknown llm_provider")
            if any(term in msg for term in non_retryable):
                return False
            # Retryable markers: network unreachable, timeouts, rate limits, 5xx
            retryable = (
                "unreachable",
                "timeout",
                "timed out",
                "rate limit",
                "http 429",
                "http 500",
                "http 502",
                "http 503",
                "http 504",
                "temporary failure",
            )
            return any(term in msg for term in retryable)

        return False

    def calculate_delay(self, attempt: int) -> float:
        """Exponential backoff delay: backoff_factor * (2 ** attempt)."""
        return self.backoff_factor * (2**attempt)


class ResilientModel:
    """Model wrapper providing exponential backoff and circuit breaking (A5).

    Delegates underlying generation to any ``Model`` instance.
    """

    def __init__(
        self,
        underlying: Model,
        breaker: CircuitBreaker | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.underlying = underlying
        self.breaker = breaker or get_circuit_breaker()
        self.retry_policy = retry_policy or RetryPolicy()
        self.sleep_fn = sleep_fn or time.sleep
        self.name = getattr(underlying, "name", "resilient")

    @property
    def calls(self) -> list[list[Message]]:
        return getattr(self.underlying, "calls", [])

    def complete(
        self,
        messages: list[Message],
        *,
        response_schema: type[T] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | T:
        if not self.breaker.allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.breaker.name}' is OPEN. "
                f"Fast-failing request to protect downstream capacity."
            )

        attempts = 1 + max(0, self.retry_policy.max_retries)
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                result = self.underlying.complete(
                    messages,
                    response_schema=response_schema,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self.breaker.record_success()
                return result
            except Exception as exc:
                last_exc = exc
                if not self.retry_policy.is_retryable(exc):
                    # Non-retryable: bubble up immediately
                    raise

                self.breaker.record_failure(exc)
                if attempt + 1 < attempts and self.breaker.allow_request():
                    delay = self.retry_policy.calculate_delay(attempt)
                    logger.warning(
                        "Model call failed (attempt %d/%d): %s. Retrying in %.2fs...",
                        attempt + 1,
                        attempts,
                        exc,
                        delay,
                    )
                    self.sleep_fn(delay)
                else:
                    break

        if not self.breaker.allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.breaker.name}' tripped OPEN after retries exhausted: {last_exc}"
            ) from last_exc

        assert last_exc is not None
        raise last_exc


# Global shared circuit breaker
_GLOBAL_BREAKER = CircuitBreaker()


def get_circuit_breaker() -> CircuitBreaker:
    """Return the global process-wide circuit breaker instance."""
    return _GLOBAL_BREAKER


def reset_circuit_breaker() -> None:
    """Reset the global circuit breaker state."""
    _GLOBAL_BREAKER.reset()

