"""Tests for A5: Retry with exponential backoff and Circuit Breaker."""

from __future__ import annotations

import time
import pytest

from agent.analyst import analyse
from agent.model import Message, ModelError
from agent.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    ResilientModel,
    RetryPolicy,
)
from contracts import Route


class FailingModel:
    """Mock model that raises errors or succeeds after n attempts."""

    def __init__(self, failure_count: int, failure_exc: Exception, success_val: str = "ok") -> None:
        self.failure_count = failure_count
        self.failure_exc = failure_exc
        self.success_val = success_val
        self.call_count = 0
        self.name = "mock-failing"
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], **kwargs):
        self.calls.append(messages)
        self.call_count += 1
        if self.call_count <= self.failure_count:
            raise self.failure_exc
        return self.success_val


def test_retry_on_transient_error_succeeds():
    """Transient timeout error retries and succeeds on the second attempt."""
    mock = FailingModel(failure_count=1, failure_exc=ModelError("model endpoint unreachable: timeout"))
    delays: list[float] = []

    resilient = ResilientModel(
        mock,
        breaker=CircuitBreaker(failure_threshold=5),
        retry_policy=RetryPolicy(max_retries=2, backoff_factor=0.1),
        sleep_fn=lambda d: delays.append(d),
    )

    out = resilient.complete([{"role": "user", "content": "hello"}])
    assert out == "ok"
    assert mock.call_count == 2
    assert len(delays) == 1
    assert delays[0] == pytest.approx(0.1)


def test_exponential_backoff_delays():
    """Retries use exponential backoff: delay = backoff_factor * (2 ** attempt)."""
    # 5 failures ensures all 3 retries (4 total attempts) fail
    mock = FailingModel(failure_count=5, failure_exc=ModelError("http 503 service unavailable"))
    delays: list[float] = []

    policy = RetryPolicy(max_retries=3, backoff_factor=0.25)
    resilient = ResilientModel(
        mock,
        breaker=CircuitBreaker(failure_threshold=10),
        retry_policy=policy,
        sleep_fn=lambda d: delays.append(d),
    )

    with pytest.raises(ModelError):
        resilient.complete([{"role": "user", "content": "hi"}])

    assert mock.call_count == 4  # 1 initial + 3 retries
    assert delays == [pytest.approx(0.25), pytest.approx(0.50), pytest.approx(1.00)]


def test_non_retryable_error_fails_immediately():
    """Non-retryable client errors (HTTP 401 Unauthorized) fail on attempt 1 without retry."""
    mock = FailingModel(failure_count=5, failure_exc=ModelError("model endpoint returned HTTP 401: Unauthorized"))
    delays: list[float] = []

    resilient = ResilientModel(
        mock,
        breaker=CircuitBreaker(failure_threshold=5),
        retry_policy=RetryPolicy(max_retries=3, backoff_factor=0.1),
        sleep_fn=lambda d: delays.append(d),
    )

    with pytest.raises(ModelError, match="HTTP 401"):
        resilient.complete([{"role": "user", "content": "hi"}])

    assert mock.call_count == 1
    assert delays == []


def test_consecutive_failures_trip_circuit_breaker_open():
    """Circuit breaker trips OPEN once consecutive failures reach failure_threshold."""
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_s=10.0)
    mock = FailingModel(failure_count=10, failure_exc=ModelError("temporary failure: timeout"))

    resilient = ResilientModel(
        mock,
        breaker=breaker,
        retry_policy=RetryPolicy(max_retries=0),  # 1 failure per complete() call
    )

    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True

    # Call 1: failure #1
    with pytest.raises(ModelError):
        resilient.complete([{"role": "user", "content": "1"}])
    assert breaker.state == CircuitState.CLOSED

    # Call 2: failure #2
    with pytest.raises(ModelError):
        resilient.complete([{"role": "user", "content": "2"}])
    assert breaker.state == CircuitState.CLOSED

    # Call 3: failure #3 -> trips OPEN
    with pytest.raises(CircuitBreakerOpenError):
        resilient.complete([{"role": "user", "content": "3"}])

    assert breaker.state == CircuitState.OPEN
    assert breaker.trip_count == 1
    assert breaker.allow_request() is False


def test_circuit_breaker_fast_fails_when_open():
    """When circuit breaker is OPEN, subsequent calls fast-fail immediately without calling model."""
    breaker = CircuitBreaker(failure_threshold=2)
    breaker.trip()  # Forcibly open

    mock = FailingModel(failure_count=0, failure_exc=RuntimeError("should not be reached"))
    resilient = ResilientModel(mock, breaker=breaker)

    with pytest.raises(CircuitBreakerOpenError, match="is OPEN"):
        resilient.complete([{"role": "user", "content": "probe"}])

    assert mock.call_count == 0


def test_circuit_breaker_manual_trip_on_request():
    """Circuit breaker can be tripped on request for demo/test (A5 demonstrable)."""
    breaker = CircuitBreaker(failure_threshold=5)
    assert breaker.state == CircuitState.CLOSED
    breaker.trip()
    assert breaker.state == CircuitState.OPEN
    assert breaker.trip_count == 1
    assert breaker.allow_request() is False

    breaker.reset()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_circuit_breaker_half_open_recovery():
    """After recovery_timeout_s, breaker enters HALF_OPEN and resets to CLOSED upon success."""
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0.01)
    mock = FailingModel(failure_count=1, failure_exc=ModelError("timeout"))

    resilient = ResilientModel(
        mock,
        breaker=breaker,
        retry_policy=RetryPolicy(max_retries=0),
    )

    # Fail once -> OPEN
    with pytest.raises(CircuitBreakerOpenError):
        resilient.complete([{"role": "user", "content": "1"}])
    assert breaker.state == CircuitState.OPEN

    # Wait for recovery timeout
    time.sleep(0.02)

    # State transitions to HALF_OPEN on query
    assert breaker.state == CircuitState.HALF_OPEN
    assert breaker.allow_request() is True

    # Next call succeeds -> CLOSED
    out = resilient.complete([{"role": "user", "content": "probe"}])
    assert out == "ok"
    assert breaker.state == CircuitState.CLOSED


def test_pipeline_degrades_to_legal_counsel_when_breaker_open():
    """Analyst loop catches CircuitBreakerOpenError and cleanly degrades to Route.LEGAL_COUNSEL."""
    breaker = CircuitBreaker()
    breaker.trip()  # Open breaker

    mock = FailingModel(failure_count=0, failure_exc=RuntimeError("unused"))
    resilient = ResilientModel(mock, breaker=breaker)

    request = {
        "question": "What are the gratuity requirements?",
        "establishment": {"headcount": 50},
    }
    result = analyse("What are the gratuity requirements?", request, model=resilient)

    assert result.route == Route.LEGAL_COUNSEL
    assert result.claims == []
    assert result.draft_answer == ""
