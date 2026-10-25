"""Routing vocabulary shared by every service."""

from enum import StrEnum


class Route(StrEnum):
    """The three permitted outcomes of a compliance query."""

    ANSWER = "answer-with-citation"
    MORE_DETAIL = "request-more-detail"
    LEGAL_COUNSEL = "legal-counsel-review"


class RunStatus(StrEnum):
    """Lifecycle of an asynchronous run."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    QUARANTINED = "quarantined"