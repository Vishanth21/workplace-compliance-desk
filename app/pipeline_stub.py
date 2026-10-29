"""Backward-compatibility alias for app.pipeline (S6).

The production pipeline builders and execution logic now live in ``app.pipeline``.
This module re-exports them for test and historical caller compatibility.
"""

from .pipeline import (
    analyst,
    verifier,
    _build_analyst,
    _build_verifier,
    set_analyst,
    set_verifier,
    run_pipeline,
)

__all__ = [
    "analyst",
    "verifier",
    "_build_analyst",
    "_build_verifier",
    "set_analyst",
    "set_verifier",
    "run_pipeline",
]
