"""OpenTelemetry distributed tracing and telemetry instrumentation (Requirement D5).

Provides:
- TracerProvider initialization with OTLP gRPC export (Jaeger / Grafana Tempo)
- Trace span utilities
- Safe fallback when OTLP endpoint is unavailable
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)

_INITIALIZED = False
_IN_MEMORY_EXPORTER = InMemorySpanExporter()


def init_telemetry(service_name: str = "workplace-compliance-desk") -> trace.Tracer:
    """Initialize OpenTelemetry tracer provider with OTLP exporter or in-memory fallback."""
    global _INITIALIZED
    if _INITIALIZED:
        return trace.get_tracer(service_name)

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # In-memory exporter for testing and instant inspection
    provider.add_span_processor(SimpleSpanProcessor(_IN_MEMORY_EXPORTER))

    # OTLP gRPC exporter if endpoint is configured or default jaeger
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info("OpenTelemetry OTLP exporter enabled targeting %s", otlp_endpoint)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialize OTLP gRPC exporter: %s", exc)

    trace.set_tracer_provider(provider)
    _INITIALIZED = True
    return trace.get_tracer(service_name)


def get_tracer(service_name: str = "workplace-compliance-desk") -> trace.Tracer:
    """Return the active OpenTelemetry tracer."""
    return init_telemetry(service_name)


def get_finished_spans():
    """Return all spans recorded in the in-memory exporter (for tests)."""
    return _IN_MEMORY_EXPORTER.get_finished_spans()


def clear_spans() -> None:
    """Reset the in-memory span store."""
    _IN_MEMORY_EXPORTER.clear()


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Generator[trace.Span, None, None]:
    """Context manager for tracing an operational unit with attributes."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    span.set_attribute(k, str(v) if isinstance(v, (dict, list)) else v)
        yield span

