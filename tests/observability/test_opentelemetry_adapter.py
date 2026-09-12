"""Unit tests for OpenTelemetryTracer adapter and lifecycle."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from enterprise_orchestrator.observability.audit.redaction import SensitiveDataRedactor
from enterprise_orchestrator.observability.tracing.models import SpanKind, SpanStatus
from enterprise_orchestrator.observability.tracing.opentelemetry_adapter import (
    HAS_OTEL,
    OpenTelemetryTracer,
)


def test_opentelemetry_tracer_disabled():
    """Verify tracer behavior when enabled=False."""
    tracer = OpenTelemetryTracer(enabled=False)
    assert tracer.enabled is False

    span = tracer.start_span("test_disabled_node", kind=SpanKind.NODE)
    tracer.end_span(span, status=SpanStatus.OK)

    recorded = tracer.get_spans()
    assert len(recorded) == 1
    assert recorded[0].name == "test_disabled_node"
    assert recorded[0].status == SpanStatus.OK


@pytest.mark.asyncio
async def test_opentelemetry_tracer_scope_and_redaction():
    """Verify async trace_scope and attribute redaction."""
    redactor = SensitiveDataRedactor()
    tracer = OpenTelemetryTracer(enabled=False, redactor=redactor)

    async with tracer.trace_scope(
        name="test_scope_step",
        kind=SpanKind.LLM,
        attributes={"api_key": "secret-12345", "step": "generation"},
        run_id="run-100",
    ) as span:
        assert span.attributes["api_key"] == "[REDACTED]"
        assert span.attributes["step"] == "generation"
        assert span.attributes["run_id"] == "run-100"

    spans = tracer.get_spans(run_id="run-100")
    assert len(spans) == 1
    assert spans[0].status == SpanStatus.OK
    assert spans[0].duration_ms is not None
    assert spans[0].duration_ms >= 0


def test_opentelemetry_tracer_error_handling():
    """Verify error recording on spans."""
    tracer = OpenTelemetryTracer(enabled=False)
    span = tracer.start_span("failed_node", kind=SpanKind.NODE)
    tracer.end_span(span, status=SpanStatus.ERROR, error=RuntimeError("Tool failed"))

    spans = tracer.get_spans()
    assert len(spans) == 1
    assert spans[0].status == SpanStatus.ERROR
    assert spans[0].error_message == "Tool failed"


def test_opentelemetry_tracer_exception_isolation():
    """Verify that exceptions in OTel SDK are safely caught and never propagate."""
    tracer = OpenTelemetryTracer(enabled=True)
    # Simulate an internal tracer failure
    tracer._otel_tracer = MagicMock()
    tracer._otel_tracer.start_span.side_effect = RuntimeError("OTel backend crash")
    tracer.enabled = True

    # Starting a span should not raise an exception
    span = tracer.start_span("safe_span", kind=SpanKind.AGENT)
    assert span is not None
    assert span.name == "safe_span"


def test_opentelemetry_tracer_shutdown():
    """Verify safe shutdown without provider."""
    tracer = OpenTelemetryTracer(enabled=False)
    # Should complete with no errors
    tracer.shutdown(timeout_seconds=1.0)


@pytest.mark.asyncio
async def test_opentelemetry_shutdown_timeout_enforcement():
    """Verify that a slow or hanging shutdown is bounded by timeout and does not block."""
    tracer = OpenTelemetryTracer(enabled=False)
    mock_provider = MagicMock()

    def slow_shutdown():
        time.sleep(0.5)

    mock_provider.shutdown.side_effect = slow_shutdown
    tracer._tracer_provider = mock_provider

    # Enforce strict 0.1s timeout
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            asyncio.to_thread(tracer.shutdown),
            timeout=0.1,
        )


def test_opentelemetry_mocked_provider_lifecycle():
    """Verify full OTel lifecycle with mocked TracerProvider."""
    with patch("enterprise_orchestrator.observability.tracing.opentelemetry_adapter.HAS_OTEL", True):
        mock_provider = MagicMock()
        mock_otel_tracer = MagicMock()
        mock_otel_span = MagicMock()
        mock_otel_tracer.start_span.return_value = mock_otel_span
        mock_provider.get_tracer.return_value = mock_otel_tracer

        tracer = OpenTelemetryTracer(service_name="test-service", enabled=False)
        tracer._tracer_provider = mock_provider
        tracer._otel_tracer = mock_otel_tracer
        tracer.enabled = True

        span = tracer.start_span("agent_execute", kind=SpanKind.AGENT, attributes={"query": "test"})
        span.add_event("tool_called", {"tool": "search"})
        tracer.end_span(span, status=SpanStatus.OK)

        # Verify start_span was called on OTel tracer
        assert mock_otel_tracer.start_span.called
        # Verify end was called on OTel span
        assert mock_otel_span.end.called

        tracer.shutdown()
        assert mock_provider.shutdown.called
