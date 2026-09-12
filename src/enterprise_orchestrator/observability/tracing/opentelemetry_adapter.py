"""OpenTelemetry tracing adapter translating internal Spans into OpenTelemetry SDK spans."""

import logging
import threading
from typing import Any, Dict, List, Optional

from enterprise_orchestrator.observability.audit.redaction import SensitiveDataRedactor
from enterprise_orchestrator.observability.tracing.base import BaseTracer
from enterprise_orchestrator.observability.tracing.models import Span, SpanKind, SpanStatus

logger = logging.getLogger(__name__)

# Optional dynamic import for OpenTelemetry
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
    from opentelemetry.trace import Status, StatusCode
    HAS_OTEL = True
except ImportError:  # pragma: no cover
    HAS_OTEL = False
    otel_trace = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None
    SpanExporter = None
    Status = None
    StatusCode = None


class OpenTelemetryTracer(BaseTracer):
    """OpenTelemetry adapter extending BaseTracer with optional OTLP/SDK bridge and fallback."""

    def __init__(
        self,
        service_name: str = "enterprise-orchestrator",
        otlp_endpoint: Optional[str] = None,
        otlp_headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 5.0,
        span_exporter: Optional[Any] = None,
        redactor: Optional[SensitiveDataRedactor] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(redactor=redactor)
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.otlp_headers = otlp_headers
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled and HAS_OTEL

        self._in_memory_spans: List[Span] = []
        self._active_otel_spans: Dict[str, Any] = {}
        self._lock = threading.Lock()

        self._tracer_provider: Optional[Any] = None
        self._otel_tracer: Optional[Any] = None

        if self.enabled:
            self._init_otel_provider(span_exporter=span_exporter)

    def _init_otel_provider(self, span_exporter: Optional[Any] = None) -> None:
        """Initialize OpenTelemetry SDK TracerProvider and SpanProcessor safely."""
        try:
            resource = Resource.create({"service.name": self.service_name})
            self._tracer_provider = TracerProvider(resource=resource)

            if span_exporter is not None:
                # Custom exporter injected (e.g. InMemorySpanExporter for testing)
                self._tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
            elif self.otlp_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    exporter = OTLPSpanExporter(
                        endpoint=self.otlp_endpoint,
                        headers=self.otlp_headers or {},
                        timeout=self.timeout_seconds,
                    )
                    self._tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                except Exception as exc:
                    logger.warning("Failed to initialize OTLPSpanExporter: %s", exc)

            self._otel_tracer = self._tracer_provider.get_tracer(
                "enterprise_orchestrator",
                "0.1.0",
            )
        except Exception as exc:
            logger.warning("Failed to initialize OpenTelemetry provider: %s", exc)
            self.enabled = False

    def _map_span_kind(self, kind: SpanKind) -> Any:
        """Map internal SpanKind enum to OpenTelemetry SpanKind."""
        if not HAS_OTEL or otel_trace is None:
            return None
        if kind in (SpanKind.LLM, SpanKind.TOOL, SpanKind.RETRIEVAL):
            return otel_trace.SpanKind.CLIENT
        if kind == SpanKind.WORKFLOW:
            return otel_trace.SpanKind.SERVER
        return otel_trace.SpanKind.INTERNAL

    def _record_start(self, span: Span) -> None:
        """Record span start in memory and forward to OpenTelemetry if active."""
        if not self.enabled or self._otel_tracer is None:
            return

        try:
            otel_kind = self._map_span_kind(span.kind)
            # Ensure attributes are sanitized
            sanitized_attrs = self.redactor.redact(span.attributes)

            otel_span = self._otel_tracer.start_span(
                name=span.name,
                kind=otel_kind,
                start_time=int(span.start_time * 1e9),
                attributes=sanitized_attrs,
            )
            with self._lock:
                self._active_otel_spans[span.span_id] = otel_span
        except Exception as exc:
            # Telemetry is strictly best-effort and must never fail orchestration
            logger.debug("Failed to record OTel span start: %s", exc)

    def _record_end(self, span: Span) -> None:
        """Record finished span in memory and close OpenTelemetry span if active."""
        with self._lock:
            self._in_memory_spans.append(span)
            otel_span = self._active_otel_spans.pop(span.span_id, None)

        if not self.enabled or otel_span is None:
            return

        try:
            # Set terminal status
            if Status is not None and StatusCode is not None:
                if span.status == SpanStatus.ERROR:
                    otel_span.set_status(
                        Status(StatusCode.ERROR, description=span.error_message or "Error")
                    )
                else:
                    otel_span.set_status(Status(StatusCode.OK))
            elif hasattr(otel_span, "set_status"):
                otel_span.set_status(span.status.value)

            # Record any events on span
            for event in span.events:
                event_attrs = self.redactor.redact(event.get("attributes", {}))
                otel_span.add_event(
                    name=event.get("name", "event"),
                    attributes=event_attrs,
                    timestamp=int(event.get("timestamp", span.end_time or span.start_time) * 1e9),
                )

            end_time_ns = int((span.end_time or span.start_time) * 1e9)
            otel_span.end(end_time=end_time_ns)
        except Exception as exc:
            # Failure isolation
            logger.debug("Failed to record OTel span end: %s", exc)

    def get_spans(self, trace_id: Optional[str] = None, run_id: Optional[str] = None) -> List[Span]:
        """Retrieve finished spans from in-memory collection."""
        with self._lock:
            spans = list(self._in_memory_spans)

        if trace_id:
            spans = [s for s in spans if s.trace_id == trace_id]
        if run_id:
            spans = [s for s in spans if s.attributes.get("run_id") == run_id]
        return spans

    def clear(self) -> None:
        """Reset internal in-memory spans and active tracking."""
        with self._lock:
            self._in_memory_spans.clear()
            self._active_otel_spans.clear()

    def shutdown(self, timeout_seconds: Optional[float] = None) -> None:
        """Synchronously flush and shut down TracerProvider if initialized."""
        if self._tracer_provider is not None:
            try:
                self._tracer_provider.shutdown()
            except Exception as exc:
                logger.warning("Error during OpenTelemetry TracerProvider shutdown: %s", exc)
