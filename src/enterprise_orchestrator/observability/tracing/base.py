"""Abstract tracer interface and scoped span context manager."""

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional
from uuid import uuid4

from enterprise_orchestrator.observability.audit.redaction import SensitiveDataRedactor
from enterprise_orchestrator.observability.tracing.context import (
    TraceContext,
    get_current_trace_context,
    set_current_trace_context,
)
from enterprise_orchestrator.observability.tracing.models import Span, SpanKind, SpanStatus


class BaseTracer(ABC):
    """Abstract interface for distributed tracing."""

    def __init__(self, redactor: Optional[SensitiveDataRedactor] = None) -> None:
        self.redactor = redactor or SensitiveDataRedactor()

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.NODE,
        attributes: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        run_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Span:
        """Create and start a new Span inheriting context if available."""
        current_ctx = get_current_trace_context()

        effective_trace_id = trace_id or (current_ctx.trace_id if current_ctx else str(uuid4())[:16])
        effective_run_id = run_id or (current_ctx.run_id if current_ctx else effective_trace_id)
        effective_parent_id = parent_span_id or (current_ctx.parent_span_id if current_ctx else None)
        effective_correlation_id = correlation_id or (current_ctx.correlation_id if current_ctx else None)
        effective_session_id = session_id or (current_ctx.session_id if current_ctx else None)

        sanitized_attributes = self.redactor.redact(attributes or {})
        if effective_run_id:
            sanitized_attributes["run_id"] = effective_run_id
        if effective_correlation_id:
            sanitized_attributes["correlation_id"] = effective_correlation_id
        if effective_session_id:
            sanitized_attributes["session_id"] = effective_session_id

        span = Span(
            trace_id=effective_trace_id,
            parent_span_id=effective_parent_id,
            name=name,
            kind=kind,
            attributes=sanitized_attributes,
        )
        self._record_start(span)
        return span

    def end_span(
        self,
        span: Span,
        status: SpanStatus = SpanStatus.OK,
        error: Optional[Exception] = None,
    ) -> None:
        """Finalize and record finished span."""
        err_msg = str(error) if error else None
        span.finish(status=status, error_message=err_msg)
        self._record_end(span)

    @asynccontextmanager
    async def trace_scope(
        self,
        name: str,
        kind: SpanKind = SpanKind.NODE,
        attributes: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[Span, None]:
        """Asynchronous context manager creating a scoped child span with automatic context propagation."""
        parent_ctx = get_current_trace_context()
        span = self.start_span(
            name=name,
            kind=kind,
            attributes=attributes,
            run_id=run_id,
            correlation_id=correlation_id,
            session_id=session_id,
        )

        new_ctx = TraceContext(
            trace_id=span.trace_id,
            run_id=span.attributes.get("run_id", span.trace_id),
            parent_span_id=span.span_id,
            correlation_id=span.attributes.get("correlation_id"),
            session_id=span.attributes.get("session_id"),
        )
        token = set_current_trace_context(new_ctx)

        try:
            yield span
            self.end_span(span, status=SpanStatus.OK)
        except Exception as exc:
            self.end_span(span, status=SpanStatus.ERROR, error=exc)
            raise
        finally:
            set_current_trace_context(parent_ctx)

    @abstractmethod
    def _record_start(self, span: Span) -> None:
        """Hook called when a span starts."""
        pass

    @abstractmethod
    def _record_end(self, span: Span) -> None:
        """Hook called when a span finishes."""
        pass

    @abstractmethod
    def get_spans(self, trace_id: Optional[str] = None, run_id: Optional[str] = None) -> List[Span]:
        """Retrieve finished spans optionally filtered."""
        pass
