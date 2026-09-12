"""Correlation context propagation management using contextvars."""

from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel, Field


class TraceContext(BaseModel):
    """Execution context carrying correlation identifiers and active parent span."""

    trace_id: str
    run_id: str
    parent_span_id: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None


_CURRENT_TRACE_CONTEXT: ContextVar[Optional[TraceContext]] = ContextVar(
    "_CURRENT_TRACE_CONTEXT", default=None
)


def get_current_trace_context() -> Optional[TraceContext]:
    """Retrieve the active trace context from async contextvars."""
    return _CURRENT_TRACE_CONTEXT.get()


def set_current_trace_context(context: Optional[TraceContext]) -> None:
    """Set the active trace context."""
    _CURRENT_TRACE_CONTEXT.set(context)
