"""Unit tests for distributed tracing, context propagation, and InMemoryTracer."""

import pytest

from enterprise_orchestrator.observability.tracing.context import (
    TraceContext,
    get_current_trace_context,
    set_current_trace_context,
)
from enterprise_orchestrator.observability.tracing.in_memory import InMemoryTracer
from enterprise_orchestrator.observability.tracing.models import SpanKind, SpanStatus


@pytest.mark.asyncio
async def test_tracing_hierarchy_and_context_propagation() -> None:
    tracer = InMemoryTracer()

    async with tracer.trace_scope("workflow_root", kind=SpanKind.WORKFLOW, run_id="run-trace-1", correlation_id="corr-1") as root_span:
        assert root_span.name == "workflow_root"
        assert root_span.attributes["run_id"] == "run-trace-1"
        assert root_span.attributes["correlation_id"] == "corr-1"

        ctx = get_current_trace_context()
        assert ctx is not None
        assert ctx.parent_span_id == root_span.span_id

        async with tracer.trace_scope("child_planner", kind=SpanKind.AGENT) as child_span:
            assert child_span.parent_span_id == root_span.span_id
            assert child_span.trace_id == root_span.trace_id
            child_span.set_attribute("tasks_planned", 3)

    spans = tracer.get_spans(run_id="run-trace-1")
    assert len(spans) == 2

    child = next(s for s in spans if s.name == "child_planner")
    root = next(s for s in spans if s.name == "workflow_root")

    assert child.parent_span_id == root.span_id
    assert child.status == SpanStatus.OK
    assert child.duration_ms is not None and child.duration_ms >= 0.0
    assert root.duration_ms is not None and root.duration_ms >= 0.0
    assert child.attributes["tasks_planned"] == 3


@pytest.mark.asyncio
async def test_tracing_error_capture() -> None:
    tracer = InMemoryTracer()

    with pytest.raises(ValueError, match="Intentional child error"):
        async with tracer.trace_scope("failing_span", kind=SpanKind.TOOL, run_id="run-err"):
            raise ValueError("Intentional child error")

    spans = tracer.get_spans(run_id="run-err")
    assert len(spans) == 1
    assert spans[0].status == SpanStatus.ERROR
    assert "Intentional child error" in (spans[0].error_message or "")
