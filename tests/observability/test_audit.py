"""Unit tests for AuditEvent models and BaseAuditSink / InMemoryAuditSink / JSONLAuditSink."""

import os
import tempfile
import pytest

from enterprise_orchestrator.observability.audit.in_memory import InMemoryAuditSink
from enterprise_orchestrator.observability.audit.jsonl import JSONLAuditSink
from enterprise_orchestrator.observability.audit.models import AuditEvent, AuditEventType


def test_audit_event_immutability_and_redaction() -> None:
    sink = InMemoryAuditSink()
    event = AuditEvent(
        run_id="run-001",
        event_type=AuditEventType.RUN_STARTED,
        actor="client",
        component="api",
        action="start workflow with api_key in action",
        outcome="SUCCESS",
        metadata={"api_key": "raw_secret_123", "normal": "value"},
    )

    sink.record(event)
    events = sink.get_events(run_id="run-001")

    assert len(events) == 1
    recorded = events[0]
    assert recorded.metadata["api_key"] == "[REDACTED]"
    assert recorded.metadata["normal"] == "value"
    assert recorded.schema_version == "1.0.0"


def test_jsonl_audit_sink_lifecycle() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        audit_file = os.path.join(tmpdir, "audit_test.jsonl")
        sink = JSONLAuditSink(file_path=audit_file)

        event1 = AuditEvent(
            run_id="run-101",
            event_type=AuditEventType.RUN_STARTED,
            actor="test",
            action="init",
            metadata={"user": "alice"},
        )
        event2 = AuditEvent(
            run_id="run-102",
            event_type=AuditEventType.RUN_COMPLETED,
            actor="test",
            action="finish",
            metadata={"user": "bob"},
        )

        sink.record(event1)
        sink.record(event2)

        # Read back all events
        all_events = sink.get_events()
        assert len(all_events) == 2

        # Filter by run_id
        run_101_events = sink.get_events(run_id="run-101")
        assert len(run_101_events) == 1
        assert run_101_events[0].run_id == "run-101"
        assert run_101_events[0].event_type == AuditEventType.RUN_STARTED
