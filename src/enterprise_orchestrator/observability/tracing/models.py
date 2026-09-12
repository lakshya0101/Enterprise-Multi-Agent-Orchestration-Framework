"""Distributed tracing domain models and span definitions."""

import time
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SpanKind(str, Enum):
    """Category of execution span."""

    WORKFLOW = "workflow"
    NODE = "node"
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    VALIDATOR = "validator"
    HUMAN_GATE = "human_gate"


class SpanStatus(str, Enum):
    """Terminal status of an execution span."""

    OK = "OK"
    ERROR = "ERROR"


class Span(BaseModel):
    """Structured execution span representing an operation unit."""

    span_id: str = Field(default_factory=lambda: str(uuid4())[:16])
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str
    kind: SpanKind = SpanKind.NODE
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    error_message: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        """Add or update an attribute on this span."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event timestamp to this span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def finish(self, status: SpanStatus = SpanStatus.OK, error_message: Optional[str] = None) -> None:
        """Mark span as finished and calculate duration."""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration_ms = max(0.0, (self.end_time - self.start_time) * 1000.0)
            self.status = status
            if error_message:
                self.error_message = error_message
