"""In-memory tracer storing span trees for offline analysis and unit test assertion."""

import threading
from typing import List, Optional

from enterprise_orchestrator.observability.audit.redaction import SensitiveDataRedactor
from enterprise_orchestrator.observability.tracing.base import BaseTracer
from enterprise_orchestrator.observability.tracing.models import Span


class InMemoryTracer(BaseTracer):
    """Thread-safe in-memory tracer implementation."""

    def __init__(self, redactor: Optional[SensitiveDataRedactor] = None) -> None:
        super().__init__(redactor=redactor)
        self._spans: List[Span] = []
        self._lock = threading.Lock()

    def _record_start(self, span: Span) -> None:
        pass

    def _record_end(self, span: Span) -> None:
        with self._lock:
            self._spans.append(span)

    def get_spans(self, trace_id: Optional[str] = None, run_id: Optional[str] = None) -> List[Span]:
        with self._lock:
            spans = list(self._spans)

        if trace_id:
            spans = [s for s in spans if s.trace_id == trace_id]
        if run_id:
            spans = [s for s in spans if s.attributes.get("run_id") == run_id]
        return spans

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
