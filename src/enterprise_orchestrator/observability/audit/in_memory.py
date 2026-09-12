"""In-memory append-only audit sink for tests and in-process logging."""

import threading
from typing import List, Optional

from enterprise_orchestrator.observability.audit.base import BaseAuditSink
from enterprise_orchestrator.observability.audit.models import AuditEvent
from enterprise_orchestrator.observability.audit.redaction import SensitiveDataRedactor


class InMemoryAuditSink(BaseAuditSink):
    """Thread-safe in-memory audit sink storing structured AuditEvents."""

    def __init__(self, redactor: Optional[SensitiveDataRedactor] = None) -> None:
        super().__init__(redactor=redactor)
        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()

    def _emit(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)

    def get_events(self, run_id: Optional[str] = None) -> List[AuditEvent]:
        with self._lock:
            if run_id:
                return [e for e in self._events if e.run_id == run_id]
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
