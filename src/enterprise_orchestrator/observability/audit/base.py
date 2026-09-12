"""Abstract audit sink interface enforcing pre-sink redaction."""

from abc import ABC, abstractmethod
from typing import List, Optional

from enterprise_orchestrator.observability.audit.models import AuditEvent
from enterprise_orchestrator.observability.audit.redaction import SensitiveDataRedactor


class BaseAuditSink(ABC):
    """Abstract base class for all audit sinks."""

    def __init__(self, redactor: Optional[SensitiveDataRedactor] = None) -> None:
        self.redactor = redactor or SensitiveDataRedactor()

    def sanitize_event(self, event: AuditEvent) -> AuditEvent:
        """Enforce redaction on audit event metadata and fields before recording."""
        sanitized_meta = self.redactor.redact(event.metadata)
        sanitized_action = self.redactor.redact_text(event.action)
        return event.model_copy(
            update={
                "metadata": sanitized_meta,
                "action": sanitized_action,
            }
        )

    def record(self, event: AuditEvent) -> None:
        """Public entry point: sanitizes event and emits to concrete sink safely."""
        try:
            sanitized = self.sanitize_event(event)
            self._emit(sanitized)
        except Exception:
            # Audit recording must never crash or raise exceptions to the caller
            pass

    @abstractmethod
    def _emit(self, event: AuditEvent) -> None:
        """Internal concrete emission logic."""
        pass

    @abstractmethod
    def get_events(self, run_id: Optional[str] = None) -> List[AuditEvent]:
        """Retrieve audit events optionally filtered by run ID."""
        pass
