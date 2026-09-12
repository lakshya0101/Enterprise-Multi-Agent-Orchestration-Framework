"""Local JSONL append-only audit sink for persistent disk audit trails."""

import os
import threading
from typing import List, Optional

from enterprise_orchestrator.observability.audit.base import BaseAuditSink
from enterprise_orchestrator.observability.audit.models import AuditEvent
from enterprise_orchestrator.observability.audit.redaction import SensitiveDataRedactor


class JSONLAuditSink(BaseAuditSink):
    """Appends serialized JSON lines to a local file audit store."""

    def __init__(
        self,
        file_path: str = "data/audit/audit.jsonl",
        redactor: Optional[SensitiveDataRedactor] = None,
    ) -> None:
        super().__init__(redactor=redactor)
        self.file_path = file_path
        self._lock = threading.Lock()
        parent_dir = os.path.dirname(os.path.abspath(file_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    def _emit(self, event: AuditEvent) -> None:
        with self._lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")

    def get_events(self, run_id: Optional[str] = None) -> List[AuditEvent]:
        if not os.path.exists(self.file_path):
            return []

        events: List[AuditEvent] = []
        with self._lock:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            ev = AuditEvent.model_validate_json(line)
                            if run_id is None or ev.run_id == run_id:
                                events.append(ev)
                        except Exception:
                            continue
        return events
