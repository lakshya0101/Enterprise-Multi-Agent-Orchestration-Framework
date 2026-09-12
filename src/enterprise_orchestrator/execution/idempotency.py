"""Process-local atomic idempotency ledger protected by asyncio.Lock."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, Optional, Tuple

from enterprise_orchestrator.errors.exceptions import OrchestrationError


class IdempotencyConflictError(OrchestrationError):
    """Raised when an idempotency key is reused with a conflicting payload."""

    def __init__(self, message: str = "Idempotency key already used with a different request payload.") -> None:
        super().__init__(message=message, code="IDEMPOTENCY_CONFLICT", retryable=False)


@dataclass
class IdempotencyRecord:
    """In-memory record tracking an in-flight or completed idempotent execution."""

    key_hash: str
    request_hash: str
    run_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    future: Optional[asyncio.Future] = None
    completed: bool = False
    cached_result: Optional[Any] = None


class InMemoryIdempotencyLedger:
    """Process-local atomic idempotency ledger.

    Guarantees:
        1. Atomic deduplication within a single running application process.
        2. Concurrent identical submissions wait on a single in-flight future.
        3. Conflicting payload hashes with the same key raise IdempotencyConflictError.
        4. Cross-instance / distributed deduplication is explicitly NOT guaranteed.
    """

    def __init__(self) -> None:
        self._records: Dict[str, IdempotencyRecord] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def compute_key_hash(subject_id: str, idempotency_key: str) -> str:
        """Compute deterministic scoped hash from subject identity and client key."""
        raw = f"{subject_id}:{idempotency_key}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_request_hash(request: str, custom_context: Optional[Dict[str, Any]] = None) -> str:
        """Compute deterministic hash of the request payload."""
        import json
        payload = {
            "request": request,
            "custom_context": custom_context or {},
        }
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    async def acquire_or_join(
        self,
        key_hash: str,
        request_hash: str,
        run_id: str,
    ) -> Tuple[bool, asyncio.Future, Optional[Any]]:
        """Atomically inspect or claim an idempotency key."""
        loop = asyncio.get_running_loop()
        async with self._lock:
            if key_hash in self._records:
                record = self._records[key_hash]
                if record.request_hash != request_hash:
                    raise IdempotencyConflictError()

                if record.completed:
                    return False, record.future, record.cached_result
                return False, record.future, None

            future = loop.create_future()
            record = IdempotencyRecord(
                key_hash=key_hash,
                request_hash=request_hash,
                run_id=run_id,
                future=future,
            )
            self._records[key_hash] = record
            return True, record.future, None

    async def complete(self, key_hash: str, result: Any) -> None:
        """Mark an idempotency record as completed with cached result."""
        async with self._lock:
            if key_hash in self._records:
                record = self._records[key_hash]
                record.completed = True
                record.cached_result = result
                if not record.future.done():
                    record.future.set_result(result)

    async def fail(self, key_hash: str, error: Exception) -> None:
        """Mark an idempotency record as failed or release it."""
        async with self._lock:
            if key_hash in self._records:
                record = self._records.pop(key_hash)
                if not record.future.done():
                    record.future.set_exception(error)

    async def clear(self) -> None:
        """Reset ledger (primarily for test cleanup)."""
        async with self._lock:
            self._records.clear()
