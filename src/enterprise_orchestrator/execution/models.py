"""Execution models and lifecycle types for the execution boundary."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional


class JobStatus(str, Enum):
    """Volatile operational status of an execution job inside the execution engine."""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionJob:
    """Operational representation of a workflow execution job in the backend."""

    job_id: str
    run_id: str
    request: str
    status: JobStatus = JobStatus.QUEUED
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    custom_context: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    is_cancelled: bool = False
    cancellation_reason: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Any] = None
