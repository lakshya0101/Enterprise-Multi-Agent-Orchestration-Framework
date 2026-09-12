"""Audit logging domain models and event taxonomy."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AuditEventType(str, Enum):
    """Standardized event taxonomy for framework governance and audit trails."""

    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    PLAN_CREATED = "plan_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TOOL_INVOKED = "tool_invoked"
    RETRIEVAL_EXECUTED = "retrieval_executed"
    VALIDATION_VERDICT = "validation_verdict"
    RETRY_TRIGGERED = "retry_triggered"
    HUMAN_ESCALATED = "human_escalated"
    HUMAN_DECISION_APPLIED = "human_decision_applied"


class AuditEvent(BaseModel):
    """Immutable structured audit event record."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    event_type: AuditEventType
    actor: str = "orchestrator"
    component: str = "core"
    action: str
    outcome: str = "SUCCESS"  # SUCCESS, FAILURE, REJECTED, ESCALATED, MODIFIED
    metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0.0"
