"""Execution metadata model for tracking, observability, and auditability."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from enterprise_orchestrator.core.types import ExecutionStatus


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


class ExecutionMetadata(BaseModel):
    """Serializable execution metadata captured during orchestration runs."""

    run_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique ID for the orchestration run")
    correlation_id: Optional[str] = Field(default=None, description="External correlation / tracing identifier")
    session_id: Optional[str] = Field(default=None, description="User or conversation session ID")
    created_at: datetime = Field(default_factory=utc_now, description="Timestamp when run was initiated")
    updated_at: datetime = Field(default_factory=utc_now, description="Timestamp of latest state update")
    current_agent: Optional[str] = Field(default=None, description="Name of the currently active agent")
    current_step: int = Field(default=0, ge=0, description="Monotonically increasing step counter")
    retry_count: int = Field(default=0, ge=0, description="Cumulative retry count across all steps")
    max_retries: int = Field(default=3, ge=0, description="Configured maximum allowable retries")
    total_tokens_used: int = Field(default=0, ge=0, description="Total tokens consumed across all LLM calls")
    execution_time_ms: float = Field(default=0.0, ge=0.0, description="Total execution duration in milliseconds")
    status: ExecutionStatus = Field(default=ExecutionStatus.IDLE, description="Current execution status")
    custom_tags: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata and telemetry tags")

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = utc_now()
