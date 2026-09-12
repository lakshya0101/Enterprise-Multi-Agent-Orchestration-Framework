"""Explicit Pydantic API request and response models."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import json
from pydantic import BaseModel, Field, field_validator


# --- Health & Readiness ---

class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = Field(default="ok", description="Service health indicator")
    version: str = Field(default="0.1.0", description="Framework version")
    timestamp: datetime = Field(..., description="Current server UTC timestamp")


class ReadinessResponse(BaseModel):
    """Readiness probe response verifying configuration readiness."""

    status: str = Field(default="ready", description="Service readiness indicator")
    state_store_type: str = Field(..., description="Configured state persistence store")
    vector_store_type: str = Field(..., description="Configured vector store type")
    default_llm_provider: str = Field(..., description="Configured default LLM provider")


# --- Run Management ---

class CreateRunRequest(BaseModel):
    """Payload to initiate a new multi-agent orchestration workflow."""

    request: str = Field(..., min_length=1, max_length=50000, description="Goal or instruction for the multi-agent system")
    correlation_id: Optional[str] = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-\.:]+$",
        description="Client-provided tracking correlation ID",
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-\.:]+$",
        description="Optional user or chat session ID",
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional client idempotency key for deduplication",
    )
    custom_context: Optional[Dict[str, Any]] = Field(default=None, description="Additional contextual parameters")

    @field_validator("custom_context")
    @classmethod
    def validate_custom_context_size(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is not None:
            if len(v) > 50:
                raise ValueError("custom_context cannot contain more than 50 top-level keys.")
            serialized = json.dumps(v)
            if len(serialized) > 65536:  # 64 KB
                raise ValueError("custom_context serialized size exceeds maximum limit of 64 KB.")
        return v


class TaskSummary(BaseModel):
    """Summary representation of an execution DAG task."""

    id: str
    title: str
    task_type: str
    status: str
    dependencies: List[str] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PlanSummary(BaseModel):
    """Summary of the decomposed DAG execution plan."""

    rationale: str
    tasks: List[TaskSummary]


class RunResponse(BaseModel):
    """Detailed response containing complete state of an orchestration run."""

    run_id: str
    request: str
    status: str
    current_step: int
    plan: Optional[PlanSummary] = None
    requires_human: bool
    final_response: Optional[str] = None
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class RunStatusResponse(BaseModel):
    """Lightweight operational status response for a run."""

    run_id: str
    status: str
    current_step: int
    requires_human: bool
    has_plan: bool
    completed_tasks: int
    total_tasks: int


class CreateRunResponse(BaseModel):
    """Initial response returned upon run creation."""

    run_id: str
    status: str
    created_at: datetime


# --- Checkpoints ---

class CheckpointResponse(BaseModel):
    """Representation of an execution step checkpoint."""

    checkpoint_id: str
    run_id: str
    step: int
    timestamp: datetime
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CheckpointListResponse(BaseModel):
    """List of all checkpoints for a run."""

    run_id: str
    total_checkpoints: int
    checkpoints: List[CheckpointResponse]


# --- Human-in-the-Loop ---

class HumanReviewResponse(BaseModel):
    """Payload representing a pending human review escalation."""

    request_id: str
    run_id: str
    task_id: Optional[str] = None
    reason: str
    requested_action: str
    context: Dict[str, Any] = Field(default_factory=dict)
    status: str
    created_at: datetime


class HumanDecisionRequest(BaseModel):
    """Human reviewer verdict and commentary."""

    request_id: str = Field(
        ...,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description="ID of the escalation request being resolved",
    )
    status: str = Field(
        ...,
        pattern=r"^(?i)(APPROVED|REJECTED|MODIFIED)$",
        description="Verdict: APPROVED, REJECTED, or MODIFIED",
    )
    decision_note: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Reviewer commentary",
    )
    modified_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated parameters if MODIFIED",
    )
    reviewer_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Identifier of the human reviewer (bound server-side)",
    )

    @field_validator("modified_payload")
    @classmethod
    def validate_modified_payload_size(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is not None:
            serialized = json.dumps(v)
            if len(serialized) > 131072:  # 128 KB
                raise ValueError("modified_payload serialized size exceeds maximum limit of 128 KB.")
        return v


class HumanDecisionResponse(BaseModel):
    """Result of submitting a human decision."""

    run_id: str
    status: str
    resumed_successfully: bool
    final_response: Optional[str] = None


# --- Error Handling ---

class ErrorResponse(BaseModel):
    """Standardized API error envelope."""

    error: str = Field(..., description="Human-readable error explanation")
    code: str = Field(..., description="Machine-readable error classification code")
    details: Optional[Any] = Field(default=None, description="Optional diagnostic details")
