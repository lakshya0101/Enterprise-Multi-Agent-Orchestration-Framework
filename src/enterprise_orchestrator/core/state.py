"""Centralized orchestration state contract."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from enterprise_orchestrator.core.human import HumanDecision, HumanEscalationRequest
from enterprise_orchestrator.core.metadata import ExecutionMetadata
from enterprise_orchestrator.core.task import ExecutionPlan, Task
from enterprise_orchestrator.core.types import ExecutionStatus, HumanRequestStatus
from enterprise_orchestrator.validation.models import ValidationResult


class OrchestrationState(BaseModel):
    """Centralized, serializable state representation traversing the orchestration graph."""

    run_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique run identifier")
    request: str = Field(..., min_length=1, description="Original user prompt or business request")
    normalized_request: Optional[str] = Field(default=None, description="Preprocessed/disambiguated user intent")
    plan: Optional[ExecutionPlan] = Field(default=None, description="Hierarchical task execution plan")
    current_step: int = Field(default=0, ge=0, description="Current graph iteration / step index")
    agent_outputs: Dict[str, Any] = Field(default_factory=dict, description="Outputs mapped by agent name or role")
    tool_results: Dict[str, Any] = Field(default_factory=dict, description="Outputs mapped by tool execution ID/name")
    retrieved_context: List[Dict[str, Any]] = Field(default_factory=list, description="Documents and chunks fetched via RAG")
    validation_results: List[ValidationResult] = Field(default_factory=list, description="History of validation verdicts")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Structured log of non-fatal and fatal errors")
    retry_count: int = Field(default=0, ge=0, description="Current retry tally across the run")
    max_retries: int = Field(default=3, ge=0, description="Configured ceiling for retries before escalation/failure")
    requires_human: bool = Field(default=False, description="Flag indicating if workflow is paused for human review")
    human_requests: List[HumanEscalationRequest] = Field(default_factory=list, description="Queue of human escalation requests")
    final_response: Optional[str] = Field(default=None, description="Synthesized final response to user")
    metadata: ExecutionMetadata = Field(default_factory=ExecutionMetadata, description="Observability & audit metadata")
    custom_context: Dict[str, Any] = Field(default_factory=dict, description="Extensible runtime parameters")

    @classmethod
    def create_initial(
        cls,
        request: str,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        max_retries: int = 3,
        custom_context: Optional[Dict[str, Any]] = None,
    ) -> "OrchestrationState":
        """Factory method to initialize a pristine orchestration state."""
        run_id = str(uuid4())
        meta = ExecutionMetadata(
            run_id=run_id,
            correlation_id=correlation_id,
            session_id=session_id,
            max_retries=max_retries,
            status=ExecutionStatus.RUNNING,
        )
        return cls(
            run_id=run_id,
            request=request,
            max_retries=max_retries,
            metadata=meta,
            custom_context=custom_context or {},
        )

    def add_agent_output(self, agent_name: str, output: Any) -> None:
        """Record output from a specialized agent."""
        self.agent_outputs[agent_name] = output
        self.metadata.touch()

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        """Record result from tool execution."""
        self.tool_results[tool_name] = result
        self.metadata.touch()

    def add_retrieved_context(self, context_item: Dict[str, Any]) -> None:
        """Append retrieved document or knowledge chunk."""
        self.retrieved_context.append(context_item)
        self.metadata.touch()

    def add_validation_result(self, validation: ValidationResult) -> None:
        """Append a validation result."""
        self.validation_results.append(validation)
        self.metadata.touch()

    def record_error(self, error_dict: Dict[str, Any]) -> None:
        """Record a structured error."""
        self.errors.append(error_dict)
        self.metadata.touch()

    def increment_step(self, current_agent: Optional[str] = None) -> None:
        """Advance the step counter and update metadata."""
        self.current_step += 1
        self.metadata.current_step = self.current_step
        if current_agent:
            self.metadata.current_agent = current_agent
        self.metadata.touch()

    def request_human_escalation(
        self,
        reason: str,
        requested_action: str,
        context: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> HumanEscalationRequest:
        """Pause orchestration and generate a human escalation request."""
        req = HumanEscalationRequest(
            run_id=self.run_id,
            task_id=task_id,
            reason=reason,
            requested_action=requested_action,
            context=context or {},
        )
        self.human_requests.append(req)
        self.requires_human = True
        self.metadata.status = ExecutionStatus.PAUSED_FOR_HUMAN
        self.metadata.touch()
        return req

    def apply_human_decision(self, decision: HumanDecision) -> None:
        """Apply human reviewer decision and update state accordingly."""
        for req in self.human_requests:
            if req.request_id == decision.request_id:
                req.status = decision.status
                req.resolved_at = decision.resolved_at
                break

        # Check if any pending escalations remain
        has_pending = any(r.status == HumanRequestStatus.PENDING for r in self.human_requests)
        self.requires_human = has_pending
        if not has_pending:
            self.metadata.status = ExecutionStatus.RUNNING
        self.metadata.touch()

    def complete_run(self, final_response: str) -> None:
        """Mark orchestration run as successfully completed."""
        self.final_response = final_response
        self.metadata.status = ExecutionStatus.COMPLETED
        self.metadata.touch()

    def fail_run(self, error_summary: str) -> None:
        """Mark orchestration run as failed."""
        self.record_error({"fatal_error": error_summary})
        self.metadata.status = ExecutionStatus.FAILED
        self.metadata.touch()
