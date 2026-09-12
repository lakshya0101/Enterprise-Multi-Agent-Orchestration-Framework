"""Orchestrator interface contracts and event definitions."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from enterprise_orchestrator.core.human import HumanDecision
from enterprise_orchestrator.core.metadata import utc_now
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import Task


class OrchestrationEventType(str, Enum):
    """Lifecycle event types emitted by the supervisor."""

    RUN_STARTED = "run_started"
    PLAN_GENERATED = "plan_generated"
    TASK_ROUTED = "task_routed"
    AGENT_EXECUTION_COMPLETED = "agent_execution_completed"
    VALIDATION_EVALUATED = "validation_evaluated"
    RETRY_TRIGGERED = "retry_triggered"
    HUMAN_ESCALATION_TRIGGERED = "human_escalation_triggered"
    HUMAN_DECISION_APPLIED = "human_decision_applied"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


class OrchestrationEvent(BaseModel):
    """Structured telemetry/audit event emitted during execution."""

    event_type: OrchestrationEventType = Field(..., description="Classification of lifecycle event")
    run_id: str = Field(..., description="Active run identifier")
    step: int = Field(..., ge=0, description="Step index when event fired")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event-specific diagnostic data")
    timestamp: Any = Field(default_factory=utc_now, description="Timestamp of event occurrence")


class BaseOrchestrator(ABC):
    """Abstract contract for the centralized supervisor orchestrator."""

    @abstractmethod
    def initialize_state(
        self,
        request: str,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> OrchestrationState:
        """Create and initialize a new orchestration state."""
        pass

    @abstractmethod
    async def route(self, state: OrchestrationState) -> Optional[Task]:
        """Determine next ready task from execution plan based on dependency graph."""
        pass

    @abstractmethod
    async def step(self, state: OrchestrationState) -> OrchestrationState:
        """Execute a single atomic step in the orchestration cycle."""
        pass

    @abstractmethod
    async def run(self, request: str, correlation_id: Optional[str] = None) -> OrchestrationState:
        """Execute the entire end-to-end orchestration loop until completion or escalation."""
        pass

    @abstractmethod
    async def resume_with_human_decision(
        self,
        state: OrchestrationState,
        decision: HumanDecision,
    ) -> OrchestrationState:
        """Resume execution of a paused state following a human review decision."""
        pass
