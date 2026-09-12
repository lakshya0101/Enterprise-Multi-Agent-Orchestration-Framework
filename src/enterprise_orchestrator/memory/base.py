"""Base state persistence and checkpointing contracts."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from enterprise_orchestrator.core.metadata import utc_now
from enterprise_orchestrator.core.state import OrchestrationState


class Checkpoint(BaseModel):
    """Point-in-time snapshot of orchestration state at a specific execution step."""

    checkpoint_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique ID for this checkpoint")
    run_id: str = Field(..., description="Run ID associated with this checkpoint")
    step: int = Field(..., ge=0, description="Step index when snapshot was taken")
    state: OrchestrationState = Field(..., description="Complete snapshot of orchestration state")
    timestamp: datetime = Field(default_factory=utc_now, description="Timestamp when checkpoint was recorded")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic and transition metadata")


class BaseStateStore(ABC):
    """Abstract storage interface for orchestration state and execution history."""

    @abstractmethod
    async def save_state(self, state: OrchestrationState) -> None:
        """Persist or update the latest state of an active orchestration run."""
        pass

    @abstractmethod
    async def get_state(self, run_id: str) -> Optional[OrchestrationState]:
        """Fetch the latest persisted state for a given run ID."""
        pass

    @abstractmethod
    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Record an immutable state snapshot at a specific step."""
        pass

    @abstractmethod
    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Retrieve a specific checkpoint by ID."""
        pass

    @abstractmethod
    async def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        """List all historical checkpoints recorded for a run in chronological order."""
        pass

    @abstractmethod
    async def delete_run(self, run_id: str) -> bool:
        """Purge state and associated checkpoints for a run."""
        pass
