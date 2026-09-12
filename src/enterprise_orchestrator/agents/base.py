"""Base agent interface contract."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from enterprise_orchestrator.agents.result import AgentResult
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import Task
from enterprise_orchestrator.core.types import AgentRole


class AgentMetadata(BaseModel):
    """Declarative capability metadata defining an agent contract."""

    name: str = Field(..., min_length=1, description="Unique programmatic name of the agent")
    role: AgentRole = Field(..., description="Role categorization in the orchestration hierarchy")
    description: str = Field(..., description="Human and supervisor readable capability description")
    capabilities: List[str] = Field(default_factory=list, description="Tags or capabilities supported by this agent")
    input_schema_desc: str = Field(default="", description="Description or JSON schema of expected inputs")
    output_schema_desc: str = Field(default="", description="Description or JSON schema of produced outputs")
    version: str = Field(default="0.1.0", description="Agent implementation version")


class BaseAgent(ABC):
    """Abstract base class for all specialized framework agents."""

    def __init__(self, metadata: AgentMetadata) -> None:
        self.metadata = metadata

    @property
    def name(self) -> str:
        """Convenience accessor for agent name."""
        return self.metadata.name

    @property
    def role(self) -> AgentRole:
        """Convenience accessor for agent role."""
        return self.metadata.role

    @abstractmethod
    def validate_input(self, state: OrchestrationState, task: Optional[Task] = None) -> bool:
        """Pre-execution validation to ensure prerequisites and schemas are satisfied."""
        pass

    @abstractmethod
    async def execute(self, state: OrchestrationState, task: Optional[Task] = None) -> AgentResult:
        """Execute agent business logic against the provided orchestration state and task."""
        pass
