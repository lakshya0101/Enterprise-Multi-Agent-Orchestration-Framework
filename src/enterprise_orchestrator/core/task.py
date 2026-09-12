"""Task and execution plan models for hierarchical task decomposition and routing."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from enterprise_orchestrator.core.metadata import utc_now
from enterprise_orchestrator.core.types import AgentRole, TaskStatus, TaskType


class Task(BaseModel):
    """Discrete unit of work routed to a specialized agent."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the task")
    title: str = Field(..., min_length=1, description="Concise summary of the task")
    description: str = Field(default="", description="Detailed instructions or specifications for the task")
    task_type: TaskType = Field(default=TaskType.CUSTOM, description="Categorization of task nature")
    assigned_agent: Optional[AgentRole] = Field(default=None, description="Role of agent assigned to execute")
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks that must complete before this task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current lifecycle state of the task")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Structured input payloads passed to the agent")
    output_data: Optional[Dict[str, Any]] = Field(default=None, description="Structured output generated upon completion")
    error: Optional[str] = Field(default=None, description="Error message if task execution failed")
    retry_count: int = Field(default=0, ge=0, description="Number of times this specific task has been retried")
    max_retries: int = Field(default=3, ge=0, description="Maximum retries permitted for this task")
    created_at: datetime = Field(default_factory=utc_now, description="Timestamp when task was created")
    completed_at: Optional[datetime] = Field(default=None, description="Timestamp when task finished execution")

    def mark_in_progress(self) -> None:
        """Mark task as actively executing."""
        self.status = TaskStatus.IN_PROGRESS

    def mark_completed(self, output: Dict[str, Any]) -> None:
        """Mark task as successfully completed with output payload."""
        self.status = TaskStatus.COMPLETED
        self.output_data = output
        self.completed_at = utc_now()

    def mark_failed(self, error_message: str) -> None:
        """Mark task as failed with error context."""
        self.status = TaskStatus.FAILED
        self.error = error_message
        self.completed_at = utc_now()

    def can_retry(self) -> bool:
        """Check if task retry budget has not been exhausted."""
        return self.retry_count < self.max_retries


class ExecutionPlan(BaseModel):
    """Hierarchical execution plan holding planned tasks and graph dependency order."""

    plan_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique plan identifier")
    tasks: List[Task] = Field(default_factory=list, description="Ordered or dependency-linked list of tasks")
    current_task_id: Optional[str] = Field(default=None, description="ID of task currently executing")
    rationale: Optional[str] = Field(default=None, description="Planner reasoning and decomposition strategy")
    created_at: datetime = Field(default_factory=utc_now, description="Timestamp when plan was generated")

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve task by its ID."""
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def get_ready_tasks(self) -> List[Task]:
        """Return tasks that are PENDING and whose dependencies have COMPLETED."""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        ready = []
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                if all(dep in completed_ids for dep in task.dependencies):
                    ready.append(task)
        return ready

    def all_completed(self) -> bool:
        """Check if all planned tasks have successfully completed."""
        return bool(self.tasks) and all(t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED) for t in self.tasks)

    def has_failed_tasks(self) -> bool:
        """Check if any task has permanently failed."""
        return any(t.status == TaskStatus.FAILED for t in self.tasks)
