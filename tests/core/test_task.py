"""Tests for Task and ExecutionPlan contracts."""

from enterprise_orchestrator.core.task import ExecutionPlan, Task
from enterprise_orchestrator.core.types import AgentRole, TaskStatus, TaskType


def test_task_lifecycle_transitions():
    """Verify task state transitions."""
    task = Task(
        title="Fetch Sales Data",
        description="Retrieve raw sales JSON",
        task_type=TaskType.RETRIEVAL,
        assigned_agent=AgentRole.RETRIEVER,
        max_retries=2,
    )
    assert task.status == TaskStatus.PENDING
    assert task.can_retry() is True

    task.mark_in_progress()
    assert task.status == TaskStatus.IN_PROGRESS

    task.mark_completed(output={"sales_count": 100})
    assert task.status == TaskStatus.COMPLETED
    assert task.output_data == {"sales_count": 100}
    assert task.completed_at is not None


def test_task_failure_and_retry_budget():
    """Verify task failure and retry check."""
    task = Task(title="Execute Query", task_type=TaskType.TOOL_EXECUTION, max_retries=1)
    task.retry_count = 1
    assert task.can_retry() is False

    task.mark_failed("Database connection timeout")
    assert task.status == TaskStatus.FAILED
    assert task.error == "Database connection timeout"


def test_execution_plan_dependencies_and_readiness():
    """Verify execution plan ready tasks resolution based on DAG dependencies."""
    task1 = Task(id="t1", title="Step 1: Ingest", task_type=TaskType.RETRIEVAL)
    task2 = Task(id="t2", title="Step 2: Transform", task_type=TaskType.TOOL_EXECUTION, dependencies=["t1"])
    task3 = Task(id="t3", title="Step 3: Validate", task_type=TaskType.VALIDATION, dependencies=["t2"])

    plan = ExecutionPlan(tasks=[task1, task2, task3], rationale="Sequential pipeline")

    # Initially only t1 should be ready
    ready = plan.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "t1"
    assert plan.all_completed() is False

    # Complete t1
    task1.mark_completed({"raw_data": [1, 2, 3]})

    # Now t2 should be ready
    ready = plan.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "t2"

    # Complete t2
    task2.mark_completed({"cleaned_data": [1, 2, 3]})

    # Now t3 should be ready
    ready = plan.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "t3"

    # Complete t3
    task3.mark_completed({"verified": True})
    assert plan.all_completed() is True
    assert plan.has_failed_tasks() is False
