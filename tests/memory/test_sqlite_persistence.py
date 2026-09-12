"""Unit and integration tests for SQLiteStateStore persistence and checkpointing."""

import pytest

from enterprise_orchestrator.core.human import HumanEscalationRequest
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import ExecutionPlan, Task
from enterprise_orchestrator.core.types import (
    AgentRole,
    ExecutionStatus,
    HumanRequestStatus,
    TaskStatus,
    TaskType,
)
from enterprise_orchestrator.errors.exceptions import OrchestrationError
from enterprise_orchestrator.memory.base import Checkpoint
from enterprise_orchestrator.memory.sqlite import SQLiteStateStore


@pytest.mark.asyncio
async def test_sqlite_state_store_nested_state_roundtrip(tmp_path):
    """Verify complete, deep nested serialization and deserialization of OrchestrationState."""
    db_path = str(tmp_path / "test_state.db")
    store = SQLiteStateStore(db_path=db_path)

    # 1. Build complex state with nested structures
    state = OrchestrationState.create_initial(
        request="Analyze quarterly finance and file tax audit",
        correlation_id="corr-999",
        session_id="sess-888",
        custom_context={"user_tier": "enterprise", "budget": 50000.0},
    )
    state.increment_step(current_agent="planner_agent")

    # Add execution plan with dependencies
    task_1 = Task(
        id="task_1",
        title="Fetch Tax Records",
        task_type=TaskType.RETRIEVAL,
        assigned_agent=AgentRole.RETRIEVER,
        status=TaskStatus.COMPLETED,
        input_data={"query": "2026 tax returns"},
        output_data={"records_found": 12},
    )
    task_2 = Task(
        id="task_2",
        title="Calculate Liabilities",
        task_type=TaskType.TOOL_EXECUTION,
        assigned_agent=AgentRole.TOOL_EXECUTOR,
        dependencies=["task_1"],
        status=TaskStatus.IN_PROGRESS,
        input_data={"tool_name": "tax_calc", "parameters": {"records": 12}},
        retry_count=1,
        max_retries=3,
    )
    state.plan = ExecutionPlan(rationale="Tax audit breakdown", tasks=[task_1, task_2])

    # Add context, tool outputs, and human escalation
    state.add_retrieved_context({"tax_docs": ["doc_a", "doc_b"]})
    state.add_tool_result("tax_calc", {"estimated_liability": 4500.0})
    state.request_human_escalation(
        reason="Liability exceeds threshold",
        requested_action="Authorize tax filing",
        context={"liability": 4500.0},
        task_id="task_2",
    )

    # 2. Save state
    await store.save_state(state)

    # 3. Retrieve and assert exact parity
    retrieved = await store.get_state(state.run_id)
    assert retrieved is not None
    assert retrieved.run_id == state.run_id
    assert retrieved.request == state.request
    assert retrieved.metadata.status == state.metadata.status
    assert retrieved.metadata.current_step == 1
    assert retrieved.custom_context == {"user_tier": "enterprise", "budget": 50000.0}

    # Verify nested Plan and Tasks
    assert retrieved.plan is not None
    assert len(retrieved.plan.tasks) == 2
    r_task_1 = retrieved.plan.get_task("task_1")
    r_task_2 = retrieved.plan.get_task("task_2")
    assert r_task_1.status == TaskStatus.COMPLETED
    assert r_task_2.status == TaskStatus.IN_PROGRESS
    assert r_task_2.dependencies == ["task_1"]
    assert r_task_2.retry_count == 1

    # Verify context, tool results, and human requests
    assert len(retrieved.retrieved_context) == 1
    assert len(retrieved.tool_results) == 1
    assert len(retrieved.human_requests) == 1
    assert retrieved.requires_human is True
    assert retrieved.human_requests[0].reason == "Liability exceeds threshold"


@pytest.mark.asyncio
async def test_sqlite_state_store_checkpoints_lifecycle(tmp_path):
    """Verify storing, retrieving, and chronologically listing step checkpoints."""
    db_path = str(tmp_path / "checkpoints.db")
    store = SQLiteStateStore(db_path=db_path)

    state = OrchestrationState.create_initial(request="Multi-step test")
    await store.save_state(state)

    cp_0 = Checkpoint(run_id=state.run_id, step=0, state=state, metadata={"event": "started"})
    state.increment_step("planner")
    cp_1 = Checkpoint(run_id=state.run_id, step=1, state=state, metadata={"event": "planned"})
    state.increment_step("retriever")
    cp_2 = Checkpoint(run_id=state.run_id, step=2, state=state, metadata={"event": "retrieved"})

    await store.save_checkpoint(cp_0)
    await store.save_checkpoint(cp_1)
    await store.save_checkpoint(cp_2)

    # Fetch individual checkpoint
    fetched_cp1 = await store.get_checkpoint(cp_1.checkpoint_id)
    assert fetched_cp1 is not None
    assert fetched_cp1.step == 1
    assert fetched_cp1.metadata["event"] == "planned"

    # List all checkpoints
    all_cps = await store.list_checkpoints(state.run_id)
    assert len(all_cps) == 3
    assert [c.step for c in all_cps] == [0, 1, 2]


@pytest.mark.asyncio
async def test_sqlite_state_store_multiple_runs_isolation_and_deletion(tmp_path):
    """Verify isolation across distinct runs and cascade deletion."""
    db_path = str(tmp_path / "isolation.db")
    store = SQLiteStateStore(db_path=db_path)

    state_a = OrchestrationState.create_initial(request="Run A")
    state_b = OrchestrationState.create_initial(request="Run B")

    await store.save_state(state_a)
    await store.save_state(state_b)
    await store.save_checkpoint(Checkpoint(run_id=state_a.run_id, step=0, state=state_a))
    await store.save_checkpoint(Checkpoint(run_id=state_b.run_id, step=0, state=state_b))

    assert (await store.get_state(state_a.run_id)) is not None
    assert (await store.get_state(state_b.run_id)) is not None
    assert len(await store.list_checkpoints(state_a.run_id)) == 1

    # Delete Run A
    deleted = await store.delete_run(state_a.run_id)
    assert deleted is True
    assert (await store.get_state(state_a.run_id)) is None
    assert len(await store.list_checkpoints(state_a.run_id)) == 0

    # Run B still intact
    assert (await store.get_state(state_b.run_id)) is not None
    assert len(await store.list_checkpoints(state_b.run_id)) == 1
