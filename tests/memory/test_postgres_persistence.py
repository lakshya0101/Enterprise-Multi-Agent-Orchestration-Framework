"""Offline unit and contract tests for PostgreSQLStateStore persistence."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from enterprise_orchestrator.core.human import HumanEscalationRequest
from enterprise_orchestrator.core.metadata import ExecutionMetadata
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import ExecutionPlan, Task
from enterprise_orchestrator.core.types import (
    ExecutionStatus,
    HumanRequestStatus,
    TaskStatus,
    TaskType,
)
from enterprise_orchestrator.errors.exceptions import ConfigurationError, OrchestrationError
from enterprise_orchestrator.memory.base import Checkpoint
from enterprise_orchestrator.memory.postgres import PostgreSQLStateStore


class MockAsyncPGConnection:
    """In-memory mock of an asyncpg connection for 100% offline deterministic testing."""

    def __init__(self) -> None:
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.checkpoints: Dict[str, Dict[str, Any]] = {}

    async def execute(self, query: str, *args: Any) -> str:
        query_upper = query.strip().upper()
        if "CREATE TABLE" in query_upper or "CREATE INDEX" in query_upper:
            return "CREATE TABLE"

        elif "INSERT INTO RUNS" in query_upper:
            run_id, status, created_at, updated_at, state_json = args
            self.runs[run_id] = {
                "run_id": run_id,
                "status": status,
                "created_at": created_at,
                "updated_at": updated_at,
                "state_json": state_json,
            }
            return "INSERT 1"

        elif "INSERT INTO CHECKPOINTS" in query_upper:
            checkpoint_id, run_id, step, created_at, checkpoint_json = args
            self.checkpoints[checkpoint_id] = {
                "checkpoint_id": checkpoint_id,
                "run_id": run_id,
                "step": step,
                "created_at": created_at,
                "checkpoint_json": checkpoint_json,
            }
            return "INSERT 1"

        elif "DELETE FROM RUNS" in query_upper:
            run_id = args[0]
            deleted = 0
            if run_id in self.runs:
                del self.runs[run_id]
                deleted = 1
            # Cascade delete checkpoints
            to_del = [cid for cid, cp in self.checkpoints.items() if cp["run_id"] == run_id]
            for cid in to_del:
                del self.checkpoints[cid]
            return f"DELETE {deleted}"

        elif "DELETE FROM CHECKPOINTS" in query_upper:
            run_id = args[0]
            to_del = [cid for cid, cp in self.checkpoints.items() if cp["run_id"] == run_id]
            for cid in to_del:
                del self.checkpoints[cid]
            return f"DELETE {len(to_del)}"

        return "OK"

    async def fetchrow(self, query: str, *args: Any) -> Optional[Dict[str, Any]]:
        query_upper = query.strip().upper()
        if "FROM RUNS" in query_upper:
            run_id = args[0]
            return self.runs.get(run_id)
        elif "FROM CHECKPOINTS" in query_upper:
            checkpoint_id = args[0]
            return self.checkpoints.get(checkpoint_id)
        return None

    async def fetch(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        query_upper = query.strip().upper()
        if "FROM CHECKPOINTS" in query_upper:
            run_id = args[0]
            matching = [cp for cp in self.checkpoints.values() if cp["run_id"] == run_id]
            matching.sort(key=lambda x: (x["step"], str(x["created_at"])))
            return matching
        return []


class MockAsyncPGPool:
    """Mock connection pool yielding MockAsyncPGConnection in async context manager."""

    def __init__(self) -> None:
        self.conn = MockAsyncPGConnection()

    @asynccontextmanager
    async def acquire(self):
        yield self.conn

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_postgres_missing_dependency_raises_configuration_error() -> None:
    """Verify clean ConfigurationError when asyncpg is not installed."""
    store = PostgreSQLStateStore(pool=None, auto_init_schema=False)
    with patch.dict("sys.modules", {"asyncpg": None}):
        with patch("builtins.__import__", side_effect=ImportError("No module named 'asyncpg'")):
            with pytest.raises(ConfigurationError) as exc_info:
                await store._get_pool()
            assert "asyncpg" in str(exc_info.value).lower()
            assert exc_info.value.code == "MISSING_POSTGRES_DEPENDENCY"


@pytest.mark.asyncio
async def test_postgres_state_store_nested_state_roundtrip() -> None:
    """Verify complex nested OrchestrationState can be saved and retrieved from PostgreSQL."""
    pool = MockAsyncPGPool()
    store = PostgreSQLStateStore(pool=pool, auto_init_schema=True)

    task1 = Task(
        id="task_001",
        title="Analyze Financial Ratios",
        task_type=TaskType.TOOL_EXECUTION,
        status=TaskStatus.COMPLETED,
        output_data={"ratio": 1.45, "status": "healthy"},
    )
    task2 = Task(
        id="task_002",
        title="Synthesize Summary",
        task_type=TaskType.SYNTHESIS,
        status=TaskStatus.PENDING,
        dependencies=["task_001"],
    )
    plan = ExecutionPlan(
        rationale="2-step financial audit workflow",
        tasks=[task1, task2],
    )

    state = OrchestrationState.create_initial(
        request="Run Q3 financial audit",
        correlation_id="corr-pg-001",
        session_id="sess-pg-001",
        custom_context={"department": "finance", "priority": "high"},
    )
    state.plan = plan
    state.current_step = 3
    state.add_tool_result("calc_ratio", {"ratio": 1.45})
    state.request_human_escalation(
        reason="Confirm high dollar threshold approval.",
        requested_action="Approve disbursement.",
        context={"threshold": 50000},
        task_id="task_002",
    )

    # 1. Save State
    await store.save_state(state)

    # 2. Retrieve State
    reloaded = await store.get_state(state.run_id)
    assert reloaded is not None
    assert reloaded.run_id == state.run_id
    assert reloaded.request == "Run Q3 financial audit"
    assert reloaded.metadata.correlation_id == "corr-pg-001"
    assert reloaded.metadata.session_id == "sess-pg-001"
    assert reloaded.custom_context["department"] == "finance"
    assert reloaded.current_step == 3
    assert len(reloaded.tool_results) == 1
    assert reloaded.tool_results["calc_ratio"] == {"ratio": 1.45}

    # 3. Verify DAG plan deserialization
    assert reloaded.plan is not None
    assert len(reloaded.plan.tasks) == 2
    assert reloaded.plan.tasks[0].id == "task_001"
    assert reloaded.plan.tasks[0].status == TaskStatus.COMPLETED
    assert reloaded.plan.tasks[0].output_data == {"ratio": 1.45, "status": "healthy"}
    assert reloaded.plan.tasks[1].dependencies == ["task_001"]

    # 4. Verify Human Escalation request deserialization
    assert len(reloaded.human_requests) == 1
    assert reloaded.human_requests[0].reason == "Confirm high dollar threshold approval."
    assert reloaded.requires_human is True


@pytest.mark.asyncio
async def test_postgres_state_store_checkpoints_lifecycle() -> None:
    """Verify saving, retrieving, and listing execution checkpoints in PostgreSQL."""
    pool = MockAsyncPGPool()
    store = PostgreSQLStateStore(pool=pool, auto_init_schema=True)

    state = OrchestrationState.create_initial(request="Checkpoint lifecycle test")
    run_id = state.run_id

    cp0 = Checkpoint(run_id=run_id, step=0, state=state, metadata={"transition": "START"})
    state.increment_step(current_agent="planner")
    cp1 = Checkpoint(run_id=run_id, step=1, state=state, metadata={"transition": "PLAN"})
    state.increment_step(current_agent="retriever")
    cp2 = Checkpoint(run_id=run_id, step=2, state=state, metadata={"transition": "RETRIEVAL"})

    await store.save_checkpoint(cp0)
    await store.save_checkpoint(cp1)
    await store.save_checkpoint(cp2)

    # 1. Fetch individual checkpoint
    single_cp = await store.get_checkpoint(cp1.checkpoint_id)
    assert single_cp is not None
    assert single_cp.checkpoint_id == cp1.checkpoint_id
    assert single_cp.step == 1
    assert single_cp.metadata["transition"] == "PLAN"

    # 2. List all checkpoints sorted by step
    checkpoints = await store.list_checkpoints(run_id)
    assert len(checkpoints) == 3
    assert [c.step for c in checkpoints] == [0, 1, 2]
    assert checkpoints[0].metadata["transition"] == "START"
    assert checkpoints[2].metadata["transition"] == "RETRIEVAL"

    # 3. Non-existent checkpoint returns None
    missing = await store.get_checkpoint("non_existent_id")
    assert missing is None


@pytest.mark.asyncio
async def test_postgres_state_store_delete_run_cascades() -> None:
    """Verify delete_run purges state and checkpoints."""
    pool = MockAsyncPGPool()
    store = PostgreSQLStateStore(pool=pool, auto_init_schema=True)

    state = OrchestrationState.create_initial(request="Delete run test")
    await store.save_state(state)
    cp = Checkpoint(run_id=state.run_id, step=0, state=state)
    await store.save_checkpoint(cp)

    assert await store.get_state(state.run_id) is not None
    assert len(await store.list_checkpoints(state.run_id)) == 1

    # Purge run
    deleted = await store.delete_run(state.run_id)
    assert deleted is True

    # Confirm clean deletion
    assert await store.get_state(state.run_id) is None
    assert len(await store.list_checkpoints(state.run_id)) == 0


@pytest.mark.asyncio
async def test_postgres_state_store_error_handling() -> None:
    """Verify database exceptions are mapped to classified OrchestrationError instances."""
    faulty_pool = MagicMock()
    faulty_conn = AsyncMock()
    faulty_conn.execute.side_effect = RuntimeError("Connection closed unexpectedly")
    faulty_conn.fetchrow.side_effect = RuntimeError("Connection closed unexpectedly")

    @asynccontextmanager
    async def faulty_acquire():
        yield faulty_conn

    faulty_pool.acquire = faulty_acquire
    store = PostgreSQLStateStore(pool=faulty_pool, auto_init_schema=False)

    state = OrchestrationState.create_initial(request="Error handling test")

    with pytest.raises(OrchestrationError) as exc_info:
        await store.save_state(state)
    assert exc_info.value.code == "POSTGRES_SAVE_STATE_ERROR"
    assert exc_info.value.retryable is True

    with pytest.raises(OrchestrationError) as exc_info:
        await store.get_state(state.run_id)
    assert exc_info.value.code == "POSTGRES_GET_STATE_ERROR"
