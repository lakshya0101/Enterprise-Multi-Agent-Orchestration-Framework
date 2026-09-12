"""Unit tests for OrchestrationRuntime graph compilation, state transitions, and checkpointing."""

import pytest

from enterprise_orchestrator.agents.planner import PlannerAgent
from enterprise_orchestrator.agents.retrieval import InMemoryDocumentStore, RetrievalAgent
from enterprise_orchestrator.agents.tool_execution import ToolExecutionAgent
from enterprise_orchestrator.agents.validator import ValidatorAgent
from enterprise_orchestrator.core.types import ExecutionStatus
from enterprise_orchestrator.memory.in_memory import InMemoryStateStore
from enterprise_orchestrator.orchestration.runtime import OrchestrationRuntime
from enterprise_orchestrator.providers.router import LLMRouter
from enterprise_orchestrator.tools.registry import ToolRegistry


def test_runtime_graph_compilation():
    """Verify LangGraph StateGraph builds and compiles with all required nodes."""
    runtime = OrchestrationRuntime()
    assert runtime.compiled_graph is not None

    # Verify graph node names exist
    node_keys = runtime._graph.nodes.keys()
    assert "supervisor" in node_keys
    assert "planner" in node_keys
    assert "retrieval" in node_keys
    assert "tool" in node_keys
    assert "validator" in node_keys
    assert "human_gate" in node_keys
    assert "synthesis" in node_keys
    assert "failed" in node_keys


@pytest.mark.asyncio
async def test_runtime_state_checkpoint_persistence():
    """Verify that execution checkpoints are created in the state store."""
    state_store = InMemoryStateStore()
    runtime = OrchestrationRuntime(state_store=state_store)

    # Simple run triggering planner failure (no mock LLM provided)
    final_state = await runtime.run(request="Empty test run")

    assert final_state.run_id is not None
    # Check that initial and final checkpoints were persisted
    checkpoints = await state_store.list_checkpoints(final_state.run_id)
    assert len(checkpoints) >= 2
    assert checkpoints[0].step == 0
    assert (await state_store.get_state(final_state.run_id)) is not None


@pytest.mark.asyncio
async def test_runtime_dependency_deadlock_detection():
    """Verify runtime detects and handles dependency issues without infinite loop."""
    from unittest.mock import MagicMock
    from enterprise_orchestrator.agents.planner import PlannedDecompositionSchema, PlannedTaskItem
    from enterprise_orchestrator.core.types import TaskType
    from enterprise_orchestrator.providers.models import LLMResponse

    router = MagicMock(spec=LLMRouter)
    # Plan where task_1 depends on non-existent task_99
    deadlock_plan = PlannedDecompositionSchema(
        rationale="Deadlocked plan",
        tasks=[
            PlannedTaskItem(
                id="task_1",
                title="Unexecutable Task",
                task_type=TaskType.RETRIEVAL,
                dependencies=["task_99"],  # unmet
            )
        ],
    )
    router.generate_structured.return_value = (
        deadlock_plan,
        LLMResponse(content="{}", model="gemini", provider="gemini"),
    )

    runtime = OrchestrationRuntime(router=router)
    final_state = await runtime.run(request="Deadlock test")

    assert final_state.metadata.status == ExecutionStatus.FAILED
    assert any("non-existent dependency" in str(err).lower() or "deadlock" in str(err).lower() for err in final_state.errors)


@pytest.mark.asyncio
async def test_runtime_human_decision_modification_reexecutes():
    """Verify human MODIFIED verdict updates input_data and re-executes task."""
    from typing import Type
    from pydantic import BaseModel, Field
    from unittest.mock import MagicMock
    from enterprise_orchestrator.agents.planner import PlannedDecompositionSchema, PlannedTaskItem
    from enterprise_orchestrator.agents.validator import CriticEvaluationSchema
    from enterprise_orchestrator.core.human import HumanDecision
    from enterprise_orchestrator.core.types import HumanRequestStatus, TaskType, ValidationStatus
    from enterprise_orchestrator.providers.models import LLMResponse
    from enterprise_orchestrator.tools.base import BaseTool, ToolMetadata
    from enterprise_orchestrator.tools.registry import ToolRegistry

    class EchoInput(BaseModel):
        msg: str = Field(...)

    class EchoOutput(BaseModel):
        echo: str = Field(...)

    class EchoTool(BaseTool):
        def __init__(self) -> None:
            super().__init__(metadata=ToolMetadata(name="echo_param", description="Echoes msg"))

        @property
        def input_schema(self) -> Type[BaseModel]:
            return EchoInput

        @property
        def output_schema(self) -> Type[BaseModel]:
            return EchoOutput

        async def execute(self, input_data: BaseModel) -> BaseModel:
            data = EchoInput.model_validate(input_data)
            return EchoOutput(echo=data.msg)

    tool_reg = ToolRegistry()
    tool_reg.register(EchoTool())

    plan = PlannedDecompositionSchema(
        rationale="Tool needing input fix",
        tasks=[
            PlannedTaskItem(
                id="task_1",
                title="Echo Task",
                task_type=TaskType.TOOL_EXECUTION,
                input_data={"tool_name": "echo_param", "parameters": {"msg": "bad_input"}},
            )
        ],
    )

    critique_escalate = CriticEvaluationSchema(
        is_valid=False,
        status=ValidationStatus.NEEDS_HUMAN_REVIEW,
        score=0.3,
        feedback="Incorrect parameter provided.",
        needs_human_review=True,
    )

    critique_valid = CriticEvaluationSchema(
        is_valid=True,
        status=ValidationStatus.VALID,
        score=1.0,
        feedback="Verified parameter.",
    )

    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    router = MagicMock(spec=LLMRouter)
    router.generate_structured.side_effect = [
        (plan, mock_resp),
        (critique_escalate, mock_resp),
        (critique_valid, mock_resp),
    ]

    runtime = OrchestrationRuntime(router=router, tool_registry=tool_reg)
    paused_state = await runtime.run(request="Execute with fixable param")

    assert paused_state.metadata.status == ExecutionStatus.PAUSED_FOR_HUMAN
    req = paused_state.human_requests[0]

    # Provide modified parameters
    decision = HumanDecision(
        request_id=req.request_id,
        status=HumanRequestStatus.MODIFIED,
        decision_note="Fixed parameter string.",
        modified_payload={"parameters": {"msg": "corrected_input"}},
    )

    resumed_state = await runtime.resume_with_human_decision(paused_state, decision)
    assert resumed_state.metadata.status == ExecutionStatus.COMPLETED
    task_1 = resumed_state.plan.get_task("task_1")
    assert task_1.status.value == "completed"
    assert task_1.output_data["result"] == {"echo": "corrected_input"}


@pytest.mark.asyncio
async def test_runtime_sqlite_state_store_integration(tmp_path):
    """Scenario B: OrchestrationRuntime executing with SQLiteStateStore backend."""
    from unittest.mock import MagicMock
    from enterprise_orchestrator.agents.planner import PlannedDecompositionSchema, PlannedTaskItem
    from enterprise_orchestrator.agents.validator import CriticEvaluationSchema
    from enterprise_orchestrator.core.types import TaskType, ValidationStatus
    from enterprise_orchestrator.memory.sqlite import SQLiteStateStore
    from enterprise_orchestrator.providers.models import LLMResponse

    db_path = str(tmp_path / "runtime_integration.db")
    sqlite_store = SQLiteStateStore(db_path=db_path)

    plan = PlannedDecompositionSchema(
        rationale="Simple single task plan",
        tasks=[
            PlannedTaskItem(
                id="task_1",
                title="Search Docs",
                task_type=TaskType.RETRIEVAL,
                input_data={"query": "Financial summary"},
            )
        ],
    )
    critique_valid = CriticEvaluationSchema(
        is_valid=True,
        status=ValidationStatus.VALID,
        score=1.0,
        feedback="Accurate result.",
    )

    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    router = MagicMock(spec=LLMRouter)
    router.generate_structured.side_effect = [
        (plan, mock_resp),
        (critique_valid, mock_resp),
    ]

    runtime = OrchestrationRuntime(
        router=router,
        state_store=sqlite_store,
    )

    final_state = await runtime.run(request="Execute SQLite persistence run")
    assert final_state.metadata.status == ExecutionStatus.COMPLETED

    # 1. Verify state reloaded directly from SQLite matches final state
    reloaded_state = await sqlite_store.get_state(final_state.run_id)
    assert reloaded_state is not None
    assert reloaded_state.run_id == final_state.run_id
    assert reloaded_state.metadata.status == ExecutionStatus.COMPLETED
    assert reloaded_state.plan.all_completed() is True

    # 2. Verify checkpoints were saved to SQLite
    checkpoints = await sqlite_store.list_checkpoints(final_state.run_id)
    assert len(checkpoints) >= 2
    assert checkpoints[0].step == 0
    assert checkpoints[-1].step == final_state.current_step
