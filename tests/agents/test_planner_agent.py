"""Unit tests for PlannerAgent DAG decomposition and validation."""

from unittest.mock import AsyncMock

import pytest

from enterprise_orchestrator.agents.planner import (
    PlannedDecompositionSchema,
    PlannedTaskItem,
    PlannerAgent,
)
from enterprise_orchestrator.agents.result import AgentStatus
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.types import AgentRole, TaskType
from enterprise_orchestrator.errors.exceptions import ProviderError, ValidationError
from enterprise_orchestrator.providers.models import LLMResponse
from enterprise_orchestrator.providers.router import LLMRouter


@pytest.fixture
def mock_router():
    router = LLMRouter()
    router.generate_structured = AsyncMock()
    return router


@pytest.mark.asyncio
async def test_planner_valid_dag_generation(mock_router):
    """Verify happy path generating a valid 3-step DAG plan."""
    sample_plan = PlannedDecompositionSchema(
        rationale="Sequential data processing pipeline",
        tasks=[
            PlannedTaskItem(
                id="task_1",
                title="Fetch quarterly sales",
                description="Retrieve raw sales data",
                task_type=TaskType.RETRIEVAL,
                assigned_agent=AgentRole.RETRIEVER,
                dependencies=[],
            ),
            PlannedTaskItem(
                id="task_2",
                title="Calculate revenue growth",
                description="Compute percentage growth from sales data",
                task_type=TaskType.TOOL_EXECUTION,
                assigned_agent=AgentRole.TOOL_EXECUTOR,
                dependencies=["task_1"],
            ),
            PlannedTaskItem(
                id="task_3",
                title="Synthesize financial report",
                description="Draft report based on calculations",
                task_type=TaskType.SYNTHESIS,
                assigned_agent=AgentRole.CUSTOM,
                dependencies=["task_2"],
            ),
        ],
    )
    mock_resp = LLMResponse(content="{}", model="gemini-2.5-flash", provider="gemini")
    mock_router.generate_structured.return_value = (sample_plan, mock_resp)

    planner = PlannerAgent(router=mock_router)
    state = OrchestrationState.create_initial(request="Generate Q3 financial report")

    result = await planner.execute(state)

    assert result.status == AgentStatus.SUCCESS
    assert result.agent_name == "planner_agent"
    plan_dict = result.output["execution_plan"]
    assert len(plan_dict["tasks"]) == 3
    assert plan_dict["tasks"][1]["dependencies"] == ["task_1"]
    assert plan_dict["tasks"][2]["dependencies"] == ["task_2"]
    assert plan_dict["rationale"] == "Sequential data processing pipeline"


@pytest.mark.asyncio
async def test_planner_rejects_duplicate_task_ids(mock_router):
    """Verify rejection when generated plan contains duplicate task IDs."""
    invalid_plan = PlannedDecompositionSchema(
        rationale="Invalid duplicates",
        tasks=[
            PlannedTaskItem(id="task_1", title="Step A"),
            PlannedTaskItem(id="task_1", title="Step B"),
        ],
    )
    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    mock_router.generate_structured.return_value = (invalid_plan, mock_resp)

    planner = PlannerAgent(router=mock_router)
    state = OrchestrationState.create_initial(request="Run pipeline")

    result = await planner.execute(state)
    assert result.status == AgentStatus.RETRYABLE_FAILURE
    assert "duplicate task IDs" in result.error
    assert result.retryable is True


@pytest.mark.asyncio
async def test_planner_rejects_self_dependency(mock_router):
    """Verify rejection when task depends on itself."""
    invalid_plan = PlannedDecompositionSchema(
        rationale="Self dep",
        tasks=[
            PlannedTaskItem(id="task_1", title="Step A", dependencies=["task_1"]),
        ],
    )
    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    mock_router.generate_structured.return_value = (invalid_plan, mock_resp)

    planner = PlannerAgent(router=mock_router)
    state = OrchestrationState.create_initial(request="Run task")

    result = await planner.execute(state)
    assert result.status == AgentStatus.RETRYABLE_FAILURE
    assert "self-dependency" in result.error


@pytest.mark.asyncio
async def test_planner_rejects_missing_dependency(mock_router):
    """Verify rejection when task references a non-existent dependency ID."""
    invalid_plan = PlannedDecompositionSchema(
        rationale="Missing dep",
        tasks=[
            PlannedTaskItem(id="task_1", title="Step A", dependencies=["non_existent_task"]),
        ],
    )
    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    mock_router.generate_structured.return_value = (invalid_plan, mock_resp)

    planner = PlannerAgent(router=mock_router)
    state = OrchestrationState.create_initial(request="Run task")

    result = await planner.execute(state)
    assert result.status == AgentStatus.RETRYABLE_FAILURE
    assert "non-existent dependency" in result.error


@pytest.mark.asyncio
async def test_planner_rejects_cyclic_dependencies(mock_router):
    """Verify rejection of cyclic dependencies (A -> B -> A)."""
    cyclic_plan = PlannedDecompositionSchema(
        rationale="Cycle test",
        tasks=[
            PlannedTaskItem(id="task_a", title="Step A", dependencies=["task_b"]),
            PlannedTaskItem(id="task_b", title="Step B", dependencies=["task_a"]),
        ],
    )
    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    mock_router.generate_structured.return_value = (cyclic_plan, mock_resp)

    planner = PlannerAgent(router=mock_router)
    state = OrchestrationState.create_initial(request="Resolve cycle")

    result = await planner.execute(state)
    assert result.status == AgentStatus.RETRYABLE_FAILURE
    assert "cyclic dependencies" in result.error


@pytest.mark.asyncio
async def test_planner_empty_request_validation(mock_router):
    """Verify immediate failure if state request is empty."""
    planner = PlannerAgent(router=mock_router)
    state = OrchestrationState(request=" ")

    result = await planner.execute(state)
    assert result.status == AgentStatus.FAILURE
    assert "empty" in result.error
    assert mock_router.generate_structured.call_count == 0


@pytest.mark.asyncio
async def test_planner_llm_failure_propagation(mock_router):
    """Verify provider error propagation from LLMRouter."""
    mock_router.generate_structured.side_effect = ProviderError(
        message="Gemini and Groq both failed",
        provider_name="gemini",
        retryable=True,
    )
    planner = PlannerAgent(router=mock_router)
    state = OrchestrationState.create_initial(request="Plan task")

    result = await planner.execute(state)
    assert result.status == AgentStatus.RETRYABLE_FAILURE
    assert "Provider error during planning" in result.error
    assert result.retryable is True
