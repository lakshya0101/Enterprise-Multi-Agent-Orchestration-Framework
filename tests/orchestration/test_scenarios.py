"""Integration scenario tests demonstrating full multi-agent orchestration loops."""

from typing import Type
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from enterprise_orchestrator.agents.planner import (
    PlannedDecompositionSchema,
    PlannedTaskItem,
)
from enterprise_orchestrator.agents.retrieval import (
    Document,
    InMemoryDocumentStore,
)
from enterprise_orchestrator.agents.validator import CriticEvaluationSchema
from enterprise_orchestrator.core.human import HumanDecision
from enterprise_orchestrator.core.types import (
    AgentRole,
    ExecutionStatus,
    HumanRequestStatus,
    TaskType,
    ValidationStatus,
)
from enterprise_orchestrator.memory.in_memory import InMemoryStateStore
from enterprise_orchestrator.orchestration.runtime import OrchestrationRuntime
from enterprise_orchestrator.providers.models import LLMResponse
from enterprise_orchestrator.providers.router import LLMRouter
from enterprise_orchestrator.tools.base import BaseTool, ToolMetadata
from enterprise_orchestrator.tools.registry import ToolRegistry


class FinancialCalcInput(BaseModel):
    revenue: float = Field(...)
    cost: float = Field(...)


class FinancialCalcOutput(BaseModel):
    profit: float = Field(...)
    margin_pct: float = Field(...)


class ProfitCalculatorTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(name="calc_profit", description="Computes profit and margin", timeout_seconds=5.0)
        )

    @property
    def input_schema(self) -> Type[BaseModel]:
        return FinancialCalcInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return FinancialCalcOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        data = FinancialCalcInput.model_validate(input_data)
        profit = data.revenue - data.cost
        margin = (profit / data.revenue) * 100.0 if data.revenue > 0 else 0.0
        return FinancialCalcOutput(profit=profit, margin_pct=round(margin, 2))


@pytest.fixture
def mock_planner_router():
    router = LLMRouter()
    router.generate_structured = AsyncMock()
    return router


@pytest.fixture
def test_environment(mock_planner_router):
    # 1. Document store with test documents
    doc_store = InMemoryDocumentStore()
    test_doc = Document(
        id="doc_financial_q3",
        content="Q3 Revenue reached 1000000 with operating cost of 600000.",
        metadata={"domain": "finance"},
    )
    doc_store._documents[test_doc.id] = test_doc

    # 2. Tool registry with profit calculator
    tool_reg = ToolRegistry()
    tool_reg.register(ProfitCalculatorTool())

    # 3. State store
    state_store = InMemoryStateStore()

    return {
        "doc_store": doc_store,
        "tool_reg": tool_reg,
        "state_store": state_store,
        "router": mock_planner_router,
    }


@pytest.mark.asyncio
async def test_scenario_1_e2e_happy_path(test_environment):
    """Scenario 1: Full DAG Pipeline (Planning -> Retrieval -> Tool -> Validation -> Complete)."""
    env = test_environment
    router = env["router"]

    # Mock Planner output: 2-step DAG
    plan = PlannedDecompositionSchema(
        rationale="Retrieve financial data then calculate profit margin",
        tasks=[
            PlannedTaskItem(
                id="task_1",
                title="Fetch Q3 Financials",
                task_type=TaskType.RETRIEVAL,
                assigned_agent=AgentRole.RETRIEVER,
                dependencies=[],
                input_data={"query": "Q3 Revenue operating cost", "top_k": 1},
            ),
            PlannedTaskItem(
                id="task_2",
                title="Calculate Profit & Margin",
                task_type=TaskType.TOOL_EXECUTION,
                assigned_agent=AgentRole.TOOL_EXECUTOR,
                dependencies=["task_1"],
                input_data={"tool_name": "calc_profit", "parameters": {"revenue": 1000000.0, "cost": 600000.0}},
            ),
        ],
    )

    # Mock Validator output: VALID
    critique_valid = CriticEvaluationSchema(
        is_valid=True,
        status=ValidationStatus.VALID,
        score=0.99,
        feedback="Verified calculation matches financial records.",
        issues=[],
        needs_retry=False,
        needs_human_review=False,
    )

    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    # First call: planner, Second call: validator task 1, Third call: validator task 2
    router.generate_structured.side_effect = [
        (plan, mock_resp),
        (critique_valid, mock_resp),
        (critique_valid, mock_resp),
    ]

    runtime = OrchestrationRuntime(
        router=router,
        document_store=env["doc_store"],
        tool_registry=env["tool_reg"],
        state_store=env["state_store"],
    )

    final_state = await runtime.run(request="Compute Q3 profit margin")

    assert final_state.metadata.status == ExecutionStatus.COMPLETED
    assert final_state.plan is not None
    assert final_state.plan.all_completed() is True
    assert "calc_profit" in final_state.tool_results
    assert final_state.tool_results["calc_profit"]["profit"] == 400000.0
    assert final_state.tool_results["calc_profit"]["margin_pct"] == 40.0
    assert len(final_state.validation_results) == 2
    assert final_state.final_response is not None


@pytest.mark.asyncio
async def test_scenario_2_validation_retry_loop(test_environment):
    """Scenario 2: Validation Failure -> Task Retry -> Valid -> Complete."""
    env = test_environment
    router = env["router"]

    plan = PlannedDecompositionSchema(
        rationale="Single step query",
        tasks=[
            PlannedTaskItem(
                id="task_1",
                title="Fetch Docs",
                task_type=TaskType.RETRIEVAL,
                input_data={"query": "Q3 Revenue"},
            ),
        ],
    )

    critique_retry = CriticEvaluationSchema(
        is_valid=False,
        status=ValidationStatus.NEEDS_RETRY,
        score=0.5,
        feedback="Format incomplete. Needs re-fetch.",
        issues=["Missing tags"],
        needs_retry=True,
    )

    critique_valid = CriticEvaluationSchema(
        is_valid=True,
        status=ValidationStatus.VALID,
        score=0.95,
        feedback="Now complete.",
    )

    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    # Call 1: Planner, Call 2: Validator (retry), Call 3: Validator (valid)
    router.generate_structured.side_effect = [
        (plan, mock_resp),
        (critique_retry, mock_resp),
        (critique_valid, mock_resp),
    ]

    runtime = OrchestrationRuntime(
        router=router,
        document_store=env["doc_store"],
        tool_registry=env["tool_reg"],
        state_store=env["state_store"],
    )

    final_state = await runtime.run(request="Fetch financial records")

    assert final_state.metadata.status == ExecutionStatus.COMPLETED
    task_1 = final_state.plan.get_task("task_1")
    assert task_1.retry_count == 1
    assert task_1.status.value == "completed"


@pytest.mark.asyncio
async def test_scenario_3_retry_budget_exhaustion(test_environment):
    """Scenario 3: Repeated Validation Failure exhausts retry budget -> Terminal Failure."""
    env = test_environment
    router = env["router"]

    plan = PlannedDecompositionSchema(
        rationale="Task with strict max retries",
        tasks=[
            PlannedTaskItem(
                id="task_1",
                title="Fetch Docs",
                task_type=TaskType.RETRIEVAL,
                input_data={"query": "Q3 Revenue"},
            ),
        ],
    )

    critique_always_retry = CriticEvaluationSchema(
        is_valid=False,
        status=ValidationStatus.NEEDS_RETRY,
        score=0.4,
        feedback="Persistently defective output.",
        needs_retry=True,
    )

    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    # Return planner, then repeated retry verdicts
    router.generate_structured.side_effect = [
        (plan, mock_resp),
        (critique_always_retry, mock_resp),
        (critique_always_retry, mock_resp),
        (critique_always_retry, mock_resp),
        (critique_always_retry, mock_resp),
    ]

    runtime = OrchestrationRuntime(
        router=router,
        document_store=env["doc_store"],
        tool_registry=env["tool_reg"],
        state_store=env["state_store"],
    )

    final_state = await runtime.run(request="Attempt stubborn task")

    assert final_state.metadata.status == ExecutionStatus.FAILED
    task_1 = final_state.plan.get_task("task_1")
    assert task_1.status.value == "failed"
    assert "Retry budget exhausted" in task_1.error


@pytest.mark.asyncio
async def test_scenario_4_human_in_the_loop_approval(test_environment):
    """Scenario 4: High risk output -> Human Escalation -> Programmatic Approval -> Complete."""
    env = test_environment
    router = env["router"]

    plan = PlannedDecompositionSchema(
        rationale="Tool execution needing review",
        tasks=[
            PlannedTaskItem(
                id="task_1",
                title="Execute Calculation",
                task_type=TaskType.TOOL_EXECUTION,
                input_data={"tool_name": "calc_profit", "parameters": {"revenue": 500000.0, "cost": 100000.0}},
            ),
        ],
    )

    critique_human = CriticEvaluationSchema(
        is_valid=False,
        status=ValidationStatus.NEEDS_HUMAN_REVIEW,
        score=0.4,
        feedback="High-value corporate calculation requires CFO authorization.",
        needs_human_review=True,
    )

    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    router.generate_structured.side_effect = [
        (plan, mock_resp),
        (critique_human, mock_resp),
    ]

    runtime = OrchestrationRuntime(
        router=router,
        document_store=env["doc_store"],
        tool_registry=env["tool_reg"],
        state_store=env["state_store"],
    )

    # 1. Initial run pauses at human gate
    paused_state = await runtime.run(request="High value transaction")
    assert paused_state.metadata.status == ExecutionStatus.PAUSED_FOR_HUMAN
    assert paused_state.requires_human is True
    assert len(paused_state.human_requests) == 1
    req = paused_state.human_requests[0]

    # 2. Programmatically provide HumanDecision (Approved)
    decision = HumanDecision(
        request_id=req.request_id,
        status=HumanRequestStatus.APPROVED,
        decision_note="Authorized by CFO.",
        reviewer_id="cfo_user",
    )

    resumed_state = await runtime.resume_with_human_decision(paused_state, decision)
    assert resumed_state.metadata.status == ExecutionStatus.COMPLETED
    assert resumed_state.requires_human is False
    assert resumed_state.plan.all_completed() is True


@pytest.mark.asyncio
async def test_scenario_5_human_in_the_loop_rejection(test_environment):
    """Scenario 5: Human Rejection terminates workflow safely."""
    env = test_environment
    router = env["router"]

    plan = PlannedDecompositionSchema(
        rationale="Tool execution",
        tasks=[
            PlannedTaskItem(
                id="task_1",
                title="Execute Tool",
                task_type=TaskType.TOOL_EXECUTION,
                input_data={"tool_name": "calc_profit", "parameters": {"revenue": 500.0, "cost": 100.0}},
            ),
        ],
    )

    critique_human = CriticEvaluationSchema(
        is_valid=False,
        status=ValidationStatus.NEEDS_HUMAN_REVIEW,
        score=0.2,
        feedback="Unverified numbers require review.",
        needs_human_review=True,
    )

    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    router.generate_structured.side_effect = [
        (plan, mock_resp),
        (critique_human, mock_resp),
    ]

    runtime = OrchestrationRuntime(
        router=router,
        document_store=env["doc_store"],
        tool_registry=env["tool_reg"],
        state_store=env["state_store"],
    )

    paused_state = await runtime.run(request="Unverified run")
    req = paused_state.human_requests[0]

    decision = HumanDecision(
        request_id=req.request_id,
        status=HumanRequestStatus.REJECTED,
        decision_note="Numbers do not match balance sheet.",
        reviewer_id="auditor_01",
    )

    resumed_state = await runtime.resume_with_human_decision(paused_state, decision)
    assert resumed_state.metadata.status == ExecutionStatus.FAILED
    assert "Workflow terminated by human reviewer" in resumed_state.errors[0]["fatal_error"]
