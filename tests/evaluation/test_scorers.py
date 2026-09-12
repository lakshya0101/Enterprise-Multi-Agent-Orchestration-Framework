"""Unit tests for Evaluation scorers."""

import pytest

from enterprise_orchestrator.core.human import HumanDecision, HumanEscalationRequest
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import ExecutionPlan, Task
from enterprise_orchestrator.core.types import ExecutionStatus, HumanRequestStatus, TaskStatus, TaskType
from enterprise_orchestrator.evaluation.models import BenchmarkScenario
from enterprise_orchestrator.evaluation.scorers import (
    HITLComplianceScorer,
    PlanFeasibilityScorer,
    RetrievalScorer,
    ValidatorFidelityScorer,
)


@pytest.mark.asyncio
async def test_plan_feasibility_scorer_valid_dag() -> None:
    scorer = PlanFeasibilityScorer()
    state = OrchestrationState.create_initial(request="Test request")
    state.plan = ExecutionPlan(
        rationale="2-step plan",
        tasks=[
            Task(id="t1", title="First", task_type=TaskType.RETRIEVAL, status=TaskStatus.COMPLETED, dependencies=[]),
            Task(id="t2", title="Second", task_type=TaskType.TOOL_EXECUTION, status=TaskStatus.COMPLETED, dependencies=["t1"]),
        ],
    )
    state.metadata.status = ExecutionStatus.COMPLETED

    res = await scorer.score(state)
    assert res.passed is True
    assert res.score == 1.0
    assert res.details["is_acyclic"] is True


@pytest.mark.asyncio
async def test_plan_feasibility_scorer_cyclic_dag() -> None:
    scorer = PlanFeasibilityScorer()
    state = OrchestrationState.create_initial(request="Test request")
    state.plan = ExecutionPlan(
        rationale="Cyclic plan",
        tasks=[
            Task(id="t1", title="First", task_type=TaskType.RETRIEVAL, status=TaskStatus.PENDING, dependencies=["t2"]),
            Task(id="t2", title="Second", task_type=TaskType.TOOL_EXECUTION, status=TaskStatus.PENDING, dependencies=["t1"]),
        ],
    )

    res = await scorer.score(state)
    assert res.passed is False
    assert res.details["is_acyclic"] is False


@pytest.mark.asyncio
async def test_retrieval_scorer_hit_rate() -> None:
    scorer = RetrievalScorer()
    state = OrchestrationState.create_initial(request="Test query")
    state.plan = ExecutionPlan(
        rationale="Retrieval plan",
        tasks=[
            Task(
                id="t1",
                title="Search",
                task_type=TaskType.RETRIEVAL,
                status=TaskStatus.COMPLETED,
                output_data={"documents": [{"id": "doc_101", "content": "relevant text"}]},
            )
        ],
    )

    scenario = BenchmarkScenario(
        benchmark_id="bench_retrieval",
        name="Retrieval test",
        description="test",
        request="test",
        expected_ground_truth_doc_ids=["doc_101"],
    )

    res = await scorer.score(state, scenario)
    assert res.passed is True
    assert res.score == 1.0
    assert res.details["hit_at_1_rate"] == 1.0


@pytest.mark.asyncio
async def test_hitl_compliance_scorer() -> None:
    scorer = HITLComplianceScorer()
    state = OrchestrationState.create_initial(request="High risk action")
    state.request_human_escalation("Review required", "Approve", {}, "t1")
    state.apply_human_decision(
        HumanDecision(
            request_id=state.human_requests[0].request_id,
            status=HumanRequestStatus.APPROVED,
            decision_note="Approved by auditor",
            reviewer_id="reviewer_1",
        )
    )
    state.metadata.status = ExecutionStatus.COMPLETED

    scenario = BenchmarkScenario(
        benchmark_id="bench_hitl",
        name="HITL test",
        description="test",
        request="test",
        expected_hitl_decision="approve",
    )

    res = await scorer.score(state, scenario)
    assert res.passed is True
    assert res.score == 1.0
