"""Unit and integration tests for /api/v1/runs/{id}/human-review endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from enterprise_orchestrator.agents.planner import PlannedDecompositionSchema, PlannedTaskItem
from enterprise_orchestrator.agents.validator import CriticEvaluationSchema
from enterprise_orchestrator.api.app import create_app
from enterprise_orchestrator.api.dependencies import get_orchestration_service
from enterprise_orchestrator.core.types import TaskType, ValidationStatus
from enterprise_orchestrator.memory.in_memory import InMemoryStateStore
from enterprise_orchestrator.orchestration.runtime import OrchestrationRuntime
from enterprise_orchestrator.providers.models import LLMResponse
from enterprise_orchestrator.providers.router import LLMRouter
from enterprise_orchestrator.services.orchestration_service import OrchestrationService
from enterprise_orchestrator.tools.base import BaseTool, ToolMetadata
from enterprise_orchestrator.tools.registry import ToolRegistry


class DummyCalcInput(BaseModel):
    amount: float = Field(...)


class DummyCalcOutput(BaseModel):
    total: float = Field(...)


class DummyCalcTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(name="calc_wire", description="Calculates wire transfer"))

    @property
    def input_schema(self):
        return DummyCalcInput

    @property
    def output_schema(self):
        return DummyCalcOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        data = DummyCalcInput.model_validate(input_data)
        return DummyCalcOutput(total=data.amount * 1.05)


@pytest.fixture
def hitl_service():
    """Build a service configured to trigger human review."""
    router = MagicMock(spec=LLMRouter)
    tool_reg = ToolRegistry()
    tool_reg.register(DummyCalcTool())

    plan = PlannedDecompositionSchema(
        rationale="High value transfer plan",
        tasks=[
            PlannedTaskItem(
                id="task_wire_1",
                title="Execute wire transfer",
                task_type=TaskType.TOOL_EXECUTION,
                input_data={"tool_name": "calc_wire", "parameters": {"amount": 50000.0}},
            )
        ],
    )
    critique_human = CriticEvaluationSchema(
        is_valid=False,
        status=ValidationStatus.NEEDS_HUMAN_REVIEW,
        score=0.4,
        feedback="Large wire transfer requires manager authorization.",
        needs_human_review=True,
    )

    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    router.generate_structured.side_effect = [
        (plan, mock_resp),
        (critique_human, mock_resp),
    ]

    state_store = InMemoryStateStore()
    runtime = OrchestrationRuntime(
        router=router,
        tool_registry=tool_reg,
        state_store=state_store,
    )
    return OrchestrationService(runtime=runtime, state_store=state_store)


def test_hitl_approval_lifecycle(hitl_service):
    """Verify human escalation review retrieval and approved decision submission."""
    app = create_app()
    app.dependency_overrides[get_orchestration_service] = lambda: hitl_service
    client = TestClient(app)

    # 1. Start run that triggers human review
    res_create = client.post("/api/v1/runs", json={"request": "Transfer $50,000"})
    assert res_create.status_code == 201
    run_data = res_create.json()
    assert run_data["status"] == "paused_for_human"
    assert run_data["requires_human"] is True
    run_id = run_data["run_id"]

    # 2. Query pending human review
    res_review = client.get(f"/api/v1/runs/{run_id}/human-review")
    assert res_review.status_code == 200
    review_data = res_review.json()
    assert "request_id" in review_data
    assert "manager authorization" in review_data["reason"].lower()
    request_id = review_data["request_id"]

    # 3. Submit APPROVED decision
    res_decision = client.post(
        f"/api/v1/runs/{run_id}/human-review",
        json={
            "request_id": request_id,
            "status": "APPROVED",
            "decision_note": "Authorized by Manager",
            "reviewer_id": "mgr_123",
        },
    )
    assert res_decision.status_code == 200
    decision_data = res_decision.json()
    assert decision_data["status"] == "completed"
    assert decision_data["resumed_successfully"] is True

    # 4. Confirm subsequent review query returns 404
    res_review_after = client.get(f"/api/v1/runs/{run_id}/human-review")
    assert res_review_after.status_code == 404


def test_hitl_rejection_lifecycle(hitl_service):
    """Verify human rejection terminates run in failed status."""
    app = create_app()
    app.dependency_overrides[get_orchestration_service] = lambda: hitl_service
    client = TestClient(app)

    res_create = client.post("/api/v1/runs", json={"request": "Transfer $50,000"})
    run_id = res_create.json()["run_id"]

    res_review = client.get(f"/api/v1/runs/{run_id}/human-review")
    request_id = res_review.json()["request_id"]

    # Submit REJECTED decision
    res_decision = client.post(
        f"/api/v1/runs/{run_id}/human-review",
        json={
            "request_id": request_id,
            "status": "REJECTED",
            "decision_note": "Suspicious request rejected",
        },
    )
    assert res_decision.status_code == 200
    assert res_decision.json()["status"] == "failed"


def test_hitl_invalid_decision_status(hitl_service):
    """Verify submitting invalid status returns 422."""
    app = create_app()
    app.dependency_overrides[get_orchestration_service] = lambda: hitl_service
    client = TestClient(app)

    res_create = client.post("/api/v1/runs", json={"request": "Transfer $50,000"})
    run_id = res_create.json()["run_id"]
    request_id = client.get(f"/api/v1/runs/{run_id}/human-review").json()["request_id"]

    res_decision = client.post(
        f"/api/v1/runs/{run_id}/human-review",
        json={"request_id": request_id, "status": "MAYBE"},
    )
    assert res_decision.status_code == 422
