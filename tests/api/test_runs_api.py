"""Unit and integration tests for /api/v1/runs endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

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


@pytest.fixture
def mock_service():
    """Build an in-memory service wired to deterministic mock LLMs."""
    router = MagicMock(spec=LLMRouter)
    plan = PlannedDecompositionSchema(
        rationale="API test execution plan",
        tasks=[
            PlannedTaskItem(
                id="task_1",
                title="Search knowledge base",
                task_type=TaskType.RETRIEVAL,
                input_data={"query": "API test query"},
            )
        ],
    )
    critique = CriticEvaluationSchema(
        is_valid=True,
        status=ValidationStatus.VALID,
        score=1.0,
        feedback="Verified output.",
    )

    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    router.generate_structured.side_effect = [
        (plan, mock_resp),
        (critique, mock_resp),
    ]

    state_store = InMemoryStateStore()
    runtime = OrchestrationRuntime(router=router, state_store=state_store)
    return OrchestrationService(runtime=runtime, state_store=state_store)


def test_create_and_get_run(mock_service):
    """Verify POST /api/v1/runs creates a run and GET /api/v1/runs/{id} retrieves it."""
    app = create_app()
    app.dependency_overrides[get_orchestration_service] = lambda: mock_service
    client = TestClient(app)

    # 1. Create Run
    res = client.post(
        "/api/v1/runs",
        json={"request": "Perform audit", "correlation_id": "corr-100"},
    )
    assert res.status_code == 201
    data = res.json()
    assert "run_id" in data
    assert data["request"] == "Perform audit"
    assert data["status"] == "completed"
    assert data["plan"] is not None
    assert len(data["plan"]["tasks"]) == 1

    run_id = data["run_id"]

    # 2. Get Run
    res_get = client.get(f"/api/v1/runs/{run_id}")
    assert res_get.status_code == 200
    get_data = res_get.json()
    assert get_data["run_id"] == run_id
    assert get_data["status"] == "completed"

    # 3. Get Status
    res_status = client.get(f"/api/v1/runs/{run_id}/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["run_id"] == run_id
    assert status_data["status"] == "completed"
    assert status_data["completed_tasks"] == 1

    # 4. Get Checkpoints
    res_cp = client.get(f"/api/v1/runs/{run_id}/checkpoints")
    assert res_cp.status_code == 200
    cp_data = res_cp.json()
    assert cp_data["total_checkpoints"] >= 2


def test_unknown_run_404(mock_service):
    """Verify non-existent run ID returns 404 with structured ErrorResponse."""
    app = create_app()
    app.dependency_overrides[get_orchestration_service] = lambda: mock_service
    client = TestClient(app)

    res = client.get("/api/v1/runs/non_existent_id")
    assert res.status_code == 404
    data = res.json()
    assert "error" in data
    assert "not found" in data["error"].lower()


def test_create_run_validation_error():
    """Verify empty request returns 422 Unprocessable Entity."""
    app = create_app()
    client = TestClient(app)

    res = client.post("/api/v1/runs", json={"request": ""})
    assert res.status_code == 422
    data = res.json()
    assert data["code"] == "VALIDATION_ERROR"
