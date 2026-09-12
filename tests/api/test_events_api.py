"""Unit and integration tests for /api/v1/runs/{id}/events SSE streaming endpoint."""

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
def events_service():
    """Build service with completed run."""
    router = MagicMock(spec=LLMRouter)
    plan = PlannedDecompositionSchema(
        rationale="SSE plan",
        tasks=[
            PlannedTaskItem(
                id="task_1",
                title="Search knowledge",
                task_type=TaskType.RETRIEVAL,
                input_data={"query": "SSE test"},
            )
        ],
    )
    critique = CriticEvaluationSchema(
        is_valid=True,
        status=ValidationStatus.VALID,
        score=1.0,
        feedback="Verified.",
    )
    mock_resp = LLMResponse(content="{}", model="gemini", provider="gemini")
    router.generate_structured.side_effect = [
        (plan, mock_resp),
        (critique, mock_resp),
    ]

    state_store = InMemoryStateStore()
    runtime = OrchestrationRuntime(router=router, state_store=state_store)
    return OrchestrationService(runtime=runtime, state_store=state_store)


def test_sse_event_streaming_content_type_and_framing(events_service):
    """Verify SSE endpoint returns text/event-stream with standard event/data formatting."""
    app = create_app()
    app.dependency_overrides[get_orchestration_service] = lambda: events_service
    client = TestClient(app)

    # 1. Create a run first
    res_create = client.post("/api/v1/runs", json={"request": "SSE stream request"})
    run_id = res_create.json()["run_id"]

    # 2. Connect to SSE event stream
    res_events = client.get(f"/api/v1/runs/{run_id}/events")
    assert res_events.status_code == 200
    assert "text/event-stream" in res_events.headers["content-type"]

    content = res_events.text
    assert "event: run_status" in content
    assert "data: {" in content
    assert "event: stream_closed" in content


def test_sse_events_unknown_run_404(events_service):
    """Verify SSE request for unknown run returns 404."""
    app = create_app()
    app.dependency_overrides[get_orchestration_service] = lambda: events_service
    client = TestClient(app)

    res = client.get("/api/v1/runs/unknown_run_123/events")
    assert res.status_code == 404
