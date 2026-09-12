"""Integration tests for FastAPI metrics endpoints."""

from typing import Any, Tuple, Type

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from enterprise_orchestrator.api.app import create_app
from enterprise_orchestrator.api.dependencies import (
    get_metrics_registry,
    get_orchestration_service,
    get_state_store,
    get_tracer,
)
from enterprise_orchestrator.memory.in_memory import InMemoryStateStore
from enterprise_orchestrator.observability.metrics.registry import MetricsRegistry
from enterprise_orchestrator.observability.tracing.in_memory import InMemoryTracer
from enterprise_orchestrator.orchestration.runtime import OrchestrationRuntime
from enterprise_orchestrator.providers.base import BaseLLMProvider
from enterprise_orchestrator.providers.models import LLMRequest, LLMResponse
from enterprise_orchestrator.providers.router import LLMRouter
from enterprise_orchestrator.services.orchestration_service import OrchestrationService
from enterprise_orchestrator.tools.base import BaseTool, ToolMetadata
from enterprise_orchestrator.tools.registry import ToolRegistry


class DummyInput(BaseModel):
    pass


class DummyOutput(BaseModel):
    echo: str = "ok"


class EchoTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(name="echo", description="echo tool"))

    @property
    def input_schema(self) -> Type[BaseModel]:
        return DummyInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return DummyOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        return DummyOutput(echo="ok")


class DummyFakeProvider(BaseLLMProvider):
    def __init__(self) -> None:
        super().__init__(provider_name="dummy_fake", default_model="model")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="OK", model="model", provider="dummy_fake")

    async def generate_structured(self, request: LLMRequest, schema: Any) -> Tuple[Any, LLMResponse]:
        schema_name = getattr(schema, "__name__", "")
        if "Plan" in schema_name or "ExecutionPlan" in schema_name:
            plan = {
                "rationale": "Dummy plan",
                "tasks": [
                    {
                        "id": "task-1",
                        "title": "Echo",
                        "task_type": "tool_execution",
                        "status": "pending",
                        "dependencies": [],
                        "input_data": {"tool_name": "echo", "parameters": {}},
                    }
                ],
            }
            return schema.model_validate(plan), LLMResponse(content="{}", model="model", provider="dummy_fake")
        elif "Critic" in schema_name or "Validation" in schema_name or "Validator" in schema_name:
            val = {
                "status": "valid",
                "is_valid": True,
                "score": 1.0,
                "feedback": "Valid",
                "issues": [],
                "needs_retry": False,
                "needs_human_review": False,
            }
            return schema.model_validate(val), LLMResponse(content="{}", model="model", provider="dummy_fake")
        return schema.model_validate({}), LLMResponse(content="{}", model="model", provider="dummy_fake")

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def test_client() -> TestClient:
    app = create_app()

    state_store = InMemoryStateStore()
    metrics = MetricsRegistry()
    tracer = InMemoryTracer()
    provider = DummyFakeProvider()
    router = LLMRouter(providers={provider.provider_name: provider}, default_provider=provider.provider_name)
    tool_reg = ToolRegistry()
    tool_reg.register(EchoTool())

    runtime = OrchestrationRuntime(
        router=router,
        tool_registry=tool_reg,
        state_store=state_store,
        metrics=metrics,
        tracer=tracer,
    )
    service = OrchestrationService(runtime=runtime, state_store=state_store)

    app.dependency_overrides[get_state_store] = lambda: state_store
    app.dependency_overrides[get_metrics_registry] = lambda: metrics
    app.dependency_overrides[get_tracer] = lambda: tracer
    app.dependency_overrides[get_orchestration_service] = lambda: service

    return TestClient(app)


def test_global_metrics_endpoint(test_client: TestClient) -> None:
    resp = test_client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()

    assert "counters" in data
    assert "gauges" in data
    assert "histograms" in data
    assert "runs_started_total" in data["counters"]
    assert "active_runs" in data["gauges"]
    assert "run_duration_seconds" in data["histograms"]


def test_run_metrics_endpoint(test_client: TestClient) -> None:
    # 1. Create a run first
    create_resp = test_client.post(
        "/api/v1/runs",
        json={"request": "Metrics test request"},
    )
    assert create_resp.status_code == 201
    run_id = create_resp.json()["run_id"]

    # 2. Query run telemetry
    metrics_resp = test_client.get(f"/api/v1/runs/{run_id}/metrics")
    assert metrics_resp.status_code == 200
    m_data = metrics_resp.json()

    assert m_data["run_id"] == run_id
    assert m_data["status"] == "completed"
    assert m_data["spans_count"] >= 1
    assert any(s["name"] == "OrchestrationRuntime.run" for s in m_data["spans"])
