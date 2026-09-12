"""Tests for request payload size boundaries, correlation ID formats, and timeout cancellations."""

import asyncio
from unittest.mock import AsyncMock
import pytest
from starlette.testclient import TestClient

from enterprise_orchestrator.api.app import create_app
from enterprise_orchestrator.api.dependencies import get_orchestration_service
from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.services.orchestration_service import OrchestrationService
from enterprise_orchestrator.execution.models import ExecutionJob


class TestRequestLimitsAndBoundaries:
    """Test size limits, correlation ID validations, and timeout cancellations."""

    def test_oversized_payload_rejected_with_413(self):
        settings = FrameworkSettings(
            environment="development",
            api_auth_enabled=False,
            max_request_body_bytes=100,  # 100 bytes limit
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        large_payload = {"request": "A" * 500}
        resp = client.post("/api/v1/runs", json=large_payload)
        assert resp.status_code == 413
        assert "exceeds the maximum allowed limit" in resp.json()["error"]

    def test_invalid_correlation_id_rejected_with_422(self):
        settings = FrameworkSettings(
            environment="development",
            api_auth_enabled=False,
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        invalid_id = "bad ID with spaces!@#"
        resp = client.get("/health", headers={"X-Correlation-ID": invalid_id})
        assert resp.status_code == 422
        assert "Invalid correlation ID format" in resp.json()["error"]

    def test_valid_correlation_id_accepted_and_reflected(self):
        settings = FrameworkSettings(
            environment="development",
            api_auth_enabled=False,
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        valid_id = "corr-123.abc:test_01"
        resp = client.get("/health", headers={"X-Correlation-ID": valid_id})
        assert resp.status_code == 200
        assert resp.headers["X-Correlation-ID"] == valid_id
        assert resp.headers["X-Request-ID"] == valid_id

    def test_request_timeout_cancellation_returns_504(self):
        settings = FrameworkSettings(
            environment="development",
            api_auth_enabled=False,
            api_request_timeout_seconds=0.05,  # 50ms deadline
        )
        app = create_app(settings=settings)
        mock_service = AsyncMock(spec=OrchestrationService)

        mock_job = ExecutionJob(job_id="slow-j", run_id="slow-run-1", request="Perform slow workflow")

        async def slow_submit_run(*args, **kwargs):
            fut = asyncio.get_running_loop().create_future()
            async def _delayed_set():
                await asyncio.sleep(0.5)  # Sleep longer than 50ms
                if not fut.done():
                    fut.set_result(OrchestrationState.create_initial(request="Slow"))
            asyncio.create_task(_delayed_set())
            return mock_job, fut

        mock_service.submit_run.side_effect = slow_submit_run
        app.dependency_overrides[get_orchestration_service] = lambda: mock_service
        client = TestClient(app)

        resp = client.post("/api/v1/runs", json={"request": "Perform slow workflow"})
        assert resp.status_code == 504
        assert "exceeded request deadline" in resp.json()["error"]
