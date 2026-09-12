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


class TestRequestLimitsAndBoundaries:
    """Test size limits, correlation ID validations, and timeout cancellations."""

    def test_oversized_payload_rejected_with_413(self):
        settings = FrameworkSettings(
            environment="development",
            api_auth_enabled=False,
            max_request_body_bytes=200,  # 200 bytes limit
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        # Send 1 KB string
        big_body = {"request": "A" * 1000}
        resp = client.post("/api/v1/runs", json=big_body)
        assert resp.status_code == 413
        assert resp.json()["code"] == "PAYLOAD_TOO_LARGE"

    def test_invalid_correlation_id_rejected_with_422(self):
        settings = FrameworkSettings(
            environment="development",
            api_auth_enabled=False,
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        # Send header with CRLF / space injection
        resp = client.get("/health", headers={"X-Correlation-ID": "invalid corr\r\nid"})
        assert resp.status_code == 422
        assert resp.json()["code"] == "INVALID_CORRELATION_ID"

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

        async def slow_create_and_run(*args, **kwargs):
            await asyncio.sleep(0.5)  # Sleep longer than 50ms
            return OrchestrationState.create_initial(request="Slow")

        mock_service.create_and_run.side_effect = slow_create_and_run
        app.dependency_overrides[get_orchestration_service] = lambda: mock_service
        client = TestClient(app)

        resp = client.post("/api/v1/runs", json={"request": "Perform slow workflow"})
        assert resp.status_code == 504
        assert "exceeded request deadline" in resp.json()["error"]
