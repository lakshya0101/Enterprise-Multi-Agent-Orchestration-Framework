"""API integration tests for Track D Scalable Execution Boundaries and Async Modes."""

import asyncio
import hashlib
from unittest.mock import AsyncMock
import pytest
from starlette.testclient import TestClient

from enterprise_orchestrator.api.app import create_app
from enterprise_orchestrator.api.dependencies import get_orchestration_service
from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.types import ExecutionStatus
from enterprise_orchestrator.errors.exceptions import OrchestrationError
from enterprise_orchestrator.execution.idempotency import IdempotencyConflictError
from enterprise_orchestrator.execution.models import ExecutionJob, JobStatus
from enterprise_orchestrator.services.orchestration_service import OrchestrationService


@pytest.fixture
def api_client():
    admin_key = "admin-secret-key"
    admin_hash = hashlib.sha256(admin_key.encode()).hexdigest()

    settings = FrameworkSettings(
        environment="development",
        api_auth_enabled=True,
        api_key_hashes={admin_hash: {"subject_id": "admin_principal", "roles": ["ADMIN"]}},
        rate_limit_enabled=False,
        api_request_timeout_seconds=2.0,
    )

    app = create_app(settings=settings)
    mock_service = AsyncMock(spec=OrchestrationService)

    app.dependency_overrides[get_orchestration_service] = lambda: mock_service
    app.state.mock_service = mock_service
    app.state.admin_key = admin_key

    client = TestClient(app)
    client.headers.update({"X-API-Key": admin_key})
    return client, mock_service


class TestExecutionAPI:
    """Test suite for execution API endpoints, sync/async modes, and cancellation."""

    def test_create_run_synchronous_default(self, api_client):
        client, mock_service = api_client
        sample_state = OrchestrationState.create_initial(request="Synchronous request")
        sample_state.metadata.status = ExecutionStatus.COMPLETED
        sample_state.final_response = "Execution completed successfully."

        mock_job = ExecutionJob(job_id="j-1", run_id=sample_state.run_id, request="Synchronous request")

        async def _submit(*args, **kwargs):
            fut = asyncio.get_running_loop().create_future()
            fut.set_result(sample_state)
            return mock_job, fut

        mock_service.submit_run.side_effect = _submit

        res = client.post("/api/v1/runs", json={"request": "Synchronous request"})
        assert res.status_code == 201
        data = res.json()
        assert data["run_id"] == sample_state.run_id
        assert data["status"] == "completed"
        assert data["final_response"] == "Execution completed successfully."

    def test_create_run_asynchronous_mode(self, api_client):
        client, mock_service = api_client
        sample_state = OrchestrationState.create_initial(request="Asynchronous request")
        sample_state.metadata.status = ExecutionStatus.IDLE

        mock_job = ExecutionJob(job_id="j-2", run_id=sample_state.run_id, request="Asynchronous request")

        async def _submit(*args, **kwargs):
            fut = asyncio.get_running_loop().create_future()
            return mock_job, fut

        mock_service.submit_run.side_effect = _submit
        mock_service.get_state.return_value = sample_state

        res = client.post("/api/v1/runs?async=true", json={"request": "Asynchronous request"})
        assert res.status_code == 202
        data = res.json()
        assert data["run_id"] == sample_state.run_id
        assert data["status"] in ("idle", "queued")

    def test_create_run_synchronous_timeout_triggers_cancellation(self, api_client):
        client, mock_service = api_client
        mock_job = ExecutionJob(job_id="j-timeout", run_id="run-timeout-1", request="Slow request")

        async def _submit(*args, **kwargs):
            fut = asyncio.get_running_loop().create_future()
            return mock_job, fut

        mock_service.submit_run.side_effect = _submit

        # Timeout configured to 2.0s
        res = client.post("/api/v1/runs", json={"request": "Slow request"})
        assert res.status_code == 504
        assert "exceeded request deadline" in res.json()["error"]
        mock_service.cancel_run.assert_awaited_with("run-timeout-1", reason="Request deadline exceeded")

    def test_cancel_run_endpoint(self, api_client):
        client, mock_service = api_client
        run_id = "run-to-cancel"
        sample_state = OrchestrationState.create_initial(request="Task to cancel")
        sample_state.run_id = run_id
        sample_state.metadata.status = ExecutionStatus.RUNNING

        cancelled_state = OrchestrationState.create_initial(request="Task to cancel")
        cancelled_state.run_id = run_id
        cancelled_state.metadata.status = ExecutionStatus.CANCELLED

        mock_service.get_state.side_effect = [sample_state, cancelled_state]
        mock_service.cancel_run.return_value = True

        res = client.post(f"/api/v1/runs/{run_id}/cancel")
        assert res.status_code == 200
        data = res.json()
        assert data["run_id"] == run_id
        assert data["status"] == "cancelled"
        mock_service.cancel_run.assert_awaited()

    def test_idempotency_conflict_returns_409(self, api_client):
        client, mock_service = api_client
        mock_service.submit_run.side_effect = IdempotencyConflictError()

        res = client.post(
            "/api/v1/runs",
            json={"request": "Conflicting payload", "idempotency_key": "same-key"},
        )
        assert res.status_code == 409
        assert res.json()["code"] == "IDEMPOTENCY_CONFLICT"

    def test_queue_full_returns_429_with_retry_after(self, api_client):
        client, mock_service = api_client
        mock_service.submit_run.side_effect = OrchestrationError(
            message="Execution queue is at maximum capacity. Please retry later.",
            code="EXECUTION_QUEUE_FULL",
            retryable=True,
        )

        res = client.post("/api/v1/runs", json={"request": "Overflow task"})
        assert res.status_code == 429
        assert res.headers.get("Retry-After") == "5"
        assert res.json()["error"] == "Execution queue is at maximum capacity. Please retry later."
