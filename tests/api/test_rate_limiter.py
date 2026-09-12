"""Tests for in-memory sliding window rate limiting and independent principal quotas."""

import hashlib
from unittest.mock import AsyncMock
import pytest
from starlette.testclient import TestClient

from enterprise_orchestrator.api.app import create_app
from enterprise_orchestrator.api.dependencies import get_orchestration_service
from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.security.rate_limiter import get_rate_limiter
from enterprise_orchestrator.services.orchestration_service import OrchestrationService


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    limiter = get_rate_limiter()
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def rate_limited_app():
    user1_key = "key-user-1"
    user2_key = "key-user-2"
    hash1 = hashlib.sha256(user1_key.encode()).hexdigest()
    hash2 = hashlib.sha256(user2_key.encode()).hexdigest()

    settings = FrameworkSettings(
        environment="production",
        api_auth_enabled=True,
        api_key_hashes={
            hash1: {"subject_id": "principal_alpha", "roles": ["ADMIN"]},
            hash2: {"subject_id": "principal_beta", "roles": ["ADMIN"]},
        },
        rate_limit_enabled=True,
        rate_limit_authenticated_per_minute=3,  # Strict low limit for testing
        rate_limit_public_per_minute=2,
    )

    app = create_app(settings=settings)
    mock_service = AsyncMock(spec=OrchestrationService)
    sample_state = OrchestrationState.create_initial(request="Test run")
    mock_service.create_and_run.return_value = sample_state
    mock_service.get_state.return_value = sample_state
    app.dependency_overrides[get_orchestration_service] = lambda: mock_service

    app.state.user1_key = user1_key
    app.state.user2_key = user2_key
    return app


class TestRateLimiter:
    """Test authenticated and public rate limiting behavior."""

    def test_independent_authenticated_principal_rate_limits(self, rate_limited_app):
        client = TestClient(rate_limited_app)
        user1_headers = {"X-API-Key": rate_limited_app.state.user1_key}
        user2_headers = {"X-API-Key": rate_limited_app.state.user2_key}

        # User 1 makes 3 requests (quota exhausted)
        for i in range(3):
            res = client.get("/api/v1/runs/run-123", headers=user1_headers)
            assert res.status_code == 200

        # User 1 4th request -> 429 Too Many Requests
        res_user1_blocked = client.get("/api/v1/runs/run-123", headers=user1_headers)
        assert res_user1_blocked.status_code == 429
        assert "Retry-After" in res_user1_blocked.headers
        assert "Rate limit exceeded" in res_user1_blocked.json()["error"]

        # User 2 makes a request -> Must succeed independently (User 1's limit does NOT block User 2)
        res_user2 = client.get("/api/v1/runs/run-123", headers=user2_headers)
        assert res_user2.status_code == 200

    def test_public_endpoint_ip_rate_limiting(self, rate_limited_app):
        client = TestClient(rate_limited_app)

        # Public limit is 2 per minute
        res1 = client.get("/health")
        assert res1.status_code == 200
        res2 = client.get("/health")
        assert res2.status_code == 200

        # 3rd request -> 429
        res3 = client.get("/health")
        assert res3.status_code == 429
        assert "Retry-After" in res3.headers
        assert res3.json()["code"] == "RATE_LIMIT_EXCEEDED"
