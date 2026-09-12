"""Comprehensive offline API authentication and RBAC integration tests."""

import hashlib
from typing import Dict
from unittest.mock import AsyncMock
import pytest
from starlette.testclient import TestClient

from enterprise_orchestrator.api.app import create_app
from enterprise_orchestrator.api.dependencies import (
    get_authenticator,
    get_orchestration_service,
    get_settings,
)
from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.core.human import HumanEscalationRequest
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.types import ExecutionStatus, HumanRequestStatus
from enterprise_orchestrator.errors.exceptions import ConfigurationError
from enterprise_orchestrator.security.authenticators import HashedAPIKeyAuthenticator
from enterprise_orchestrator.services.orchestration_service import OrchestrationService


@pytest.fixture
def auth_keys():
    admin_key = "admin-secret-key-12345"
    viewer_key = "viewer-secret-key-67890"
    reviewer_key = "reviewer-secret-key-abcde"
    return {
        "admin_key": admin_key,
        "admin_hash": hashlib.sha256(admin_key.encode()).hexdigest(),
        "viewer_key": viewer_key,
        "viewer_hash": hashlib.sha256(viewer_key.encode()).hexdigest(),
        "reviewer_key": reviewer_key,
        "reviewer_hash": hashlib.sha256(reviewer_key.encode()).hexdigest(),
    }


@pytest.fixture
def secure_app(auth_keys):
    settings = FrameworkSettings(
        environment="production",
        api_auth_enabled=True,
        api_key_hashes={
            auth_keys["admin_hash"]: {"subject_id": "admin_principal", "roles": ["ADMIN"]},
            auth_keys["viewer_hash"]: {"subject_id": "viewer_principal", "roles": ["VIEWER"]},
            auth_keys["reviewer_hash"]: {"subject_id": "reviewer_principal", "roles": ["REVIEWER"]},
        },
    )

    app = create_app(settings=settings)

    # Mock service
    service = AsyncMock(spec=OrchestrationService)

    sample_state = OrchestrationState.create_initial(
        request="Test secure run",
        correlation_id="corr-sec-1",
    )
    run_id = sample_state.run_id
    service.create_and_run.return_value = sample_state
    service.get_state.return_value = sample_state

    pending_req = HumanEscalationRequest(
        request_id="req-sec-999",
        run_id=run_id,
        task_id="task-1",
        reason="Needs approval",
        requested_action="approve_or_reject",
        status=HumanRequestStatus.PENDING,
    )
    service.get_pending_human_review.return_value = pending_req
    service.submit_human_decision.return_value = sample_state

    app.dependency_overrides[get_orchestration_service] = lambda: service
    app.state.sample_run_id = run_id
    return app


class TestPublicAndProtectedEndpoints:
    """Test public route accessibility and protected route authentication enforcement."""

    def test_public_health_and_ready_no_credentials_required(self, secure_app):
        client = TestClient(secure_app)
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "ok"

        res_ready = client.get("/ready")
        assert res_ready.status_code == 200
        assert res_ready.json()["status"] == "ready"

    def test_protected_endpoint_without_credentials_returns_401(self, secure_app):
        client = TestClient(secure_app)
        res = client.post("/api/v1/runs", json={"request": "Test request"})
        assert res.status_code == 401
        assert "WWW-Authenticate" in res.headers
        data = res.json()
        assert data["code"] == "MISSING_CREDENTIALS"

    def test_protected_endpoint_with_invalid_credentials_returns_401(self, secure_app):
        client = TestClient(secure_app)
        res = client.post(
            "/api/v1/runs",
            json={"request": "Test request"},
            headers={"X-API-Key": "invalid-wrong-key"},
        )
        assert res.status_code == 401
        data = res.json()
        assert data["code"] == "AUTHENTICATION_FAILED"

    def test_protected_endpoint_with_valid_admin_credentials_returns_201(self, secure_app, auth_keys):
        client = TestClient(secure_app)
        res = client.post(
            "/api/v1/runs",
            json={"request": "Test request"},
            headers={"X-API-Key": auth_keys["admin_key"]},
        )
        assert res.status_code == 201
        assert res.json()["run_id"] == secure_app.state.sample_run_id


class TestRBACPermissionsEnforcement:
    """Test 403 Forbidden for authenticated principals lacking required permissions."""

    def test_viewer_cannot_create_run_403(self, secure_app, auth_keys):
        client = TestClient(secure_app)
        res = client.post(
            "/api/v1/runs",
            json={"request": "Viewer trying to create run"},
            headers={"X-API-Key": auth_keys["viewer_key"]},
        )
        assert res.status_code == 403
        data = res.json()
        assert data["code"] == "FORBIDDEN"
        assert "runs:create" in data["error"]

    def test_viewer_can_read_run_200(self, secure_app, auth_keys):
        client = TestClient(secure_app)
        run_id = secure_app.state.sample_run_id
        res = client.get(
            f"/api/v1/runs/{run_id}",
            headers={"X-API-Key": auth_keys["viewer_key"]},
        )
        assert res.status_code == 200
        assert res.json()["run_id"] == run_id

    def test_viewer_can_read_metrics_200(self, secure_app, auth_keys):
        client = TestClient(secure_app)
        res = client.get(
            "/api/v1/metrics",
            headers={"X-API-Key": auth_keys["viewer_key"]},
        )
        assert res.status_code == 200


class TestHITLServerSideIdentityBinding:
    """Test server-side identity resolution and non-repudiation in HITL decisions."""

    def test_viewer_cannot_approve_hitl_403(self, secure_app, auth_keys):
        client = TestClient(secure_app)
        run_id = secure_app.state.sample_run_id
        res = client.post(
            f"/api/v1/runs/{run_id}/human-review",
            json={
                "request_id": "req-sec-999",
                "status": "APPROVED",
                "decision_note": "Viewer attempt",
            },
            headers={"X-API-Key": auth_keys["viewer_key"]},
        )
        assert res.status_code == 403
        assert "hitl:approve" in res.json()["error"]

    def test_reviewer_can_approve_and_identity_is_bound_server_side(self, secure_app, auth_keys):
        client = TestClient(secure_app)
        run_id = secure_app.state.sample_run_id
        res = client.post(
            f"/api/v1/runs/{run_id}/human-review",
            json={
                "request_id": "req-sec-999",
                "status": "APPROVED",
                "decision_note": "Looks good",
                "reviewer_id": "forged_client_id_attacker",
            },
            headers={"X-API-Key": auth_keys["reviewer_key"]},
        )
        assert res.status_code == 200

        # Verify that OrchestrationService received the authenticated subject_id, NOT the forged reviewer_id
        service: AsyncMock = secure_app.dependency_overrides[get_orchestration_service]()
        submitted_decision = service.submit_human_decision.call_args[1]["decision"]
        assert submitted_decision.reviewer_id == "reviewer_principal"
        assert submitted_decision.reviewer_id != "forged_client_id_attacker"


class TestFailClosedProductionAndLeakage:
    """Test fail-closed production semantics and verify credentials never leak."""

    def test_production_fails_closed_when_auth_disabled(self):
        settings = FrameworkSettings(
            environment="production",
            api_auth_enabled=False,
        )
        # Direct dependency resolution raises ConfigurationError
        with pytest.raises(ConfigurationError) as exc_info:
            get_authenticator(settings=settings)
        assert "Authentication cannot be disabled in a production environment" in str(exc_info.value)

        # API endpoint invocation returns 400/error response
        app = create_app(settings=settings)
        client = TestClient(app)
        res = client.get("/api/v1/runs/run-123")
        assert res.status_code == 400
        assert res.json()["code"] == "CONFIGURATION_ERROR"

    def test_production_fails_closed_when_no_keys_configured(self):
        settings = FrameworkSettings(
            environment="production",
            api_auth_enabled=True,
            api_key_hashes={},
            jwt_secret_key=None,
        )
        # Direct dependency resolution raises ConfigurationError
        with pytest.raises(ConfigurationError) as exc_info:
            get_authenticator(settings=settings)
        assert "Production environment requires" in str(exc_info.value)

        # API endpoint invocation returns 400/error response
        app = create_app(settings=settings)
        client = TestClient(app)
        res = client.get("/api/v1/runs/run-123")
        assert res.status_code == 400
        assert res.json()["code"] == "CONFIGURATION_ERROR"

    def test_credential_leakage_defense(self, secure_app, auth_keys):
        client = TestClient(secure_app)
        # 1. Invalid key error response does not contain the key or hash
        res = client.post(
            "/api/v1/runs",
            json={"request": "Test"},
            headers={"X-API-Key": "my-secret-key-123"},
        )
        body = res.text
        assert "my-secret-key-123" not in body
        assert auth_keys["admin_hash"] not in body
        assert "stack_trace" not in body
