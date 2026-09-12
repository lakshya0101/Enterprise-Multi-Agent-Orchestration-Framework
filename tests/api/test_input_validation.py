"""Tests for schema-level input validation constraints and payload boundary enforcement."""

import pytest
from starlette.testclient import TestClient

from enterprise_orchestrator.api.app import create_app
from enterprise_orchestrator.config.settings import FrameworkSettings


class TestInputValidation:
    """Test schema constraints across API requests."""

    @pytest.fixture
    def client(self):
        settings = FrameworkSettings(
            environment="development",
            api_auth_enabled=False,
            max_request_body_bytes=1000000,
        )
        app = create_app(settings=settings)
        return TestClient(app)

    def test_mega_prompt_exceeding_50k_chars_rejected_422(self, client):
        resp = client.post("/api/v1/runs", json={"request": "x" * 50001})
        assert resp.status_code == 422
        assert resp.json()["code"] == "VALIDATION_ERROR"

    def test_custom_context_too_many_keys_rejected_422(self, client):
        too_many_keys = {f"key_{i}": i for i in range(51)}
        resp = client.post("/api/v1/runs", json={"request": "Valid prompt", "custom_context": too_many_keys})
        assert resp.status_code == 422
        assert "cannot contain more than 50 top-level keys" in str(resp.json()["details"])

    def test_invalid_decision_status_rejected_422(self, client):
        resp = client.post(
            "/api/v1/runs/run-123/human-review",
            json={"request_id": "req-1", "status": "INVALID_VERDICT"},
        )
        assert resp.status_code == 422
        assert resp.json()["code"] == "VALIDATION_ERROR"

    def test_invalid_request_id_special_chars_rejected_422(self, client):
        resp = client.post(
            "/api/v1/runs/run-123/human-review",
            json={"request_id": "../../etc/passwd", "status": "APPROVED"},
        )
        assert resp.status_code == 422
        assert resp.json()["code"] == "VALIDATION_ERROR"

    def test_decision_note_exceeding_5000_chars_rejected_422(self, client):
        resp = client.post(
            "/api/v1/runs/run-123/human-review",
            json={"request_id": "req-1", "status": "APPROVED", "decision_note": "y" * 5001},
        )
        assert resp.status_code == 422
        assert resp.json()["code"] == "VALIDATION_ERROR"
