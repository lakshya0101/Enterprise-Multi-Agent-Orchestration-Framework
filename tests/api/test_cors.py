"""Tests for CORS configuration, explicit origin allowlists, and production fail-closed rules."""

import pytest
from starlette.testclient import TestClient

from enterprise_orchestrator.api.app import create_app
from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.errors.exceptions import ConfigurationError


class TestCORS:
    """Test CORS preflight, allowed origins, and fail-closed production invariants."""

    def test_explicit_origin_allowed(self):
        settings = FrameworkSettings(
            environment="production",
            api_auth_enabled=True,
            api_key_hashes={"dummy_hash": "admin"},
            cors_allowed_origins=["https://app.corp.internal"],
            cors_allow_credentials=True,
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        # Preflight OPTIONS request
        resp = client.options(
            "/api/v1/runs",
            headers={
                "Origin": "https://app.corp.internal",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-API-Key, Content-Type",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["Access-Control-Allow-Origin"] == "https://app.corp.internal"
        assert resp.headers["Access-Control-Allow-Credentials"] == "true"
        assert "POST" in resp.headers["Access-Control-Allow-Methods"]

    def test_disallowed_origin_rejected(self):
        settings = FrameworkSettings(
            environment="production",
            api_auth_enabled=True,
            api_key_hashes={"dummy_hash": "admin"},
            cors_allowed_origins=["https://app.corp.internal"],
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        resp = client.options(
            "/api/v1/runs",
            headers={
                "Origin": "https://malicious-attacker.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "Access-Control-Allow-Origin" not in resp.headers

    def test_production_fails_closed_on_wildcard_origin_without_credentials(self):
        settings = FrameworkSettings(
            environment="production",
            api_auth_enabled=True,
            api_key_hashes={"dummy_hash": "admin"},
            cors_allowed_origins=["*"],
            cors_allow_credentials=False,
        )
        with pytest.raises(ConfigurationError) as exc_info:
            create_app(settings=settings)
        assert "Wildcard CORS origin '*' is strictly forbidden in production" in str(exc_info.value)

    def test_production_fails_closed_on_wildcard_origin_with_credentials(self):
        settings = FrameworkSettings(
            environment="production",
            api_auth_enabled=True,
            api_key_hashes={"dummy_hash": "admin"},
            cors_allowed_origins=["*"],
            cors_allow_credentials=True,
        )
        with pytest.raises(ConfigurationError) as exc_info:
            create_app(settings=settings)
        assert "Wildcard CORS origin '*' is strictly forbidden in production" in str(exc_info.value)
