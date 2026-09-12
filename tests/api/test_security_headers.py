"""Tests for standard defensive HTTP security headers."""

from starlette.testclient import TestClient

from enterprise_orchestrator.api.app import create_app
from enterprise_orchestrator.config.settings import FrameworkSettings


class TestSecurityHeaders:
    """Test security header injection across public and protected endpoints."""

    def test_security_headers_present_on_all_responses(self):
        settings = FrameworkSettings(
            environment="development",
            api_auth_enabled=False,
            hsts_enabled=True,
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200

        # Mandatory defensive security headers
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert resp.headers["X-XSS-Protection"] == "0"
        assert resp.headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"
        assert resp.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"

    def test_hsts_omitted_when_disabled(self):
        settings = FrameworkSettings(
            environment="development",
            api_auth_enabled=False,
            hsts_enabled=False,
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        assert "Strict-Transport-Security" not in resp.headers
