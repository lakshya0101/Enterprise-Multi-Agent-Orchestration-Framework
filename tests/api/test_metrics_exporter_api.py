"""Comprehensive tests for /metrics endpoint, authentication, and headers."""

import hashlib
import pytest
from starlette.testclient import TestClient

from enterprise_orchestrator.api.app import create_app
from enterprise_orchestrator.api.dependencies import get_metrics_registry
from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.observability.metrics.registry import MetricsRegistry


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


def test_metrics_dev_unauthenticated():
    """Verify /metrics in development environment returns Prometheus text format."""
    settings = FrameworkSettings(
        environment="development",
        api_auth_enabled=False,
        prometheus_metrics_require_auth=False,
        prometheus_metrics_enabled=True,
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "runs_started_total" in response.text
    assert "# TYPE runs_started_total counter" in response.text
    assert "X-Content-Type-Options" in response.headers


def test_metrics_production_auth_required(auth_keys):
    """Verify /metrics in production requires authentication and Permission.METRICS_READ."""
    settings = FrameworkSettings(
        environment="production",
        api_auth_enabled=True,
        prometheus_metrics_require_auth=True,
        prometheus_metrics_enabled=True,
        api_key_hashes={
            auth_keys["admin_hash"]: {"subject_id": "admin_principal", "roles": ["ADMIN"]},
            auth_keys["viewer_hash"]: {"subject_id": "viewer_principal", "roles": ["VIEWER"]},
            auth_keys["reviewer_hash"]: {"subject_id": "reviewer_principal", "roles": ["REVIEWER"]},
        },
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    # 1. No credentials -> 401 Unauthorized
    resp_unauth = client.get("/metrics")
    assert resp_unauth.status_code == 401

    # 2. Reviewer lacks METRICS_READ -> 403 Forbidden
    resp_forbidden = client.get("/metrics", headers={"X-API-Key": auth_keys["reviewer_key"]})
    assert resp_forbidden.status_code == 403

    # 3. Viewer has METRICS_READ -> 200 OK
    resp_ok_viewer = client.get("/metrics", headers={"X-API-Key": auth_keys["viewer_key"]})
    assert resp_ok_viewer.status_code == 200
    assert "text/plain" in resp_ok_viewer.headers["content-type"]
    assert "runs_started_total" in resp_ok_viewer.text

    # 4. Admin has METRICS_READ -> 200 OK
    resp_ok_admin = client.get("/metrics", headers={"X-API-Key": auth_keys["admin_key"]})
    assert resp_ok_admin.status_code == 200
    assert "runs_started_total" in resp_ok_admin.text


def test_metrics_production_explicit_unauthenticated_exception(auth_keys):
    """Verify /metrics in production when PROMETHEUS_METRICS_REQUIRE_AUTH=False (deployment exception)."""
    settings = FrameworkSettings(
        environment="production",
        api_auth_enabled=True,
        prometheus_metrics_require_auth=False,
        prometheus_metrics_enabled=True,
        api_key_hashes={
            auth_keys["admin_hash"]: {"subject_id": "admin_principal", "roles": ["ADMIN"]},
        },
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    # Scrape without headers succeeds under explicit unauthenticated exception
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "runs_started_total" in response.text


def test_metrics_disabled():
    """Verify /metrics returns 404 when disabled."""
    settings = FrameworkSettings(
        environment="development",
        prometheus_metrics_enabled=False,
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 404


def test_existing_json_metrics_endpoint_unchanged():
    """Verify existing /api/v1/metrics endpoint returns JSON MetricSnapshot."""
    settings = FrameworkSettings(
        environment="development",
        api_auth_enabled=False,
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

    data = response.json()
    assert "counters" in data
    assert "gauges" in data
    assert "histograms" in data
    assert "runs_started_total" in data["counters"]
    assert "active_runs" in data["gauges"]
