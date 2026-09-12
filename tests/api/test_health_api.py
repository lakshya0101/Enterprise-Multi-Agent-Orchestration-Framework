"""Unit tests for /health and /ready endpoints."""

from fastapi.testclient import TestClient

from enterprise_orchestrator.api.app import create_app


def test_health_and_readiness_endpoints():
    """Verify /health and /ready return 200 OK and expected schemas."""
    app = create_app()
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "ok"
    assert data_health["version"] == "0.1.0"
    assert "timestamp" in data_health

    # 2. Readiness check
    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    data_ready = res_ready.json()
    assert data_ready["status"] == "ready"
    assert "state_store_type" in data_ready
    assert "vector_store_type" in data_ready
    assert "default_llm_provider" in data_ready
