"""Static contract and configuration tests for Dockerfile, Compose, and CI workflows."""

import os
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_dockerfile_contracts():
    """Verify Dockerfile structure, base image, non-root user, and entrypoint."""
    dockerfile_path = REPO_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist at repository root."

    content = dockerfile_path.read_text(encoding="utf-8")

    # Multi-stage validation
    assert "FROM python:3.10-slim-bookworm AS builder" in content
    assert "FROM python:3.10-slim-bookworm AS runtime" in content

    # Non-root user validation
    assert "10001" in content
    assert "orchestrator" in content
    assert "USER orchestrator" in content

    # Runtime hardening flags
    assert "PYTHONDONTWRITEBYTECODE=1" in content
    assert "PYTHONUNBUFFERED=1" in content
    assert "PIP_NO_CACHE_DIR=1" in content

    # Workdir and entrypoint
    assert "WORKDIR /app" in content
    assert "EXPOSE 8000" in content
    assert "uvicorn" in content
    assert "enterprise_orchestrator.api.app:create_app" in content
    assert "--factory" in content
    assert "0.0.0.0" in content
    assert "8000" in content


def test_dockerignore_contracts():
    """Verify .dockerignore exclusions for secrets and development artifacts."""
    dockerignore_path = REPO_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist at repository root."

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    required_exclusions = [
        ".git",
        ".env",
        ".env.*",
        "__pycache__",
        "*.py[cod]",
        ".pytest_cache",
        ".venv",
        "data/",
        "*.db",
        "*.sqlite3",
    ]
    for pattern in required_exclusions:
        assert pattern in lines, f"Missing required exclusion: {pattern}"


def test_docker_compose_contracts():
    """Verify docker-compose.yml services, networking, healthchecks, and secrets safety."""
    compose_path = REPO_ROOT / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist at repository root."

    content = compose_path.read_text(encoding="utf-8")

    # Service definitions
    assert "services:" in content
    assert "db:" in content
    assert "app:" in content
    assert "postgres:16-alpine" in content

    # Security: No hardcoded passwords or insecure fallback defaults
    assert "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?" in content
    assert "postgres_dev_password" not in content

    # PostgreSQL port must NOT be published to the host
    assert '"5432:5432"' not in content
    assert "'5432:5432'" not in content

    # Application port mapping
    assert '"${PORT:-8000}:8000"' in content or "'${PORT:-8000}:8000'" in content or "${PORT:-8000}:8000" in content

    # Healthchecks
    assert "pg_isready" in content
    assert "condition: service_healthy" in content
    assert "/health" in content
    assert "urllib.request" in content

    # Networks and Volumes
    assert "orchestrator_net:" in content
    assert "pgdata:" in content

    # Environment variables wiring
    assert "STATE_STORE_TYPE=postgres" in content
    assert "POSTGRES_HOST=db" in content
    assert "POSTGRES_PORT=5432" in content


def test_github_actions_ci_workflow():
    """Verify GitHub Actions CI workflow for zero-cost build validation."""
    workflow_path = REPO_ROOT / ".github" / "workflows" / "docker-build.yml"
    assert workflow_path.exists(), "CI workflow must exist at .github/workflows/docker-build.yml."

    content = workflow_path.read_text(encoding="utf-8")

    # Trigger events
    assert "push:" in content
    assert "pull_request:" in content
    assert "branches: [ \"main\" ]" in content or "branches: [ main ]" in content or "main" in content

    # Build validation steps without push
    assert "docker/build-push-action" in content or "docker build" in content
    assert "push: false" in content
    assert "enterprise-orchestrator:test" in content
