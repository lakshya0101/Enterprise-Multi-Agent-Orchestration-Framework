# ==============================================================================
# Enterprise Multi-Agent Orchestration Framework - Production Dockerfile
# Multi-stage build based on python:3.10-slim-bookworm
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder Stage
# ------------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS builder

WORKDIR /build

# Install minimal compilation tools for native extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy package definitions and source to build wheels
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Build wheels for project and all optional extras (PostgreSQL + OpenTelemetry)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels ".[postgres,telemetry]"

# ------------------------------------------------------------------------------
# Stage 2: Runtime Stage
# ------------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS runtime

# Python and environment runtime hardening flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Create unprivileged non-root system user and group
RUN groupadd -g 10001 orchestrator && \
    useradd -u 10001 -g orchestrator -s /bin/bash -m orchestrator

WORKDIR /app

# Install pre-built wheels from builder stage without compiler tools
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application source code with non-root ownership
COPY --chown=orchestrator:orchestrator . /app

# Ensure data persistence directory exists with proper permissions
RUN mkdir -p /app/data && \
    chown -R orchestrator:orchestrator /app

# Switch to non-root execution user
USER orchestrator

# Expose FastAPI application port
EXPOSE 8000

# Production startup using Uvicorn application factory
CMD ["uvicorn", "enterprise_orchestrator.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
