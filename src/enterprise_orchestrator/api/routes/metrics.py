"""Operational telemetry and metrics endpoints."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from enterprise_orchestrator.api.dependencies import (
    get_authenticator,
    get_metrics_registry,
    get_orchestration_service,
    get_settings,
    get_tracer,
)
from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.observability.metrics.models import MetricSnapshot
from enterprise_orchestrator.observability.metrics.prometheus import (
    PrometheusTextSerializer,
)
from enterprise_orchestrator.observability.metrics.registry import MetricsRegistry
from enterprise_orchestrator.observability.tracing.base import BaseTracer
from enterprise_orchestrator.security.authorizers import require_permission
from enterprise_orchestrator.security.models import AuthenticatedIdentity, Permission
from enterprise_orchestrator.security.rate_limiter import get_rate_limiter, require_rate_limit
from enterprise_orchestrator.services.orchestration_service import OrchestrationService

router = APIRouter(tags=["Metrics & Telemetry"])
_PROMETHEUS_SERIALIZER = PrometheusTextSerializer()


async def verify_prometheus_access(
    request: Request,
    settings: FrameworkSettings = Depends(get_settings),
    authenticator: Any = Depends(get_authenticator),
) -> Optional[AuthenticatedIdentity]:
    """Verify access policy for Prometheus scrape endpoint."""
    if not settings.prometheus_metrics_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prometheus metrics endpoint is disabled.",
        )

    if settings.prometheus_metrics_require_auth:
        identity = await authenticator.authenticate(request)
        if not identity.has_permission(Permission.METRICS_READ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Principal lacks required permission: metrics:read",
            )

        if settings.rate_limit_enabled:
            rate_limiter = get_rate_limiter()
            limit = settings.rate_limit_authenticated_per_minute
            is_allowed, _, retry_after = await rate_limiter.check_rate_limit(
                key=f"sub:{identity.subject_id}",
                max_requests=limit,
                window_seconds=60,
            )
            if not is_allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(int(retry_after) or 1)},
                )
        return identity

    return None


@router.get("/metrics")
async def get_prometheus_metrics(
    request: Request,
    registry: MetricsRegistry = Depends(get_metrics_registry),
    _auth: Optional[AuthenticatedIdentity] = Depends(verify_prometheus_access),
) -> Response:
    """Export operational metrics in standard Prometheus text exposition format."""
    content = _PROMETHEUS_SERIALIZER.serialize_registry(registry)
    return Response(
        content=content,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/api/v1/metrics", response_model=MetricSnapshot)
async def get_metrics(
    registry: MetricsRegistry = Depends(get_metrics_registry),
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.METRICS_READ)),
    rate_limit: AuthenticatedIdentity = Depends(require_rate_limit()),
) -> MetricSnapshot:
    """Retrieve global operational metrics snapshot with counters, gauges, and percentile histograms."""
    return registry.get_snapshot()


@router.get("/api/v1/runs/{run_id}/metrics")
async def get_run_metrics(
    run_id: str,
    service: OrchestrationService = Depends(get_orchestration_service),
    tracer: BaseTracer = Depends(get_tracer),
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.METRICS_READ)),
    rate_limit: AuthenticatedIdentity = Depends(require_rate_limit()),
) -> Dict[str, Any]:
    """Retrieve telemetry summary and span execution breakdown for a specific run ID."""
    state = await service.get_state(run_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    spans = tracer.get_spans(run_id=run_id)
    span_summaries = [
        {
            "name": s.name,
            "kind": s.kind.value,
            "duration_ms": s.duration_ms,
            "status": s.status.value,
            "error": s.error_message,
        }
        for s in spans
    ]

    total_tasks = len(state.plan.tasks) if state.plan else 0
    completed_tasks = sum(1 for t in state.plan.tasks if t.status.value == "completed") if state.plan else 0

    return {
        "run_id": run_id,
        "status": state.metadata.status.value,
        "current_step": state.current_step,
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "requires_human": state.requires_human,
        "spans_count": len(spans),
        "spans": span_summaries,
    }
