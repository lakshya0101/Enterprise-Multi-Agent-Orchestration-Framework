"""Operational telemetry and metrics endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from enterprise_orchestrator.api.dependencies import (
    get_metrics_registry,
    get_orchestration_service,
    get_tracer,
)
from enterprise_orchestrator.observability.metrics.models import MetricSnapshot
from enterprise_orchestrator.observability.metrics.registry import MetricsRegistry
from enterprise_orchestrator.observability.tracing.base import BaseTracer
from enterprise_orchestrator.security.authorizers import require_permission
from enterprise_orchestrator.security.models import AuthenticatedIdentity, Permission
from enterprise_orchestrator.security.rate_limiter import require_rate_limit
from enterprise_orchestrator.services.orchestration_service import OrchestrationService

router = APIRouter(tags=["Metrics & Telemetry"])


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
