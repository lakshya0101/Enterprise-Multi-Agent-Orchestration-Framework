from typing import List

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from enterprise_orchestrator.api.dependencies import get_orchestration_service, get_settings
from enterprise_orchestrator.api.schemas import (
    CheckpointListResponse,
    CheckpointResponse,
    CreateRunRequest,
    PlanSummary,
    RunResponse,
    RunStatusResponse,
    TaskSummary,
)
from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.memory.base import Checkpoint
from enterprise_orchestrator.security.authorizers import require_permission
from enterprise_orchestrator.security.models import AuthenticatedIdentity, Permission
from enterprise_orchestrator.security.rate_limiter import require_rate_limit
from enterprise_orchestrator.services.orchestration_service import OrchestrationService

router = APIRouter(prefix="/api/v1/runs", tags=["Runs"])


def _state_to_run_response(state: OrchestrationState) -> RunResponse:
    """Map internal domain OrchestrationState to API RunResponse schema."""
    plan_summary = None
    if state.plan:
        tasks_summary = [
            TaskSummary(
                id=t.id,
                title=t.title,
                task_type=t.task_type.value,
                status=t.status.value,
                dependencies=t.dependencies,
                retry_count=t.retry_count,
                max_retries=t.max_retries,
                output_data=t.output_data,
                error=t.error,
            )
            for t in state.plan.tasks
        ]
        plan_summary = PlanSummary(
            rationale=state.plan.rationale,
            tasks=tasks_summary,
        )

    return RunResponse(
        run_id=state.run_id,
        request=state.request,
        status=state.metadata.status.value,
        current_step=state.current_step,
        plan=plan_summary,
        requires_human=state.requires_human,
        final_response=state.final_response,
        errors=state.errors,
        created_at=state.metadata.created_at,
        updated_at=state.metadata.updated_at,
    )


def _state_to_status_response(state: OrchestrationState) -> RunStatusResponse:
    """Map domain OrchestrationState to lightweight RunStatusResponse."""
    completed = 0
    total = 0
    if state.plan:
        total = len(state.plan.tasks)
        completed = sum(1 for t in state.plan.tasks if t.status.value == "completed")

    return RunStatusResponse(
        run_id=state.run_id,
        status=state.metadata.status.value,
        current_step=state.current_step,
        requires_human=state.requires_human,
        has_plan=state.plan is not None,
        completed_tasks=completed,
        total_tasks=total,
    )


def _checkpoint_to_response(cp: Checkpoint) -> CheckpointResponse:
    """Map domain Checkpoint to CheckpointResponse."""
    return CheckpointResponse(
        checkpoint_id=cp.checkpoint_id,
        run_id=cp.run_id,
        step=cp.step,
        timestamp=cp.timestamp,
        status=cp.state.metadata.status.value,
        metadata=cp.metadata,
    )


@router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
async def create_run(
    payload: CreateRunRequest,
    service: OrchestrationService = Depends(get_orchestration_service),
    settings: FrameworkSettings = Depends(get_settings),
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.RUNS_CREATE)),
    rate_limit: AuthenticatedIdentity = Depends(require_rate_limit()),
) -> RunResponse:
    """Create and execute a new multi-agent orchestration run."""
    try:
        final_state = await asyncio.wait_for(
            service.create_and_run(
                request=payload.request,
                correlation_id=payload.correlation_id,
                session_id=payload.session_id,
                custom_context=payload.custom_context,
            ),
            timeout=settings.api_request_timeout_seconds,
        )
        return _state_to_run_response(final_state)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Workflow execution exceeded request deadline.",
        )


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    service: OrchestrationService = Depends(get_orchestration_service),
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.RUNS_READ)),
    rate_limit: AuthenticatedIdentity = Depends(require_rate_limit()),
) -> RunResponse:
    """Retrieve full orchestration state for a given run ID."""
    state = await service.get_state(run_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )
    return _state_to_run_response(state)


@router.get("/{run_id}/status", response_model=RunStatusResponse)
async def get_run_status(
    run_id: str,
    service: OrchestrationService = Depends(get_orchestration_service),
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.RUNS_READ)),
    rate_limit: AuthenticatedIdentity = Depends(require_rate_limit()),
) -> RunStatusResponse:
    """Retrieve lightweight operational status for a given run ID."""
    state = await service.get_state(run_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )
    return _state_to_status_response(state)


@router.get("/{run_id}/checkpoints", response_model=CheckpointListResponse)
async def get_run_checkpoints(
    run_id: str,
    service: OrchestrationService = Depends(get_orchestration_service),
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.RUNS_READ)),
    rate_limit: AuthenticatedIdentity = Depends(require_rate_limit()),
) -> CheckpointListResponse:
    """Retrieve all chronological execution step checkpoints for a run."""
    state = await service.get_state(run_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    checkpoints = await service.list_checkpoints(run_id)
    responses = [_checkpoint_to_response(cp) for cp in checkpoints]
    return CheckpointListResponse(
        run_id=run_id,
        total_checkpoints=len(responses),
        checkpoints=responses,
    )
