"""Server-Sent Events (SSE) streaming endpoint for live execution progress."""

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from enterprise_orchestrator.api.dependencies import get_orchestration_service
from enterprise_orchestrator.security.authorizers import require_permission
from enterprise_orchestrator.security.models import AuthenticatedIdentity, Permission
from enterprise_orchestrator.services.orchestration_service import OrchestrationService

router = APIRouter(prefix="/api/v1/runs", tags=["Events"])


async def _sse_generator(
    request: Request,
    service: OrchestrationService,
    run_id: str,
) -> AsyncGenerator[str, None]:
    """Format event dictionaries into standard text/event-stream chunks with disconnect detection."""
    async for event in service.subscribe_events(run_id):
        if await request.is_disconnected():
            break
        event_name = event.get("event", "message")
        payload = json.dumps(event)
        yield f"event: {event_name}\ndata: {payload}\n\n"


@router.get("/{run_id}/events")
async def stream_run_events(
    request: Request,
    run_id: str,
    service: OrchestrationService = Depends(get_orchestration_service),
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.RUNS_STREAM)),
) -> StreamingResponse:
    """Stream live workflow execution status events using Server-Sent Events (SSE)."""
    state = await service.get_state(run_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    return StreamingResponse(
        _sse_generator(request, service, run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
