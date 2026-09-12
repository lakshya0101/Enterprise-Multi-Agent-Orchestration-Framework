"""Human-in-the-loop review and decision endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from enterprise_orchestrator.api.dependencies import get_current_user, get_orchestration_service
from enterprise_orchestrator.api.schemas import (
    HumanDecisionRequest,
    HumanDecisionResponse,
    HumanReviewResponse,
)
from enterprise_orchestrator.core.human import HumanDecision
from enterprise_orchestrator.core.types import HumanRequestStatus
from enterprise_orchestrator.errors.exceptions import AuthorizationError, OrchestrationError
from enterprise_orchestrator.security.authorizers import require_permission
from enterprise_orchestrator.security.models import AuthenticatedIdentity, Permission
from enterprise_orchestrator.security.rate_limiter import require_rate_limit
from enterprise_orchestrator.services.orchestration_service import OrchestrationService

router = APIRouter(prefix="/api/v1/runs", tags=["Human-in-the-Loop"])


@router.get("/{run_id}/human-review", response_model=HumanReviewResponse)
async def get_human_review(
    run_id: str,
    service: OrchestrationService = Depends(get_orchestration_service),
    identity: AuthenticatedIdentity = Depends(require_permission(Permission.HITL_READ)),
    rate_limit: AuthenticatedIdentity = Depends(require_rate_limit()),
) -> HumanReviewResponse:
    """Retrieve pending human escalation request for a paused run."""
    state = await service.get_state(run_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    pending_req = await service.get_pending_human_review(run_id)
    if not pending_req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pending human review request for run '{run_id}'.",
        )

    return HumanReviewResponse(
        request_id=pending_req.request_id,
        run_id=pending_req.run_id,
        task_id=pending_req.task_id,
        reason=pending_req.reason,
        requested_action=pending_req.requested_action,
        context=pending_req.context,
        status=pending_req.status.value,
        created_at=pending_req.created_at,
    )


@router.post("/{run_id}/human-review", response_model=HumanDecisionResponse)
async def submit_human_decision(
    run_id: str,
    payload: HumanDecisionRequest,
    service: OrchestrationService = Depends(get_orchestration_service),
    identity: AuthenticatedIdentity = Depends(get_current_user),
    rate_limit: AuthenticatedIdentity = Depends(require_rate_limit()),
) -> HumanDecisionResponse:
    """Submit a human decision to approve, modify, or reject an escalation and resume execution."""
    status_str = payload.status.strip().lower()
    try:
        decision_status = HumanRequestStatus(status_str)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid decision status '{payload.status}'. Must be one of: APPROVED, REJECTED, MODIFIED.",
        )

    # Dynamic RBAC check based on decision action
    required_permission = {
        HumanRequestStatus.APPROVED: Permission.HITL_APPROVE,
        HumanRequestStatus.MODIFIED: Permission.HITL_MODIFY,
        HumanRequestStatus.REJECTED: Permission.HITL_REJECT,
    }.get(decision_status)

    if required_permission and not identity.has_permission(required_permission):
        raise AuthorizationError(
            message=f"Forbidden: Principal lacks required permission '{required_permission.value}' for {decision_status.value} decision.",
            code="FORBIDDEN",
            details={"required_permission": required_permission.value, "roles": [r.value for r in identity.roles]},
        )

    # Security: Server-side identity binding. Do NOT trust client-supplied reviewer_id.
    decision = HumanDecision(
        request_id=payload.request_id,
        status=decision_status,
        decision_note=payload.decision_note,
        modified_payload=payload.modified_payload,
        reviewer_id=identity.subject_id,
    )

    try:
        resumed_state = await service.submit_human_decision(run_id=run_id, decision=decision)
        return HumanDecisionResponse(
            run_id=resumed_state.run_id,
            status=resumed_state.metadata.status.value,
            resumed_successfully=True,
            final_response=resumed_state.final_response,
        )
    except OrchestrationError as e:
        if e.code == "RUN_NOT_FOUND":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)
        elif e.code in ("NO_HUMAN_REVIEW_PENDING", "RUN_TERMINAL"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.message)
