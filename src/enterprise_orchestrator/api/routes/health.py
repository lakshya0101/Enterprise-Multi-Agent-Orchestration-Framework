"""Health and readiness liveness probes."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from enterprise_orchestrator.api.dependencies import get_settings
from enterprise_orchestrator.api.schemas import HealthResponse, ReadinessResponse
from enterprise_orchestrator.config.settings import FrameworkSettings

router = APIRouter(tags=["Health & Readiness"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe indicating that the FastAPI service is running."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(
    settings: FrameworkSettings = Depends(get_settings),
) -> ReadinessResponse:
    """Readiness probe verifying framework configuration and subsystem initialization."""
    return ReadinessResponse(
        status="ready",
        state_store_type=settings.state_store_type,
        vector_store_type=settings.vector_store_type,
        default_llm_provider=settings.default_provider,
    )
