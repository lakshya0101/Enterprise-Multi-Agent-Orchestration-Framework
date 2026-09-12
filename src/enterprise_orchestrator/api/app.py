"""FastAPI application factory, lifespan management, and global error handling."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from enterprise_orchestrator.api.dependencies import get_settings
from enterprise_orchestrator.api.middleware import (
    CorrelationIdMiddleware,
    PublicRateLimitMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from enterprise_orchestrator.api.routes import (
    events_router,
    health_router,
    human_router,
    metrics_router,
    runs_router,
)
from enterprise_orchestrator.api.schemas import ErrorResponse
from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.errors.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    OrchestratorError,
)


from enterprise_orchestrator.execution.idempotency import IdempotencyConflictError


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for application initialization and clean teardown."""
    # Startup: ensure configuration is accessible and execution backend starts if present
    service = getattr(app.state, "orchestration_service", None)
    if service and hasattr(service, "execution_backend"):
        await service.execution_backend.start()
    yield
    # Shutdown: clean up and drain worker tasks
    if service and hasattr(service, "execution_backend"):
        await service.execution_backend.shutdown(timeout_seconds=30.0)


def create_app(settings: Optional[FrameworkSettings] = None) -> FastAPI:
    """Construct and configure FastAPI application instance."""
    cfg = settings or FrameworkSettings()

    # Fail-closed validation for production CORS configuration
    if cfg.environment == "production" and ("*" in cfg.cors_allowed_origins or any(o.strip() == "*" for o in cfg.cors_allowed_origins)):
        raise ConfigurationError("Wildcard CORS origin '*' is strictly forbidden in production environment.")

    app = FastAPI(
        title="Enterprise Multi-Agent Orchestration API",
        description="REST and SSE API for coordinating multi-agent workflows, DAG execution, and human review.",
        version="0.1.0",
        debug=cfg.api_debug,
        lifespan=app_lifespan,
    )

    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: cfg

    # 1. Register ASGI Middlewares (Ordered pipeline)
    app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=cfg.max_request_body_bytes)
    app.add_middleware(SecurityHeadersMiddleware, hsts_enabled=cfg.hsts_enabled)
    app.add_middleware(CorrelationIdMiddleware)

    if cfg.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_allowed_origins,
            allow_credentials=cfg.cors_allow_credentials,
            allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Correlation-ID", "WWW-Authenticate"],
            max_age=600,
        )

    app.add_middleware(PublicRateLimitMiddleware, settings=cfg)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        from fastapi.encoders import jsonable_encoder

        error_resp = ErrorResponse(
            error="Request validation failed.",
            code="VALIDATION_ERROR",
            details=jsonable_encoder(exc.errors()),
        )
        return JSONResponse(
            status_code=422,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        error_resp = ErrorResponse(
            error=str(exc.detail),
            code="HTTP_ERROR",
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_resp.model_dump(),
            headers=exc.headers,
        )

    @app.exception_handler(AuthenticationError)
    async def authentication_exception_handler(
        request: Request, exc: AuthenticationError
    ) -> JSONResponse:
        error_resp = ErrorResponse(
            error=exc.message,
            code=exc.code,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=error_resp.model_dump(),
            headers={"WWW-Authenticate": "Bearer, ApiKey"},
        )

    @app.exception_handler(AuthorizationError)
    async def authorization_exception_handler(
        request: Request, exc: AuthorizationError
    ) -> JSONResponse:
        error_resp = ErrorResponse(
            error=exc.message,
            code=exc.code,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(IdempotencyConflictError)
    async def idempotency_conflict_handler(
        request: Request, exc: IdempotencyConflictError
    ) -> JSONResponse:
        error_resp = ErrorResponse(
            error=exc.message,
            code=exc.code,
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(OrchestratorError)
    async def framework_exception_handler(
        request: Request, exc: OrchestratorError
    ) -> JSONResponse:
        status_code = status.HTTP_400_BAD_REQUEST
        if exc.code == "RUN_NOT_FOUND":
            status_code = status.HTTP_404_NOT_FOUND

        error_resp = ErrorResponse(
            error=exc.message,
            code=exc.code,
        )
        return JSONResponse(
            status_code=status_code,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        error_resp = ErrorResponse(
            error="An unexpected internal server error occurred.",
            code="INTERNAL_SERVER_ERROR",
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_resp.model_dump(),
        )

    # 2. Mount Routers
    app.include_router(health_router)
    app.include_router(runs_router)
    app.include_router(human_router)
    app.include_router(events_router)
    app.include_router(metrics_router)

    return app
