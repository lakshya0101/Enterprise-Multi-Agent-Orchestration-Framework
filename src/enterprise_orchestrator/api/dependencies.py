"""FastAPI dependency injection providers."""

from functools import lru_cache
from typing import Optional

from fastapi import Depends, Request

from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.memory.base import BaseStateStore
from enterprise_orchestrator.memory.in_memory import InMemoryStateStore
from enterprise_orchestrator.memory.postgres import PostgreSQLStateStore
from enterprise_orchestrator.memory.sqlite import SQLiteStateStore
from enterprise_orchestrator.observability.audit.base import BaseAuditSink
from enterprise_orchestrator.observability.audit.in_memory import InMemoryAuditSink
from enterprise_orchestrator.observability.metrics.registry import (
    MetricsRegistry,
    get_global_metrics,
)
from enterprise_orchestrator.observability.tracing.base import BaseTracer
from enterprise_orchestrator.observability.tracing.in_memory import InMemoryTracer
from enterprise_orchestrator.orchestration.runtime import OrchestrationRuntime
from enterprise_orchestrator.security.authenticators import (
    BaseAuthenticator,
    BearerTokenAuthenticator,
    CompositeAuthenticator,
    DevelopmentBypassAuthenticator,
    HashedAPIKeyAuthenticator,
    TestAuthenticator,
)
from enterprise_orchestrator.security.models import AuthenticatedIdentity
from enterprise_orchestrator.services.orchestration_service import OrchestrationService

_GLOBAL_TRACER = InMemoryTracer()
_GLOBAL_AUDIT_SINK = InMemoryAuditSink()


@lru_cache()
def get_settings() -> FrameworkSettings:
    """Cached accessor for FrameworkSettings."""
    return FrameworkSettings()


def get_metrics_registry() -> MetricsRegistry:
    """Provide global metrics registry."""
    return get_global_metrics()


def get_tracer() -> BaseTracer:
    """Provide global tracer."""
    return _GLOBAL_TRACER


def get_audit_sink() -> BaseAuditSink:
    """Provide global audit sink."""
    return _GLOBAL_AUDIT_SINK


def get_state_store(settings: FrameworkSettings = Depends(get_settings)) -> BaseStateStore:
    """Provide configured state persistence store."""
    if settings.state_store_type == "sqlite":
        return SQLiteStateStore(db_path=settings.sqlite_database_path)
    elif settings.state_store_type == "postgres":
        dsn = settings.postgres_dsn or f"postgresql://{settings.postgres_user}:{settings.postgres_password or ''}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        return PostgreSQLStateStore(dsn=dsn)
    return InMemoryStateStore()


def get_runtime(
    state_store: BaseStateStore = Depends(get_state_store),
    metrics: MetricsRegistry = Depends(get_metrics_registry),
    tracer: BaseTracer = Depends(get_tracer),
    audit_sink: BaseAuditSink = Depends(get_audit_sink),
) -> OrchestrationRuntime:
    """Provide OrchestrationRuntime instance with injected state store and observability."""
    return OrchestrationRuntime(
        state_store=state_store,
        metrics=metrics,
        tracer=tracer,
        audit_sink=audit_sink,
    )


def get_orchestration_service(
    runtime: OrchestrationRuntime = Depends(get_runtime),
) -> OrchestrationService:
    """Provide OrchestrationService for coordinating API requests."""
    return OrchestrationService(runtime=runtime)


def get_authenticator(
    settings: FrameworkSettings = Depends(get_settings),
) -> BaseAuthenticator:
    """Resolve configured authentication provider according to environment tier and auth toggle."""
    from enterprise_orchestrator.errors.exceptions import ConfigurationError

    if not settings.api_auth_enabled:
        if settings.environment == "production":
            raise ConfigurationError(
                "Authentication cannot be disabled in a production environment."
            )
        return DevelopmentBypassAuthenticator(environment=settings.environment)

    authenticators: list[BaseAuthenticator] = []

    if settings.api_key_hashes:
        authenticators.append(
            HashedAPIKeyAuthenticator(api_key_hashes=settings.api_key_hashes)
        )

    if settings.jwt_secret_key:
        authenticators.append(
            BearerTokenAuthenticator(
                secret_key=settings.jwt_secret_key,
                algorithm=settings.jwt_algorithm,
            )
        )

    if settings.environment == "test":
        authenticators.append(TestAuthenticator())

    if not authenticators:
        if settings.environment == "production":
            raise ConfigurationError(
                "Production environment requires API_KEY_HASHES or JWT_SECRET_KEY to be configured."
            )
        # In non-production, if auth was requested but no keys configured, add test auth
        authenticators.append(TestAuthenticator())

    return CompositeAuthenticator(authenticators) if len(authenticators) > 1 else authenticators[0]


async def get_current_user(
    request: Request,
    authenticator: BaseAuthenticator = Depends(get_authenticator),
) -> AuthenticatedIdentity:
    """Resolve authenticated identity with fail-closed exception propagation."""
    identity = await authenticator.authenticate(request)
    return identity
