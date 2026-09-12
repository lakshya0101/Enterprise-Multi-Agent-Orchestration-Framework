"""FastAPI authorization dependencies enforcing RBAC permissions and roles."""

from typing import Callable

from fastapi import Depends

from enterprise_orchestrator.errors.exceptions import AuthorizationError
from enterprise_orchestrator.security.models import AuthenticatedIdentity, Permission, Role


def require_authenticated_user() -> Callable:
    """Dependency ensuring the request is made by a verified authenticated identity."""

    from enterprise_orchestrator.api.dependencies import get_current_user

    async def _dependency(
        identity: AuthenticatedIdentity = Depends(get_current_user),
    ) -> AuthenticatedIdentity:
        if not identity or not identity.is_authenticated:
            raise AuthorizationError("Forbidden: Unverified principal.")
        return identity

    return _dependency


def require_permission(permission: Permission) -> Callable:
    """Dependency enforcing that the caller possesses a specific Permission."""

    from enterprise_orchestrator.api.dependencies import get_current_user

    async def _dependency(
        identity: AuthenticatedIdentity = Depends(get_current_user),
    ) -> AuthenticatedIdentity:
        if not identity.has_permission(permission):
            raise AuthorizationError(
                message=f"Forbidden: Principal lacks required permission '{permission.value}'.",
                code="FORBIDDEN",
                details={"required_permission": permission.value, "roles": [r.value for r in identity.roles]},
            )
        return identity

    return _dependency


def require_role(role: Role) -> Callable:
    """Dependency enforcing that the caller possesses a specific Role."""

    from enterprise_orchestrator.api.dependencies import get_current_user

    async def _dependency(
        identity: AuthenticatedIdentity = Depends(get_current_user),
    ) -> AuthenticatedIdentity:
        if not identity.has_role(role):
            raise AuthorizationError(
                message=f"Forbidden: Principal lacks required role '{role.value}'.",
                code="FORBIDDEN",
                details={"required_role": role.value, "roles": [r.value for r in identity.roles]},
            )
        return identity

    return _dependency
