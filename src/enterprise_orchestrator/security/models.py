"""Security domain models, roles, permissions, and authenticated identity contracts."""

from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Enterprise access control roles."""

    ADMIN = "admin"
    OPERATOR = "operator"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class Permission(str, Enum):
    """Granular system action permissions."""

    # Workflow operations
    RUNS_CREATE = "runs:create"
    RUNS_READ = "runs:read"
    RUNS_DELETE = "runs:delete"
    RUNS_STREAM = "runs:stream"

    # Human-in-the-loop governance
    HITL_READ = "hitl:read"
    HITL_APPROVE = "hitl:approve"
    HITL_REJECT = "hitl:reject"
    HITL_MODIFY = "hitl:modify"

    # Telemetry and Audit
    METRICS_READ = "metrics:read"
    AUDIT_READ = "audit:read"


ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.RUNS_CREATE,
        Permission.RUNS_READ,
        Permission.RUNS_DELETE,
        Permission.RUNS_STREAM,
        Permission.HITL_READ,
        Permission.HITL_APPROVE,
        Permission.HITL_REJECT,
        Permission.HITL_MODIFY,
        Permission.METRICS_READ,
        Permission.AUDIT_READ,
    },
    Role.OPERATOR: {
        Permission.RUNS_CREATE,
        Permission.RUNS_READ,
        Permission.RUNS_STREAM,
        Permission.HITL_READ,
        Permission.HITL_APPROVE,
        Permission.HITL_REJECT,
        Permission.HITL_MODIFY,
        Permission.METRICS_READ,
    },
    Role.REVIEWER: {
        Permission.RUNS_READ,
        Permission.RUNS_STREAM,
        Permission.HITL_READ,
        Permission.HITL_APPROVE,
        Permission.HITL_REJECT,
        Permission.HITL_MODIFY,
    },
    Role.VIEWER: {
        Permission.RUNS_READ,
        Permission.RUNS_STREAM,
        Permission.HITL_READ,
        Permission.METRICS_READ,
    },
}


class AuthenticatedIdentity(BaseModel):
    """Verified principal identity established by an authentication provider."""

    subject_id: str = Field(..., min_length=1, description="Unique immutable principal identifier (user_id/service_id)")
    username: str = Field(..., description="Human-readable principal or service name")
    roles: Set[Role] = Field(default_factory=set, description="Assigned role memberships")
    permissions: Set[Permission] = Field(default_factory=set, description="Resolved granular permissions")
    auth_method: str = Field(..., description="Authentication scheme used ('hashed_api_key', 'bearer_jwt', 'dev_bypass')")
    is_authenticated: bool = Field(default=True, description="Whether authentication was successfully verified")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Non-sensitive claims (e.g. email, department)")

    def has_permission(self, permission: Permission) -> bool:
        """Check if identity possesses the required permission."""
        if permission in self.permissions:
            return True
        for role in self.roles:
            if permission in ROLE_PERMISSIONS.get(role, set()):
                return True
        return False

    def has_role(self, role: Role) -> bool:
        """Check if identity has a specific role."""
        return role in self.roles
