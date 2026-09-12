"""RBAC Authorization and Permission Dependency unit tests."""

import pytest

from enterprise_orchestrator.errors.exceptions import AuthorizationError
from enterprise_orchestrator.security.authorizers import (
    require_authenticated_user,
    require_permission,
    require_role,
)
from enterprise_orchestrator.security.models import AuthenticatedIdentity, Permission, Role


class TestRBACAuthorization:
    """Test RBAC authorization logic and permission enforcement."""

    @pytest.fixture
    def admin_user(self):
        return AuthenticatedIdentity(
            subject_id="admin-1",
            username="admin",
            roles=[Role.ADMIN],
            auth_method="api_key",
        )

    @pytest.fixture
    def operator_user(self):
        return AuthenticatedIdentity(
            subject_id="operator-1",
            username="operator",
            roles=[Role.OPERATOR],
            auth_method="api_key",
        )

    @pytest.fixture
    def reviewer_user(self):
        return AuthenticatedIdentity(
            subject_id="reviewer-1",
            username="reviewer",
            roles=[Role.REVIEWER],
            auth_method="api_key",
        )

    @pytest.fixture
    def viewer_user(self):
        return AuthenticatedIdentity(
            subject_id="viewer-1",
            username="viewer",
            roles=[Role.VIEWER],
            auth_method="api_key",
        )

    @pytest.mark.asyncio
    async def test_require_permission_admin_has_all(self, admin_user):
        for perm in Permission:
            dep = require_permission(perm)
            result = await dep(admin_user)
            assert result == admin_user

    @pytest.mark.asyncio
    async def test_require_permission_operator(self, operator_user):
        dep_create = require_permission(Permission.RUNS_CREATE)
        assert await dep_create(operator_user) == operator_user

        dep_approve = require_permission(Permission.HITL_APPROVE)
        assert await dep_approve(operator_user) == operator_user

        # Operator lacks RUNS_DELETE and AUDIT_READ
        dep_delete = require_permission(Permission.RUNS_DELETE)
        with pytest.raises(AuthorizationError) as exc_info:
            await dep_delete(operator_user)
        assert exc_info.value.code == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_require_permission_reviewer(self, reviewer_user):
        dep_read = require_permission(Permission.HITL_READ)
        assert await dep_read(reviewer_user) == reviewer_user

        dep_approve = require_permission(Permission.HITL_APPROVE)
        assert await dep_approve(reviewer_user) == reviewer_user

        dep_reject = require_permission(Permission.HITL_REJECT)
        assert await dep_reject(reviewer_user) == reviewer_user

        dep_modify = require_permission(Permission.HITL_MODIFY)
        assert await dep_modify(reviewer_user) == reviewer_user

        # Reviewer lacks RUNS_CREATE
        dep_create = require_permission(Permission.RUNS_CREATE)
        with pytest.raises(AuthorizationError):
            await dep_create(reviewer_user)

    @pytest.mark.asyncio
    async def test_require_permission_viewer_read_only(self, viewer_user):
        dep_read = require_permission(Permission.RUNS_READ)
        assert await dep_read(viewer_user) == viewer_user

        dep_metrics = require_permission(Permission.METRICS_READ)
        assert await dep_metrics(viewer_user) == viewer_user

        # Viewer cannot create runs or approve reviews
        with pytest.raises(AuthorizationError):
            await require_permission(Permission.RUNS_CREATE)(viewer_user)

        with pytest.raises(AuthorizationError):
            await require_permission(Permission.HITL_APPROVE)(viewer_user)

        with pytest.raises(AuthorizationError):
            await require_permission(Permission.HITL_REJECT)(viewer_user)

    @pytest.mark.asyncio
    async def test_require_role(self, operator_user, viewer_user):
        dep_op = require_role(Role.OPERATOR)
        assert await dep_op(operator_user) == operator_user

        with pytest.raises(AuthorizationError) as exc_info:
            await dep_op(viewer_user)
        assert "Principal lacks required role" in str(exc_info.value)
