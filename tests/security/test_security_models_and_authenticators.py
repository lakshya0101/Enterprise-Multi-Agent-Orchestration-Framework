"""Security domain models and authenticator unit tests."""

import hashlib
import time
from unittest.mock import MagicMock
import pytest
from starlette.requests import Request

from enterprise_orchestrator.errors.exceptions import (
    AuthenticationError,
    ConfigurationError,
)
from enterprise_orchestrator.security.authenticators import (
    BaseAuthenticator,
    BearerTokenAuthenticator,
    CompositeAuthenticator,
    DevelopmentBypassAuthenticator,
    HashedAPIKeyAuthenticator,
    TestAuthenticator,
)
from enterprise_orchestrator.security.models import (
    AuthenticatedIdentity,
    Permission,
    Role,
    ROLE_PERMISSIONS,
)


def _make_dummy_request(headers: dict[str, str] = None, client_host: str = "127.0.0.1") -> Request:
    raw_headers = []
    if headers:
        for k, v in headers.items():
            raw_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": raw_headers,
        "client": (client_host, 12345),
    }
    return Request(scope)


class TestSecurityModels:
    """Test AuthenticatedIdentity, Role, and Permission behavior."""

    def test_role_permissions_matrix(self):
        """Verify role hierarchy and permissions mapping."""
        assert Permission.RUNS_CREATE in ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.RUNS_DELETE in ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.AUDIT_READ in ROLE_PERMISSIONS[Role.ADMIN]

        assert Permission.RUNS_CREATE in ROLE_PERMISSIONS[Role.OPERATOR]
        assert Permission.RUNS_DELETE not in ROLE_PERMISSIONS[Role.OPERATOR]
        assert Permission.HITL_APPROVE in ROLE_PERMISSIONS[Role.OPERATOR]

        assert Permission.HITL_APPROVE in ROLE_PERMISSIONS[Role.REVIEWER]
        assert Permission.HITL_REJECT in ROLE_PERMISSIONS[Role.REVIEWER]
        assert Permission.HITL_MODIFY in ROLE_PERMISSIONS[Role.REVIEWER]
        assert Permission.RUNS_CREATE not in ROLE_PERMISSIONS[Role.REVIEWER]

        assert Permission.RUNS_READ in ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.METRICS_READ in ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.HITL_APPROVE not in ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.HITL_REJECT not in ROLE_PERMISSIONS[Role.VIEWER]

    def test_authenticated_identity_permissions_resolution(self):
        """Verify identity permissions caching and resolution from multiple roles."""
        identity = AuthenticatedIdentity(
            subject_id="user_123",
            username="analyst_alice",
            roles=[Role.VIEWER, Role.REVIEWER],
            auth_method="api_key",
        )
        assert identity.is_authenticated is True
        assert identity.has_role(Role.VIEWER) is True
        assert identity.has_role(Role.REVIEWER) is True
        assert identity.has_role(Role.ADMIN) is False

        assert identity.has_permission(Permission.RUNS_READ) is True
        assert identity.has_permission(Permission.HITL_APPROVE) is True
        assert identity.has_permission(Permission.RUNS_CREATE) is False

    def test_identity_does_not_store_secrets(self):
        """Verify AuthenticatedIdentity model schema has no secret fields."""
        fields = AuthenticatedIdentity.model_fields.keys()
        forbidden = ["api_key", "password", "token", "secret", "bearer_token", "private_key"]
        for f in forbidden:
            assert f not in fields


class TestHashedAPIKeyAuthenticator:
    """Test constant-time SHA-256 API key authentication."""

    @pytest.fixture
    def test_keys(self):
        key1 = "secret-api-key-admin-12345"
        hash1 = hashlib.sha256(key1.encode("utf-8")).hexdigest()
        key2 = "secret-api-key-viewer-67890"
        hash2 = hashlib.sha256(key2.encode("utf-8")).hexdigest()
        return {
            "key1": key1,
            "hash1": hash1,
            "key2": key2,
            "hash2": hash2,
            "mapping": {
                hash1: {"subject_id": "admin-1", "username": "admin_user", "roles": ["ADMIN"]},
                hash2: {"subject_id": "viewer-1", "username": "viewer_user", "roles": ["VIEWER"]},
            },
        }

    @pytest.mark.asyncio
    async def test_valid_api_key_header_x_api_key(self, test_keys):
        auth = HashedAPIKeyAuthenticator(api_key_hashes=test_keys["mapping"])
        req = _make_dummy_request(headers={"X-API-Key": test_keys["key1"]})
        identity = await auth.authenticate(req)
        assert identity.subject_id == "admin-1"
        assert identity.username == "admin_user"
        assert Role.ADMIN in identity.roles
        assert identity.auth_method == "api_key"

    @pytest.mark.asyncio
    async def test_valid_api_key_bearer_format(self, test_keys):
        auth = HashedAPIKeyAuthenticator(api_key_hashes=test_keys["mapping"])
        req = _make_dummy_request(headers={"Authorization": f"ApiKey {test_keys['key2']}"})
        identity = await auth.authenticate(req)
        assert identity.subject_id == "viewer-1"
        assert Role.VIEWER in identity.roles

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_401(self, test_keys):
        auth = HashedAPIKeyAuthenticator(api_key_hashes=test_keys["mapping"])
        req = _make_dummy_request(headers={"X-API-Key": "invalid-wrong-key"})
        with pytest.raises(AuthenticationError) as exc_info:
            await auth.authenticate(req)
        assert exc_info.value.code == "AUTHENTICATION_FAILED"
        assert "Invalid" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_missing_api_key_raises_401(self, test_keys):
        auth = HashedAPIKeyAuthenticator(api_key_hashes=test_keys["mapping"])
        req = _make_dummy_request(headers={})
        with pytest.raises(AuthenticationError) as exc_info:
            await auth.authenticate(req)
        assert exc_info.value.code == "MISSING_CREDENTIALS"

    @pytest.mark.asyncio
    async def test_malformed_authorization_header(self, test_keys):
        auth = HashedAPIKeyAuthenticator(api_key_hashes=test_keys["mapping"])
        req = _make_dummy_request(headers={"Authorization": "MalformedHeaderWithNoSpace"})
        with pytest.raises(AuthenticationError):
            await auth.authenticate(req)


class TestBearerTokenAuthenticator:
    """Test cryptographic JWT bearer token authentication."""

    SECRET = "super-secret-jwt-key-for-test-suite-minimum-length-32-chars"

    @pytest.fixture
    def jwt_lib(self):
        try:
            import jwt
            return jwt
        except ImportError:
            pytest.skip("PyJWT not installed")

    @pytest.mark.asyncio
    async def test_valid_jwt_token(self, jwt_lib):
        auth = BearerTokenAuthenticator(secret_key=self.SECRET, algorithm="HS256")
        token = jwt_lib.encode(
            {
                "sub": "user_jwt_42",
                "username": "charlie",
                "roles": ["OPERATOR"],
                "exp": time.time() + 3600,
            },
            self.SECRET,
            algorithm="HS256",
        )
        req = _make_dummy_request(headers={"Authorization": f"Bearer {token}"})
        identity = await auth.authenticate(req)
        assert identity.subject_id == "user_jwt_42"
        assert identity.username == "charlie"
        assert Role.OPERATOR in identity.roles
        assert identity.auth_method == "bearer_token"

    @pytest.mark.asyncio
    async def test_expired_jwt_token_raises_401(self, jwt_lib):
        auth = BearerTokenAuthenticator(secret_key=self.SECRET, algorithm="HS256")
        token = jwt_lib.encode(
            {
                "sub": "user_expired",
                "username": "expired_user",
                "roles": ["VIEWER"],
                "exp": time.time() - 3600,  # Expired 1 hour ago
            },
            self.SECRET,
            algorithm="HS256",
        )
        req = _make_dummy_request(headers={"Authorization": f"Bearer {token}"})
        with pytest.raises(AuthenticationError) as exc_info:
            await auth.authenticate(req)
        assert exc_info.value.code == "TOKEN_EXPIRED"

    @pytest.mark.asyncio
    async def test_invalid_signature_raises_401(self, jwt_lib):
        auth = BearerTokenAuthenticator(secret_key=self.SECRET, algorithm="HS256")
        token = jwt_lib.encode(
            {
                "sub": "forged_user",
                "roles": ["ADMIN"],
                "exp": time.time() + 3600,
            },
            "wrong-secret-key-attacker",
            algorithm="HS256",
        )
        req = _make_dummy_request(headers={"Authorization": f"Bearer {token}"})
        with pytest.raises(AuthenticationError) as exc_info:
            await auth.authenticate(req)
        assert exc_info.value.code == "INVALID_TOKEN"

    @pytest.mark.asyncio
    async def test_unsupported_algorithm_rejected(self, jwt_lib):
        auth = BearerTokenAuthenticator(
            secret_key=self.SECRET,
            algorithm="HS256",
            allowed_algorithms=["HS256"],
        )
        # Attempt to forge using 'none' algorithm
        token = jwt_lib.encode(
            {"sub": "attacker", "roles": ["ADMIN"], "exp": time.time() + 3600},
            key="",
            algorithm="none",
        )
        req = _make_dummy_request(headers={"Authorization": f"Bearer {token}"})
        with pytest.raises(AuthenticationError):
            await auth.authenticate(req)

    @pytest.mark.asyncio
    async def test_missing_sub_claim_raises_401(self, jwt_lib):
        auth = BearerTokenAuthenticator(secret_key=self.SECRET, algorithm="HS256")
        token = jwt_lib.encode(
            {"username": "no_sub_claim", "roles": ["VIEWER"], "exp": time.time() + 3600},
            self.SECRET,
            algorithm="HS256",
        )
        req = _make_dummy_request(headers={"Authorization": f"Bearer {token}"})
        with pytest.raises(AuthenticationError) as exc_info:
            await auth.authenticate(req)
        assert exc_info.value.code == "INVALID_CLAIMS"

    def test_missing_secret_key_raises_config_error(self):
        with pytest.raises(ConfigurationError):
            BearerTokenAuthenticator(secret_key="")


class TestDevelopmentAndBypassAuthenticators:
    """Test DevelopmentBypassAuthenticator and fail-closed checks."""

    @pytest.mark.asyncio
    async def test_dev_bypass_in_development_mode(self):
        auth = DevelopmentBypassAuthenticator(environment="development")
        req = _make_dummy_request()
        identity = await auth.authenticate(req)
        assert identity.is_authenticated is True
        assert identity.subject_id == "dev-admin"
        assert Role.ADMIN in identity.roles
        assert identity.auth_method == "development_bypass"

    def test_dev_bypass_in_production_raises_config_error(self):
        with pytest.raises(ConfigurationError) as exc_info:
            DevelopmentBypassAuthenticator(environment="production")
        assert "production" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_test_authenticator_header_role_override(self):
        auth = TestAuthenticator()
        req = _make_dummy_request(
            headers={
                "X-Test-Subject": "custom-tester",
                "X-Test-Roles": "VIEWER,REVIEWER",
            }
        )
        identity = await auth.authenticate(req)
        assert identity.subject_id == "custom-tester"
        assert Role.VIEWER in identity.roles
        assert Role.REVIEWER in identity.roles
        assert Role.ADMIN not in identity.roles


class TestCompositeAuthenticator:
    """Test chaining multiple authenticators."""

    @pytest.mark.asyncio
    async def test_composite_falls_back_to_next_on_missing_credentials(self):
        key = "valid-key"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        api_auth = HashedAPIKeyAuthenticator(
            api_key_hashes={key_hash: {"subject_id": "key_user", "roles": ["OPERATOR"]}}
        )
        test_auth = TestAuthenticator()
        composite = CompositeAuthenticator([api_auth, test_auth])

        # 1. Provide API key -> Handled by API auth
        req1 = _make_dummy_request(headers={"X-API-Key": key})
        id1 = await composite.authenticate(req1)
        assert id1.subject_id == "key_user"

        # 2. Provide no API key -> Falls through to test_auth
        req2 = _make_dummy_request(headers={})
        id2 = await composite.authenticate(req2)
        assert id2.subject_id == "test-runner"
