"""Authentication providers validating API keys, Bearer tokens, and development bypasses."""

from abc import ABC, abstractmethod
import hashlib
import hmac
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import Request

from enterprise_orchestrator.errors.exceptions import AuthenticationError, ConfigurationError
from enterprise_orchestrator.security.models import (
    AuthenticatedIdentity,
    ROLE_PERMISSIONS,
    Permission,
    Role,
)


class BaseAuthenticator(ABC):
    """Abstract authentication provider contract."""

    @abstractmethod
    async def authenticate(self, request: Request) -> AuthenticatedIdentity:
        """Inspect request and return verified AuthenticatedIdentity.
        
        Raises:
            AuthenticationError: If credentials are missing, malformed, or invalid.
        """
        pass


class HashedAPIKeyAuthenticator(BaseAuthenticator):
    """Authenticates requests via cryptographic SHA-256 hashed API key comparisons."""

    def __init__(
        self,
        api_key_hashes: Dict[str, Union[str, Dict[str, Any]]],
        header_name: str = "X-API-Key",
    ) -> None:
        """Initialize with a mapping of SHA-256 hex digests to assigned role string or identity dict.
        
        Example:
            api_key_hashes = {
                "5d37d2f9a0d18b08709e...": "admin",
                "a849f7e8b2c19e7104b...": {"subject_id": "op-1", "roles": ["OPERATOR"]}
            }
        """
        self.api_key_hashes: Dict[str, Union[str, Dict[str, Any]]] = {
            k.lower().strip(): v for k, v in api_key_hashes.items()
        }
        self.header_name = header_name

    def _extract_raw_key(self, request: Request) -> Optional[str]:
        """Extract API key from header or Authorization: ApiKey scheme."""
        # 1. Check direct header (e.g. X-API-Key)
        header_val = request.headers.get(self.header_name)
        if header_val is not None:
            val = header_val.strip()
            if not val:
                raise AuthenticationError("Missing API Key header.", code="MISSING_CREDENTIALS")
            return val

        # 2. Check Authorization header with ApiKey scheme
        auth_header = request.headers.get("Authorization")
        if auth_header is not None:
            parts = auth_header.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0].lower() in ("apikey", "api-key"):
                val = parts[1].strip()
                if not val:
                    raise AuthenticationError("Missing API Key header.", code="MISSING_CREDENTIALS")
                return val
            elif len(parts) == 1 and parts[0].lower() in ("apikey", "api-key"):
                raise AuthenticationError("Missing API Key credential in Authorization header.", code="MISSING_CREDENTIALS")
            else:
                raise AuthenticationError("Malformed or unrecognized Authorization scheme for API key.", code="AUTHENTICATION_FAILED")

        return None

    async def authenticate(self, request: Request) -> AuthenticatedIdentity:
        raw_key = self._extract_raw_key(request)
        if not raw_key:
            raise AuthenticationError("Missing API Key credentials.", code="MISSING_CREDENTIALS")

        # Compute SHA-256 digest of incoming key
        incoming_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest().lower()

        # Constant-time comparison against configured hashes
        matched_val: Optional[Union[str, Dict[str, Any]]] = None
        matched_hash_prefix: Optional[str] = None

        for stored_hash, val in self.api_key_hashes.items():
            if hmac.compare_digest(incoming_hash, stored_hash):
                matched_val = val
                matched_hash_prefix = stored_hash[:8]
                break

        if matched_val is None:
            raise AuthenticationError("Invalid or unrecognized API Key.", code="AUTHENTICATION_FAILED")

        subject_id = f"apikey_{matched_hash_prefix}"
        username = f"apikey_user_{matched_hash_prefix}"
        roles: Set[Role] = set()

        if isinstance(matched_val, str):
            try:
                r_enum = Role(matched_val.lower().strip())
                roles.add(r_enum)
            except ValueError:
                roles.add(Role.VIEWER)
            username = f"apikey_{matched_val.lower().strip()}"
        elif isinstance(matched_val, dict):
            subject_id = matched_val.get("subject_id", subject_id)
            username = matched_val.get("username", username)
            raw_roles = matched_val.get("roles", ["VIEWER"])
            if isinstance(raw_roles, str):
                raw_roles = [raw_roles]
            for r in raw_roles:
                try:
                    roles.add(Role(str(r).lower().strip()))
                except ValueError:
                    continue

        if not roles:
            roles.add(Role.VIEWER)

        permissions: Set[Permission] = set()
        for r in roles:
            permissions.update(ROLE_PERMISSIONS.get(r, set()))

        return AuthenticatedIdentity(
            subject_id=subject_id,
            username=username,
            roles=roles,
            permissions=permissions,
            auth_method="api_key",
            is_authenticated=True,
        )


class BearerTokenAuthenticator(BaseAuthenticator):
    """Authenticates Bearer tokens using cryptographically signed JWTs."""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        allowed_algorithms: Optional[List[str]] = None,
    ) -> None:
        if not secret_key or not secret_key.strip():
            raise ConfigurationError("BearerTokenAuthenticator requires a non-empty JWT secret key.")
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.allowed_algorithms = allowed_algorithms or [algorithm]

    def _extract_token(self, request: Request) -> Optional[str]:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None
        parts = auth_header.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        elif len(parts) == 1 and parts[0].lower() == "bearer":
            raise AuthenticationError("Missing Bearer token credential in Authorization header.", code="MISSING_CREDENTIALS")
        return None

    async def authenticate(self, request: Request) -> AuthenticatedIdentity:
        token = self._extract_token(request)
        if not token:
            raise AuthenticationError("Missing Bearer token in Authorization header.", code="MISSING_CREDENTIALS")

        try:
            import jwt
        except ImportError as e:
            raise ConfigurationError(
                "PyJWT is required for Bearer token verification. Install with 'pip install pyjwt'."
            ) from e

        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=self.allowed_algorithms,
                options={"require": ["exp", "sub"]},
            )
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Bearer token has expired.", code="TOKEN_EXPIRED")
        except jwt.InvalidAlgorithmError:
            raise AuthenticationError("Unsupported or unallowed token signing algorithm.", code="INVALID_TOKEN")
        except jwt.MissingRequiredClaimError as e:
            raise AuthenticationError(f"Bearer token missing required claim: {e}", code="INVALID_CLAIMS")
        except jwt.PyJWTError as e:
            raise AuthenticationError(f"Invalid Bearer token: {e}", code="INVALID_TOKEN")

        subject = payload.get("sub")
        if not subject:
            raise AuthenticationError("Bearer token missing 'sub' claim.", code="INVALID_CLAIMS")

        subject = str(subject)
        username = str(payload.get("name") or payload.get("username") or subject)
        raw_roles = payload.get("roles", ["viewer"])
        if isinstance(raw_roles, str):
            raw_roles = [raw_roles]

        roles: Set[Role] = set()
        permissions: Set[Permission] = set()

        for r_str in raw_roles:
            try:
                r_enum = Role(str(r_str).lower())
                roles.add(r_enum)
                permissions.update(ROLE_PERMISSIONS.get(r_enum, set()))
            except ValueError:
                continue

        if not roles:
            roles.add(Role.VIEWER)
            permissions.update(ROLE_PERMISSIONS[Role.VIEWER])

        return AuthenticatedIdentity(
            subject_id=subject,
            username=username,
            roles=roles,
            permissions=permissions,
            auth_method="bearer_token",
            is_authenticated=True,
            metadata={"claims": {k: v for k, v in payload.items() if k not in ("sub", "roles")}},
        )


class DevelopmentBypassAuthenticator(BaseAuthenticator):
    """Development-only authenticator granting full access when auth is explicitly disabled."""

    def __init__(self, environment: str = "development") -> None:
        if environment == "production":
            raise ConfigurationError(
                "DevelopmentBypassAuthenticator cannot be used in a production environment."
            )
        self.environment = environment

    async def authenticate(self, request: Request) -> AuthenticatedIdentity:
        return AuthenticatedIdentity(
            subject_id="dev-admin",
            username="development_administrator",
            roles={Role.ADMIN},
            permissions=ROLE_PERMISSIONS[Role.ADMIN],
            auth_method="development_bypass",
            is_authenticated=True,
        )


class CompositeAuthenticator(BaseAuthenticator):
    """Sequentially evaluates multiple authenticators until one succeeds."""

    def __init__(self, authenticators: List[BaseAuthenticator]) -> None:
        self.authenticators = authenticators

    async def authenticate(self, request: Request) -> AuthenticatedIdentity:
        if not self.authenticators:
            raise AuthenticationError("No authenticators configured.", code="CONFIGURATION_ERROR")

        last_error = AuthenticationError("Missing authentication credentials.", code="MISSING_CREDENTIALS")

        for auth in self.authenticators:
            try:
                identity = await auth.authenticate(request)
                if identity and identity.is_authenticated:
                    return identity
            except AuthenticationError as e:
                # If credentials were provided but invalid, preserve the specific failure
                if e.code != "MISSING_CREDENTIALS":
                    last_error = e
                continue

        raise last_error


class TestAuthenticator(BaseAuthenticator):
    """Deterministic offline test authenticator mapping test keys/headers to identities."""

    TEST_KEYS: Dict[str, Role] = {
        "test-key-admin": Role.ADMIN,
        "test-key-operator": Role.OPERATOR,
        "test-key-reviewer": Role.REVIEWER,
        "test-key-viewer": Role.VIEWER,
    }

    async def authenticate(self, request: Request) -> AuthenticatedIdentity:
        # 1. Test header overrides
        test_subject = request.headers.get("X-Test-Subject")
        test_roles_raw = request.headers.get("X-Test-Roles")
        if test_subject or test_roles_raw:
            roles: Set[Role] = set()
            if test_roles_raw:
                for r in test_roles_raw.split(","):
                    try:
                        roles.add(Role(r.strip().lower()))
                    except ValueError:
                        pass
            if not roles:
                roles.add(Role.ADMIN)
            perms: Set[Permission] = set()
            for r in roles:
                perms.update(ROLE_PERMISSIONS.get(r, set()))
            return AuthenticatedIdentity(
                subject_id=test_subject or "test-runner",
                username=test_subject or "test_runner",
                roles=roles,
                permissions=perms,
                auth_method="test_authenticator",
                is_authenticated=True,
            )

        # 2. Test key mappings
        key = request.headers.get("X-API-Key")
        if not key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer test-token-"):
                key = auth_header.replace("Bearer test-token-", "test-key-")
            elif auth_header.startswith("ApiKey "):
                key = auth_header.split(" ", 1)[1]

        if key and key in self.TEST_KEYS:
            role = self.TEST_KEYS[key]
            return AuthenticatedIdentity(
                subject_id=f"test_{role.value}",
                username=f"test_{role.value}_user",
                roles={role},
                permissions=ROLE_PERMISSIONS[role],
                auth_method="test_authenticator",
                is_authenticated=True,
            )

        # 3. Default fallback test identity
        return AuthenticatedIdentity(
            subject_id="test-runner",
            username="test_runner",
            roles={Role.ADMIN},
            permissions=ROLE_PERMISSIONS[Role.ADMIN],
            auth_method="test_authenticator",
            is_authenticated=True,
        )
