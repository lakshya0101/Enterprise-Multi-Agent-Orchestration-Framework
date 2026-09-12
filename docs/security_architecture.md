# Security Architecture & RBAC Guide

## Overview

The Enterprise Multi-Agent Orchestration Framework employs a provider-agnostic, fail-closed security boundary wrapping the FastAPI transport layer. It enforces Role-Based Access Control (RBAC), constant-time API key verification, optional cryptographic Bearer JWT validation, and server-side Human-in-the-Loop (HITL) identity binding.

---

## 1. Environment Security Modes

| Environment | Default Auth State | Fail-Closed Behavior | Purpose |
| :--- | :--- | :--- | :--- |
| `production` | **Mandatory** (`API_AUTH_ENABLED=True`) | **Strict**: Application fails closed if auth disabled or keys unconfigured. | Hardened enterprise deployments. |
| `development`| **Optional** (Defaults to bypass if disabled) | **Warns**: Emits security bypass warnings on startup. | Local interactive prototyping. |
| `test` | **Flexible** (TestAuthenticator fallback) | **Permissive**: Offline mock test identities without credentials. | Fast deterministic offline testing. |

---

## 2. Authentication Providers

### Hashed API Key Authenticator (`HashedAPIKeyAuthenticator`)
- **Key Ingestion**: Accepts raw API keys via `X-API-Key` header or `Authorization: ApiKey <KEY>`.
- **Cryptographic Storage**: Only SHA-256 digests are stored in server configuration (`api_key_hashes`). Plaintext keys are **never** persisted or logged.
- **Verification**: Evaluated using constant-time `hmac.compare_digest` to prevent timing attacks.

#### API Key Hash Generation Procedure
To generate a SHA-256 digest for configuration:
```bash
python -c "import hashlib; print(hashlib.sha256('your-secure-raw-key'.encode()).hexdigest())"
```

### Bearer Token Authenticator (`BearerTokenAuthenticator`)
- **Cryptographic Signatures**: Validates HMAC-SHA256 JWTs (`HS256`, `HS384`, `HS512`).
- **Claim Enforcement**: Enforces `exp` (expiration) and `sub` (subject identifier) claims. Rejects unsigned or algorithm-altered tokens (`none`).
- **Graceful Dependency**: PyJWT is an optional dependency; if missing, falls back cleanly with actionable configuration error.

### Development Bypass & Test Authenticators
- **`DevelopmentBypassAuthenticator`**: Grants default `ADMIN` role in `development` mode only. Throws `ConfigurationError` if invoked in `production`.
- **`TestAuthenticator`**: Deterministic offline provider mapping test keys (`test-key-admin`, `test-key-operator`, etc.) or headers (`X-Test-Subject`, `X-Test-Roles`).

---

## 3. RBAC Roles & Permissions Matrix

### Roles
1. **`ADMIN`**: Full administrative governance across all operations, runs, HITL decisions, metrics, and audit sinks.
2. **`OPERATOR`**: Day-to-day workflow manager. Can create, inspect, stream, and manage HITL reviews.
3. **`REVIEWER`**: Dedicated human auditor for HITL escalations. Can read runs and approve, reject, or modify human review requests.
4. **`VIEWER`**: Read-only principal. Can inspect run statuses, logs, and telemetry metrics.

### Permissions Matrix
| Permission | Admin | Operator | Reviewer | Viewer | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `runs:create` | ✅ | ✅ | ❌ | ❌ | Initiate new workflow runs |
| `runs:read` | ✅ | ✅ | ✅ | ✅ | Inspect state, status, and checkpoints |
| `runs:delete` | ✅ | ❌ | ❌ | ❌ | Hard delete runs and state history |
| `runs:stream` | ✅ | ✅ | ✅ | ✅ | Connect to live Server-Sent Events |
| `hitl:read` | ✅ | ✅ | ✅ | ❌ | Inspect pending human review requests |
| `hitl:approve` | ✅ | ✅ | ✅ | ❌ | Approve pending human escalations |
| `hitl:reject` | ✅ | ✅ | ✅ | ❌ | Reject pending human escalations |
| `hitl:modify` | ✅ | ✅ | ✅ | ❌ | Modify parameters and approve |
| `metrics:read` | ✅ | ✅ | ✅ | ✅ | Read Prometheus-style telemetry metrics |
| `audit:read` | ✅ | ❌ | ❌ | ❌ | Read immutable audit event logs |

---

## 4. Protected API Endpoints

| Endpoint | Method | Required Permission | Public / Protected |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | *None* | **Public** |
| `/ready` | `GET` | *None* | **Public** |
| `/api/v1/runs` | `POST` | `runs:create` | **Protected** |
| `/api/v1/runs/{id}` | `GET` | `runs:read` | **Protected** |
| `/api/v1/runs/{id}/status` | `GET` | `runs:read` | **Protected** |
| `/api/v1/runs/{id}/checkpoints` | `GET` | `runs:read` | **Protected** |
| `/api/v1/runs/{id}/events` | `GET` | `runs:stream` | **Protected** |
| `/api/v1/runs/{id}/human-review` | `GET` | `hitl:read` | **Protected** |
| `/api/v1/runs/{id}/human-review` | `POST` | `hitl:approve` / `hitl:modify` / `hitl:reject` | **Protected** |
| `/api/v1/metrics` | `GET` | `metrics:read` | **Protected** |
| `/api/v1/runs/{id}/metrics` | `GET` | `metrics:read` | **Protected** |

---

## 5. Security Principles & Non-Repudiation

### 401 Unauthorized vs. 403 Forbidden
- **HTTP 401 (`AuthenticationError`)**: Missing, expired, malformed, or unrecognized credentials. Emits standard `WWW-Authenticate: Bearer, ApiKey` header.
- **HTTP 403 (`AuthorizationError`)**: Verified principal lacks the granular permission required for the specific endpoint or action.

### HITL Server-Side Identity Binding
To enforce non-repudiation:
- Client-supplied `reviewer_id` in request payloads is strictly ignored and overwritten.
- `decision.reviewer_id` is bound server-side to `identity.subject_id` resolved from the verified authentication token/key.
- The verified subject ID propagates directly to downstream `AuditEvent.actor` records.

### Credential Leakage Defense
- `FrameworkSettings.model_dump_safe()` automatically redacts `jwt_secret_key`, `postgres_password`, and LLM API keys.
- Error handlers sanitize stack traces and exclude credential digests or tokens from JSON responses.
- `AuthenticatedIdentity` models contain no raw keys, passwords, or bearer tokens.

---

## 6. Architecture Limitations & Deferred Work

1. **Multi-Tenancy**: Data isolation across distinct tenant organizations remains deferred for future infrastructure milestones.
2. **Cloud IdP Integrations**: OAuth2/OIDC discovery (e.g., Okta, Auth0, Keycloak) and external JWKS polling are intentionally excluded from the local offline core and deferred to cloud deployment packages.
3. **Session Revocation**: JWT bearer token invalidation relies on expiration timestamps (`exp`); distributed token blacklisting remains deferred.
