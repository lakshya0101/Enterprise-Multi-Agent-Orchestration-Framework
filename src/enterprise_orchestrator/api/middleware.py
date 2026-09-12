"""Production-grade ASGI middlewares for size limits, security headers, correlation IDs, and public rate limiting."""

import re
import uuid
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from enterprise_orchestrator.api.schemas import ErrorResponse
from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.security.rate_limiter import get_rate_limiter

_CORRELATION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.:]{1,64}$")


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects incoming HTTP requests whose payload exceeds the configured maximum size."""

    def __init__(self, app: ASGIApp, max_body_bytes: int = 1048576) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.max_body_bytes:
                    error_resp = ErrorResponse(
                        error=f"Request payload size ({length} bytes) exceeds the maximum allowed limit of {self.max_body_bytes} bytes.",
                        code="PAYLOAD_TOO_LARGE",
                    )
                    return JSONResponse(
                        status_code=413,
                        content=error_resp.model_dump(),
                    )
            except ValueError:
                pass

        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attaches standard defensive HTTP security headers to all responses."""

    def __init__(self, app: ASGIApp, hsts_enabled: bool = True) -> None:
        super().__init__(app)
        self.hsts_enabled = hsts_enabled

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "0"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"

        if self.hsts_enabled:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Extracts, validates, or generates deterministic request and correlation IDs."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        raw_corr_id = request.headers.get("X-Correlation-ID") or request.headers.get("X-Request-ID")

        if raw_corr_id is not None:
            raw_corr_id = raw_corr_id.strip()
            if not _CORRELATION_ID_PATTERN.match(raw_corr_id):
                error_resp = ErrorResponse(
                    error="Invalid correlation ID format. Must match regex ^[a-zA-Z0-9_\\-\\.:]{1,64}$ without newlines or special characters.",
                    code="INVALID_CORRELATION_ID",
                )
                return JSONResponse(
                    status_code=422,
                    content=error_resp.model_dump(),
                )
            corr_id = raw_corr_id
        else:
            corr_id = f"req_{uuid.uuid4().hex}"

        request.state.correlation_id = corr_id
        request.state.request_id = corr_id

        response: Response = await call_next(request)
        response.headers["X-Correlation-ID"] = corr_id
        response.headers["X-Request-ID"] = corr_id
        return response


class PublicRateLimitMiddleware(BaseHTTPMiddleware):
    """Enforces IP-based rate limits on unauthenticated public endpoints (/health, /ready)."""

    def __init__(
        self,
        app: ASGIApp,
        settings: FrameworkSettings,
    ) -> None:
        super().__init__(app)
        self.settings = settings
        self.public_paths = {"/health", "/ready"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.settings.rate_limit_enabled or request.url.path not in self.public_paths:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown_ip"
        # If trusted proxy configured, resolve X-Forwarded-For safely
        if self.settings.trusted_proxies and client_ip in self.settings.trusted_proxies:
            x_forwarded = request.headers.get("X-Forwarded-For")
            if x_forwarded:
                client_ip = x_forwarded.split(",")[0].strip()

        limiter = get_rate_limiter()
        is_allowed, remaining, retry_after = await limiter.check_rate_limit(
            key=f"ip:{client_ip}",
            max_requests=self.settings.rate_limit_public_per_minute,
            window_seconds=60,
        )

        if not is_allowed:
            error_resp = ErrorResponse(
                error=f"Public endpoint rate limit exceeded. Try again in {retry_after} seconds.",
                code="RATE_LIMIT_EXCEEDED",
            )
            return JSONResponse(
                status_code=429,
                content=error_resp.model_dump(),
                headers={
                    "Retry-After": str(int(retry_after) or 1),
                    "X-RateLimit-Limit": str(self.settings.rate_limit_public_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(retry_after) or 1),
                },
            )

        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.settings.rate_limit_public_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
