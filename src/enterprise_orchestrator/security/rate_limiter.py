"""In-memory sliding-window rate limiting engine and FastAPI route dependencies."""

from abc import ABC, abstractmethod
import asyncio
from collections import OrderedDict
import time
from typing import Callable, Dict, List, Optional, Tuple

from fastapi import Depends, HTTPException, Request, status

from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.security.models import AuthenticatedIdentity


class BaseRateLimiter(ABC):
    """Abstract interface for rate limiting providers."""

    @abstractmethod
    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int = 60,
    ) -> Tuple[bool, int, float]:
        """Check if request is allowed under rate limits.

        Returns:
            Tuple[bool, int, float]: (is_allowed, remaining_quota, retry_after_seconds)
        """
        pass


class InMemorySlidingWindowRateLimiter(BaseRateLimiter):
    """Thread-safe, process-local in-memory sliding window rate limiter."""

    def __init__(self, max_keys: int = 10000) -> None:
        self.max_keys = max_keys
        self._store: OrderedDict[str, List[float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int = 60,
    ) -> Tuple[bool, int, float]:
        now = time.time()
        window_start = now - window_seconds

        async with self._lock:
            # 1. Clean up stale keys if cache size exceeds max_keys
            if len(self._store) > self.max_keys:
                stale_cutoff = now - (window_seconds * 2)
                stale_keys = [k for k, timestamps in self._store.items() if not timestamps or timestamps[-1] < stale_cutoff]
                for k in stale_keys:
                    self._store.pop(k, None)

            # 2. Get or create history for this key
            timestamps = self._store.get(key, [])
            # Prune events outside current sliding window
            valid_timestamps = [t for t in timestamps if t > window_start]

            if len(valid_timestamps) >= max_requests:
                earliest_event = valid_timestamps[0]
                retry_after = max(0.1, round(window_seconds - (now - earliest_event), 2))
                self._store[key] = valid_timestamps
                self._store.move_to_end(key)
                return False, 0, retry_after

            # Allowed: record timestamp
            valid_timestamps.append(now)
            self._store[key] = valid_timestamps
            self._store.move_to_end(key)
            remaining = max(0, max_requests - len(valid_timestamps))
            return True, remaining, 0.0

    def reset(self) -> None:
        """Reset rate limiter store (primarily for test isolation)."""
        self._store.clear()


_GLOBAL_RATE_LIMITER = InMemorySlidingWindowRateLimiter()


def get_rate_limiter() -> BaseRateLimiter:
    """Provide process-global in-memory rate limiter instance."""
    return _GLOBAL_RATE_LIMITER


def require_rate_limit(
    max_requests: Optional[int] = None,
    window_seconds: int = 60,
) -> Callable:
    """FastAPI route dependency enforcing rate limits on authenticated principals.

    Execution Order:
        1. Resolves verified identity from Track B `get_current_user`.
        2. Evaluates sliding window quota for key `sub:{identity.subject_id}`.
        3. Raises HTTP 429 with standard headers if quota is exhausted.
    """

    from enterprise_orchestrator.api.dependencies import get_current_user, get_settings

    async def _rate_limit_dependency(
        identity: AuthenticatedIdentity = Depends(get_current_user),
        settings: FrameworkSettings = Depends(get_settings),
        rate_limiter: BaseRateLimiter = Depends(get_rate_limiter),
    ) -> AuthenticatedIdentity:
        if not settings.rate_limit_enabled:
            return identity

        limit = max_requests or settings.rate_limit_authenticated_per_minute
        key = f"sub:{identity.subject_id}"

        is_allowed, remaining, retry_after = await rate_limiter.check_rate_limit(
            key=key,
            max_requests=limit,
            window_seconds=window_seconds,
        )

        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                headers={
                    "Retry-After": str(int(retry_after) or 1),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(retry_after) or 1),
                },
            )

        return identity

    return _rate_limit_dependency
