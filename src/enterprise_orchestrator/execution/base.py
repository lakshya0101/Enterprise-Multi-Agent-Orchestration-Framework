"""Abstract base execution backend contract."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import asyncio

from enterprise_orchestrator.core.human import HumanDecision
from enterprise_orchestrator.execution.models import ExecutionJob


class BaseExecutionBackend(ABC):
    """Abstract interface defining the execution boundary between services and runtime."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize worker loops, queues, and operational resources."""
        pass

    @abstractmethod
    async def submit_run(
        self,
        run_id: str,
        request: str,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        custom_context: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        subject_id: Optional[str] = None,
    ) -> Tuple[ExecutionJob, asyncio.Future]:
        """Enqueue a new workflow execution run.

        Returns:
            Tuple[ExecutionJob, asyncio.Future]: Job metadata and future resolving to final OrchestrationState.
        """
        pass

    @abstractmethod
    async def submit_resume(
        self,
        run_id: str,
        decision: HumanDecision,
    ) -> Tuple[ExecutionJob, asyncio.Future]:
        """Enqueue resumption of a HITL-paused run.

        Returns:
            Tuple[ExecutionJob, asyncio.Future]: Job metadata and future resolving to resumed OrchestrationState.
        """
        pass

    @abstractmethod
    async def cancel_run(self, run_id: str, reason: str = "User cancelled") -> bool:
        """Request cooperative cancellation of an active or queued workflow."""
        pass

    @abstractmethod
    async def get_job(self, run_id: str) -> Optional[ExecutionJob]:
        """Retrieve live execution job metadata if present in the backend."""
        pass

    @abstractmethod
    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Gracefully drain queues, await active workers, and terminate execution engine."""
        pass
