"""Application service mediating between FastAPI transport endpoints and OrchestrationRuntime."""

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from enterprise_orchestrator.core.human import HumanDecision, HumanEscalationRequest
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.types import ExecutionStatus
from enterprise_orchestrator.errors.exceptions import OrchestrationError
from enterprise_orchestrator.memory.base import BaseStateStore, Checkpoint
from enterprise_orchestrator.orchestration.runtime import OrchestrationRuntime


class OrchestrationService:
    """Application service coordinating workflow execution, state persistence, and event streams."""

    CRITICAL_EVENTS = {
        "run_started",
        "human_review_required",
        "human_decision_submitted",
        "run_completed",
        "run_failed",
        "stream_closed",
    }

    def __init__(
        self,
        runtime: OrchestrationRuntime,
        state_store: Optional[BaseStateStore] = None,
        max_subscribers_per_run: int = 10,
    ) -> None:
        self.runtime = runtime
        self.state_store = state_store or runtime.state_store
        self.max_subscribers_per_run = max_subscribers_per_run
        # In-memory pub-sub channels per run_id: run_id -> List[asyncio.Queue]
        self._event_queues: Dict[str, List[asyncio.Queue]] = {}

    def _publish_event(self, run_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an event payload to active subscribers with critical event retention under backpressure."""
        event_payload = {
            "event": event_type,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        queues = list(self._event_queues.get(run_id, []))
        for q in queues:
            if q.full():
                if event_type in self.CRITICAL_EVENTS:
                    # Drop oldest item to ensure critical lifecycle event is delivered
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        q.put_nowait(event_payload)
                    except asyncio.QueueFull:
                        pass
                # Non-critical events are dropped safely when queue is full
            else:
                try:
                    q.put_nowait(event_payload)
                except asyncio.QueueFull:
                    pass

    async def create_and_run(
        self,
        request: str,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        custom_context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationState:
        """Start a new workflow run and execute it through the orchestration runtime."""
        initial_state = OrchestrationState.create_initial(
            request=request,
            correlation_id=correlation_id,
            session_id=session_id,
            custom_context=custom_context,
        )
        run_id = initial_state.run_id

        self._publish_event(run_id, "run_started", {"request": request, "status": "running"})

        try:
            # We invoke runtime directly with request parameters
            final_state = await self.runtime.run(
                request=request,
                correlation_id=correlation_id,
                session_id=session_id,
                custom_context=custom_context,
            )

            # Publish appropriate completion or pause event
            if final_state.metadata.status == ExecutionStatus.PAUSED_FOR_HUMAN:
                self._publish_event(
                    final_state.run_id,
                    "human_review_required",
                    {"reason": final_state.human_requests[-1].reason if final_state.human_requests else ""},
                )
            elif final_state.metadata.status == ExecutionStatus.COMPLETED:
                self._publish_event(
                    final_state.run_id,
                    "run_completed",
                    {"response": final_state.final_response},
                )
            elif final_state.metadata.status == ExecutionStatus.FAILED:
                self._publish_event(
                    final_state.run_id,
                    "run_failed",
                    {"errors": final_state.errors},
                )

            return final_state

        except Exception as e:
            self._publish_event(run_id, "run_failed", {"error": str(e)})
            raise

    async def get_state(self, run_id: str) -> Optional[OrchestrationState]:
        """Fetch latest persisted OrchestrationState."""
        return await self.state_store.get_state(run_id)

    async def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        """List historical step checkpoints."""
        return await self.state_store.list_checkpoints(run_id)

    async def get_pending_human_review(self, run_id: str) -> Optional[HumanEscalationRequest]:
        """Retrieve active pending human review request if present."""
        state = await self.get_state(run_id)
        if not state:
            return None
        if state.requires_human and state.human_requests:
            return state.human_requests[-1]
        return None

    async def submit_human_decision(
        self,
        run_id: str,
        decision: HumanDecision,
    ) -> OrchestrationState:
        """Apply a submitted human decision and resume paused workflow."""
        state = await self.get_state(run_id)
        if not state:
            raise OrchestrationError(
                message=f"Run '{run_id}' not found.",
                code="RUN_NOT_FOUND",
                retryable=False,
            )

        if not state.requires_human:
            raise OrchestrationError(
                message=f"Run '{run_id}' is not awaiting human review (status: {state.metadata.status.value}).",
                code="NO_HUMAN_REVIEW_PENDING",
                retryable=False,
            )

        if state.metadata.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED):
            raise OrchestrationError(
                message=f"Run '{run_id}' is in terminal status '{state.metadata.status.value}' and cannot be resumed.",
                code="RUN_TERMINAL",
                retryable=False,
            )

        self._publish_event(
            run_id,
            "human_decision_submitted",
            {"status": decision.status.value, "decision_note": decision.decision_note},
        )

        final_state = await self.runtime.resume_with_human_decision(state, decision)

        if final_state.metadata.status == ExecutionStatus.COMPLETED:
            self._publish_event(
                final_state.run_id,
                "run_completed",
                {"response": final_state.final_response},
            )
        elif final_state.metadata.status == ExecutionStatus.FAILED:
            self._publish_event(
                final_state.run_id,
                "run_failed",
                {"errors": final_state.errors},
            )
        elif final_state.metadata.status == ExecutionStatus.PAUSED_FOR_HUMAN:
            self._publish_event(
                final_state.run_id,
                "human_review_required",
                {"reason": final_state.human_requests[-1].reason if final_state.human_requests else ""},
            )

        return final_state

    async def subscribe_events(
        self,
        run_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to live Server-Sent Events stream for a specific run ID."""
        state = await self.get_state(run_id)
        if not state:
            yield {
                "event": "error",
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {"message": f"Run '{run_id}' not found."},
            }
            return

        # Check max concurrent subscribers limit
        current_subscribers = self._event_queues.get(run_id, [])
        if len(current_subscribers) >= self.max_subscribers_per_run:
            yield {
                "event": "error",
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {"message": f"Maximum concurrent SSE subscribers ({self.max_subscribers_per_run}) reached for run '{run_id}'."},
            }
            return

        # Queue subscription for live events with bounded capacity
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        if run_id not in self._event_queues:
            self._event_queues[run_id] = []
        self._event_queues[run_id].append(queue)

        try:
            # Emit initial status event
            yield {
                "event": "run_status",
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "status": state.metadata.status.value,
                    "current_step": state.current_step,
                    "requires_human": state.requires_human,
                },
            }

            # If already terminal, close stream
            if state.metadata.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED):
                yield {
                    "event": "stream_closed",
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {"message": "Run is in terminal state."},
                }
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=2.0)
                    yield event
                    if event.get("event") in ("run_completed", "run_failed"):
                        break
                except asyncio.TimeoutError:
                    # Check if state transitioned to terminal state
                    current_state = await self.get_state(run_id)
                    if current_state and current_state.metadata.status in (
                        ExecutionStatus.COMPLETED,
                        ExecutionStatus.FAILED,
                    ):
                        break
        finally:
            if run_id in self._event_queues and queue in self._event_queues[run_id]:
                self._event_queues[run_id].remove(queue)
                if not self._event_queues[run_id]:
                    del self._event_queues[run_id]
