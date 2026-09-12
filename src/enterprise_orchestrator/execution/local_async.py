"""Process-local asynchronous execution engine using a fixed worker pool and bounded queue."""

import asyncio
from datetime import datetime, timezone
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid

from enterprise_orchestrator.core.human import HumanDecision
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.types import ExecutionStatus
from enterprise_orchestrator.errors.exceptions import OrchestrationError
from enterprise_orchestrator.execution.base import BaseExecutionBackend
from enterprise_orchestrator.execution.idempotency import InMemoryIdempotencyLedger
from enterprise_orchestrator.execution.models import ExecutionJob, JobStatus
from enterprise_orchestrator.memory.base import BaseStateStore
from enterprise_orchestrator.observability.audit.base import BaseAuditSink
from enterprise_orchestrator.observability.audit.models import AuditEvent, AuditEventType
from enterprise_orchestrator.observability.metrics.registry import MetricsRegistry
from enterprise_orchestrator.observability.tracing.base import BaseTracer
from enterprise_orchestrator.orchestration.runtime import OrchestrationRuntime

logger = logging.getLogger(__name__)


class LocalAsyncExecutionBackend(BaseExecutionBackend):
    """Process-local execution backend managing asynchronous workers and admission queues.

    Invariants:
        1. Fixed async worker pool of size max_concurrency.
        2. Bounded admission queue of capacity max_queue_size.
        3. Logical cancellation for queued runs without queue internal mutation.
        4. Immediate capacity release when workflows pause for human review (HITL).
        5. Process-local atomic idempotency deduplication.
        6. Zero modification to frozen LangGraph orchestration runtime.
    """

    def __init__(
        self,
        runtime: OrchestrationRuntime,
        state_store: Optional[BaseStateStore] = None,
        metrics: Optional[MetricsRegistry] = None,
        tracer: Optional[BaseTracer] = None,
        audit_sink: Optional[BaseAuditSink] = None,
        max_concurrency: int = 10,
        max_queue_size: int = 100,
        event_publisher: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.runtime = runtime
        self.state_store = state_store or runtime.state_store
        self.metrics = metrics or runtime.metrics
        self.tracer = tracer or runtime.tracer
        self.audit_sink = audit_sink or runtime.audit_sink
        self.max_concurrency = max_concurrency
        self.max_queue_size = max_queue_size
        self._event_publisher = event_publisher

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._workers: List[asyncio.Task] = []
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._jobs: Dict[str, ExecutionJob] = {}
        self._futures: Dict[str, asyncio.Future] = {}
        self._cancelled_runs: Set[str] = set()
        self._idempotency_ledger = InMemoryIdempotencyLedger()

        self._is_started = False
        self._is_shutting_down = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Initialize Phase 7 Metrics
        if self.metrics:
            self.metrics.gauge("execution_queue_depth", "Current number of queued execution jobs")
            self.metrics.gauge("execution_active_tasks", "Current number of actively executing worker tasks")
            self.metrics.counter("execution_rejections_total", "Total execution submissions rejected due to backpressure")
            self.metrics.counter("execution_cancellations_total", "Total execution jobs successfully cancelled")
            self.metrics.histogram(
                "execution_job_duration_seconds",
                "End-to-end execution duration of worker jobs",
            )

    def _publish_event(self, run_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Helper to invoke optional SSE event publisher."""
        if self._event_publisher:
            self._event_publisher(run_id, event_type, data)

    def _update_metrics(self) -> None:
        """Update live gauge metrics."""
        if self.metrics:
            self.metrics.gauge("execution_queue_depth").set(float(self._queue.qsize()))
            self.metrics.gauge("execution_active_tasks").set(float(len(self._active_tasks)))

    async def start(self) -> None:
        """Start long-running worker coroutines on active event loop."""
        current_loop = asyncio.get_running_loop()
        if (
            self._is_started
            and self._loop is current_loop
            and self._workers
            and all(not w.done() for w in self._workers)
        ):
            return

        self._loop = current_loop
        self._is_started = True
        self._is_shutting_down = False
        if getattr(self._queue, "_loop", None) is not None and self._queue._loop is not current_loop:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._workers = [
            asyncio.create_task(self._worker_loop(i), name=f"local-exec-worker-{i}")
            for i in range(self.max_concurrency)
        ]
        self._update_metrics()

    async def _worker_loop(self, worker_id: int) -> None:
        """Internal worker coroutine dequeuing work and executing runtime steps."""
        while True:
            try:
                work_item = await self._queue.get()
            except asyncio.CancelledError:
                break

            if work_item is None:
                # Sentinel termination signal
                self._queue.task_done()
                break

            run_id, task_type, args, kwargs, future, key_hash = work_item
            job = self._jobs.get(run_id)

            # 1. Check logical cancellation before starting execution
            if run_id in self._cancelled_runs or (job and job.is_cancelled):
                await self._finalize_cancelled(run_id, future, key_hash)
                self._queue.task_done()
                self._update_metrics()
                continue

            current_task = asyncio.current_task()
            self._active_tasks[run_id] = current_task

            if job:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now(timezone.utc)

            self._update_metrics()
            start_time = time.perf_counter()

            try:
                # 2. Execute via frozen OrchestrationRuntime
                if task_type == "run":
                    (request,) = args
                    state = await self.runtime.run(
                        request=request,
                        correlation_id=kwargs.get("correlation_id"),
                        session_id=kwargs.get("session_id"),
                        custom_context=kwargs.get("custom_context"),
                    )
                    if state.run_id != run_id:
                        orig_run_id = state.run_id
                        checkpoints = await self.state_store.list_checkpoints(orig_run_id)
                        for cp in checkpoints:
                            await self.state_store.save_checkpoint(cp.model_copy(update={"run_id": run_id}))
                        if self.tracer and hasattr(self.tracer, "_spans"):
                            for s in getattr(self.tracer, "_spans", []):
                                if s.attributes.get("run_id") == orig_run_id:
                                    s.attributes["run_id"] = run_id
                        state = state.model_copy(update={"run_id": run_id})
                        await self.state_store.save_state(state)
                elif task_type == "resume":
                    state, decision = args
                    state.metadata.status = ExecutionStatus.RUNNING
                    state.metadata.touch()
                    await self.state_store.save_state(state)

                    state = await self.runtime.resume_with_human_decision(state, decision)
                    if state.run_id != run_id:
                        orig_run_id = state.run_id
                        checkpoints = await self.state_store.list_checkpoints(orig_run_id)
                        for cp in checkpoints:
                            await self.state_store.save_checkpoint(cp.model_copy(update={"run_id": run_id}))
                        if self.tracer and hasattr(self.tracer, "_spans"):
                            for s in getattr(self.tracer, "_spans", []):
                                if s.attributes.get("run_id") == orig_run_id:
                                    s.attributes["run_id"] = run_id
                        state = state.model_copy(update={"run_id": run_id})
                        await self.state_store.save_state(state)
                else:
                    raise ValueError(f"Unknown task type: {task_type}")

                duration = time.perf_counter() - start_time
                if self.metrics:
                    self.metrics.histogram("execution_job_duration_seconds").observe(duration)

                # 3. Handle Workflow Result
                if state.metadata.status == ExecutionStatus.PAUSED_FOR_HUMAN:
                    if job:
                        job.status = JobStatus.WAITING_FOR_HUMAN
                        job.result = state
                    if not future.done():
                        future.set_result(state)
                    if key_hash:
                        await self._idempotency_ledger.complete(key_hash, state)

                    self._publish_event(
                        run_id,
                        "human_review_required",
                        {"reason": state.human_requests[-1].reason if state.human_requests else ""},
                    )
                else:
                    if job:
                        job.status = JobStatus.COMPLETED if state.metadata.status == ExecutionStatus.COMPLETED else JobStatus.FAILED
                        job.completed_at = datetime.now(timezone.utc)
                        job.result = state
                    if not future.done():
                        future.set_result(state)
                    if key_hash:
                        await self._idempotency_ledger.complete(key_hash, state)

                    if state.metadata.status == ExecutionStatus.COMPLETED:
                        self._publish_event(run_id, "run_completed", {"response": state.final_response})
                    else:
                        self._publish_event(run_id, "run_failed", {"errors": state.errors})

            except asyncio.CancelledError:
                await self._finalize_cancelled(run_id, future, key_hash)
            except Exception as exc:
                logger.exception(f"Unhandled error in worker {worker_id} for run '{run_id}': {exc}")
                if job:
                    job.status = JobStatus.FAILED
                    job.error = str(exc)
                    job.completed_at = datetime.now(timezone.utc)
                if not future.done():
                    future.set_exception(exc)
                if key_hash:
                    await self._idempotency_ledger.fail(key_hash, exc)
                self._publish_event(run_id, "run_failed", {"error": str(exc)})
            finally:
                self._active_tasks.pop(run_id, None)
                self._queue.task_done()
                self._update_metrics()

    async def _finalize_cancelled(self, run_id: str, future: asyncio.Future, key_hash: Optional[str] = None) -> None:
        """Safely transition state to CANCELLED and emit events/audit records."""
        job = self._jobs.get(run_id)
        if job:
            job.status = JobStatus.CANCELLED
            job.is_cancelled = True
            job.completed_at = datetime.now(timezone.utc)

        state = await self.state_store.get_state(run_id)
        if state:
            state.metadata.status = ExecutionStatus.CANCELLED
            state.metadata.touch()
            await self.state_store.save_state(state)
            if not future.done():
                future.set_result(state)
            if key_hash:
                await self._idempotency_ledger.complete(key_hash, state)
        else:
            if not future.done():
                future.set_exception(asyncio.CancelledError(f"Run '{run_id}' cancelled."))
            if key_hash:
                await self._idempotency_ledger.fail(key_hash, asyncio.CancelledError(f"Run '{run_id}' cancelled."))

        if self.metrics:
            self.metrics.counter("execution_cancellations_total").inc()

        if self.audit_sink:
            self.audit_sink.record(
                AuditEvent(
                    run_id=run_id,
                    event_type=AuditEventType.RUN_FAILED,
                    component="LocalAsyncExecutionBackend",
                    action="cancel_run",
                    outcome="CANCELLED",
                    metadata={"reason": job.cancellation_reason if job else "Cancelled"},
                )
            )

        self._publish_event(run_id, "run_cancelled", {"reason": job.cancellation_reason if job else "Cancelled"})

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
        """Submit a new workflow run for asynchronous worker execution."""
        if self._is_shutting_down:
            raise OrchestrationError(
                message="Execution backend is shutting down.",
                code="SERVICE_UNAVAILABLE",
                retryable=True,
            )
        await self.start()

        # 1. Process-local atomic idempotency handling
        key_hash: Optional[str] = None
        future: Optional[asyncio.Future] = None
        if idempotency_key and subject_id:
            key_hash = InMemoryIdempotencyLedger.compute_key_hash(subject_id, idempotency_key)
            request_hash = InMemoryIdempotencyLedger.compute_request_hash(request, custom_context)
            is_initiator, future, cached_result = await self._idempotency_ledger.acquire_or_join(
                key_hash=key_hash,
                request_hash=request_hash,
                run_id=run_id,
            )
            if not is_initiator:
                existing_job = self._jobs.get(run_id) or ExecutionJob(
                    job_id=str(uuid.uuid4()),
                    run_id=run_id,
                    request=request,
                    status=JobStatus.COMPLETED if cached_result else JobStatus.RUNNING,
                    result=cached_result,
                )
                if cached_result is not None and not future.done():
                    future.set_result(cached_result)
                return existing_job, future

        # 2. Backpressure admission check
        if self._queue.full():
            if self.metrics:
                self.metrics.counter("execution_rejections_total").inc()
            raise OrchestrationError(
                message="Execution queue is at maximum capacity. Please retry later.",
                code="EXECUTION_QUEUE_FULL",
                retryable=True,
            )

        # 3. Create domain initial state and persist as IDLE
        initial_state = OrchestrationState.create_initial(
            request=request,
            correlation_id=correlation_id,
            session_id=session_id,
            custom_context=custom_context,
        )
        # Synchronize run_id
        initial_state.run_id = run_id
        initial_state.metadata.status = ExecutionStatus.IDLE
        await self.state_store.save_state(initial_state)

        # 4. Create Execution Job & Future
        job = ExecutionJob(
            job_id=str(uuid.uuid4()),
            run_id=run_id,
            request=request,
            status=JobStatus.QUEUED,
            correlation_id=correlation_id,
            session_id=session_id,
            custom_context=custom_context,
            idempotency_key=idempotency_key,
        )
        self._jobs[run_id] = job
        if future is None:
            future = asyncio.get_running_loop().create_future()
        self._futures[run_id] = future

        # 5. Enqueue for workers
        self._queue.put_nowait((
            run_id,
            "run",
            (request,),
            {"correlation_id": correlation_id, "session_id": session_id, "custom_context": custom_context},
            future,
            key_hash,
        ))

        self._publish_event(run_id, "run_started", {"request": request, "status": "queued"})
        self._update_metrics()
        return job, future

    async def submit_resume(
        self,
        run_id: str,
        decision: HumanDecision,
    ) -> Tuple[ExecutionJob, asyncio.Future]:
        """Submit a resumption job for a HITL-paused run."""
        if self._is_shutting_down:
            raise OrchestrationError(
                message="Execution backend is shutting down.",
                code="SERVICE_UNAVAILABLE",
                retryable=True,
            )
        await self.start()

        state = await self.state_store.get_state(run_id)
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

        if state.metadata.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED):
            raise OrchestrationError(
                message=f"Run '{run_id}' is in terminal status '{state.metadata.status.value}' and cannot be resumed.",
                code="RUN_TERMINAL",
                retryable=False,
            )

        if self._queue.full():
            if self.metrics:
                self.metrics.counter("execution_rejections_total").inc()
            raise OrchestrationError(
                message="Execution queue is at maximum capacity. Please retry later.",
                code="EXECUTION_QUEUE_FULL",
                retryable=True,
            )

        job = self._jobs.get(run_id) or ExecutionJob(
            job_id=str(uuid.uuid4()),
            run_id=run_id,
            request=state.request,
            status=JobStatus.QUEUED,
        )
        job.status = JobStatus.QUEUED
        job.is_cancelled = False
        self._jobs[run_id] = job
        self._cancelled_runs.discard(run_id)

        future = asyncio.get_running_loop().create_future()
        self._futures[run_id] = future

        self._publish_event(
            run_id,
            "human_decision_submitted",
            {"status": decision.status.value, "decision_note": decision.decision_note},
        )

        self._queue.put_nowait((
            run_id,
            "resume",
            (state, decision),
            {},
            future,
            None,
        ))

        self._update_metrics()
        return job, future

    async def cancel_run(self, run_id: str, reason: str = "User cancelled") -> bool:
        """Cancel a queued, running, or paused workflow."""
        self._cancelled_runs.add(run_id)
        job = self._jobs.get(run_id)
        if job:
            job.is_cancelled = True
            job.cancellation_reason = reason

        # Case 1: Active running task in worker pool
        if run_id in self._active_tasks:
            task = self._active_tasks[run_id]
            task.cancel()
            return True

        # Case 2: In queue or paused for human review
        state = await self.state_store.get_state(run_id)
        if state and state.metadata.status in (
            ExecutionStatus.IDLE,
            ExecutionStatus.PAUSED_FOR_HUMAN,
            ExecutionStatus.RUNNING,
        ):
            state.metadata.status = ExecutionStatus.CANCELLED
            state.metadata.touch()
            await self.state_store.save_state(state)

            if run_id in self._futures and not self._futures[run_id].done():
                self._futures[run_id].set_result(state)

            if job:
                job.status = JobStatus.CANCELLED

            if self.metrics:
                self.metrics.counter("execution_cancellations_total").inc()

            self._publish_event(run_id, "run_cancelled", {"reason": reason})
            return True

        return False

    async def get_job(self, run_id: str) -> Optional[ExecutionJob]:
        """Fetch live operational job metadata."""
        return self._jobs.get(run_id)

    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Gracefully drain admission queue and await worker termination."""
        if not self._is_started or self._is_shutting_down:
            return
        self._is_shutting_down = True

        # 1. Drain and cancel unstarted queued items
        while not self._queue.empty():
            try:
                work_item = self._queue.get_nowait()
                if work_item:
                    run_id, _, _, _, future, key_hash = work_item
                    await self._finalize_cancelled(run_id, future, key_hash)
                self._queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break

        # 2. Await active executing tasks up to configured timeout
        start_wait = time.perf_counter()
        while self._active_tasks and (time.perf_counter() - start_wait < timeout_seconds):
            await asyncio.sleep(0.01)

        # 3. Cancel any remaining active tasks that exceeded timeout
        if self._active_tasks:
            for t in list(self._active_tasks.values()):
                if not t.done():
                    t.cancel()

        # 4. Terminate worker loops
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

        for w in self._workers:
            if not w.done():
                w.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._active_tasks.clear()
        self._is_started = False
        self._is_shutting_down = False
        self._update_metrics()
