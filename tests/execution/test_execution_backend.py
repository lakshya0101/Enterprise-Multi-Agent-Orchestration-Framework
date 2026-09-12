"""Unit and integration tests for LocalAsyncExecutionBackend and InMemoryIdempotencyLedger."""

import asyncio
from datetime import datetime
import pytest

from enterprise_orchestrator.core.human import HumanDecision, HumanRequestStatus
from enterprise_orchestrator.core.metadata import ExecutionMetadata
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.types import ExecutionStatus
from enterprise_orchestrator.errors.exceptions import OrchestrationError
from enterprise_orchestrator.execution.idempotency import (
    IdempotencyConflictError,
    InMemoryIdempotencyLedger,
)
from enterprise_orchestrator.execution.local_async import LocalAsyncExecutionBackend
from enterprise_orchestrator.execution.models import JobStatus
from enterprise_orchestrator.memory.in_memory import InMemoryStateStore
from enterprise_orchestrator.observability.audit.in_memory import InMemoryAuditSink
from enterprise_orchestrator.observability.metrics.registry import MetricsRegistry
from enterprise_orchestrator.observability.tracing.in_memory import InMemoryTracer
from enterprise_orchestrator.orchestration.runtime import OrchestrationRuntime


from unittest.mock import AsyncMock, MagicMock
from enterprise_orchestrator.agents.planner import PlannedDecompositionSchema, PlannedTaskItem
from enterprise_orchestrator.agents.validator import CriticEvaluationSchema
from enterprise_orchestrator.core.types import TaskType, ValidationStatus
from enterprise_orchestrator.providers.models import LLMResponse
from enterprise_orchestrator.providers.router import LLMRouter


@pytest.fixture
def test_setup():
    store = InMemoryStateStore()
    metrics = MetricsRegistry()
    tracer = InMemoryTracer()
    audit_sink = InMemoryAuditSink()

    router = AsyncMock(spec=LLMRouter)
    mock_task = PlannedTaskItem(
        id="task_1",
        title="Mock retrieval task",
        task_type=TaskType.RETRIEVAL,
        input_data={"query": "test context"},
    )
    mock_plan = PlannedDecompositionSchema(
        rationale="Mock execution plan",
        tasks=[mock_task],
    )
    critic_verdict = CriticEvaluationSchema(
        is_valid=True,
        status=ValidationStatus.VALID,
        score=1.0,
        feedback="Verified valid output",
        needs_human_review=False,
    )
    mock_resp = LLMResponse(content="Mock response synthesis", model="mock", provider="mock")

    async def _gen_structured(request, schema, *args, **kwargs):
        if schema == PlannedDecompositionSchema:
            return mock_plan, mock_resp
        return critic_verdict, mock_resp

    router.generate_structured.side_effect = _gen_structured
    router.generate.return_value = mock_resp

    runtime = OrchestrationRuntime(
        router=router,
        state_store=store,
        metrics=metrics,
        tracer=tracer,
        audit_sink=audit_sink,
    )
    backend = LocalAsyncExecutionBackend(
        runtime=runtime,
        state_store=store,
        metrics=metrics,
        tracer=tracer,
        audit_sink=audit_sink,
        max_concurrency=2,
        max_queue_size=3,
    )
    return backend, store, runtime, metrics


class TestLocalAsyncExecutionBackend:
    """Test suite for process-local asynchronous worker engine."""

    @pytest.mark.asyncio
    async def test_worker_pool_execution_happy_path(self, test_setup):
        backend, store, runtime, metrics = test_setup
        await backend.start()

        run_id = "run-happy-1"
        job, future = await backend.submit_run(
            run_id=run_id,
            request="Test happy path request",
        )

        assert job.status == JobStatus.QUEUED
        final_state: OrchestrationState = await asyncio.wait_for(future, timeout=5.0)

        assert final_state.run_id == run_id
        assert final_state.metadata.status in (ExecutionStatus.COMPLETED, ExecutionStatus.PAUSED_FOR_HUMAN)
        assert job.status in (JobStatus.COMPLETED, JobStatus.WAITING_FOR_HUMAN)

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_concurrency_limiting_and_queueing(self, test_setup):
        backend, store, runtime, metrics = test_setup
        await backend.start()

        # Submit 3 runs (concurrency is 2, queue capacity is 3)
        jobs = []
        futures = []
        for i in range(3):
            j, f = await backend.submit_run(
                run_id=f"run-concurrent-{i}",
                request=f"Concurrent task {i}",
            )
            jobs.append(j)
            futures.append(f)

        results = await asyncio.gather(*futures)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, OrchestrationState)

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_queue_full_rejection(self, test_setup):
        backend, store, runtime, metrics = test_setup
        await backend.start()

        block_event = asyncio.Event()

        async def _blocking_run(*args, **kwargs):
            await block_event.wait()
            state = OrchestrationState.create_initial(request="unblocked")
            state.metadata.status = ExecutionStatus.COMPLETED
            return state

        runtime.run = AsyncMock(side_effect=_blocking_run)

        try:
            # Submit 2 runs to occupy the 2 workers
            _, f0 = await backend.submit_run(run_id="run-occ-1", request="Task 0")
            _, f1 = await backend.submit_run(run_id="run-occ-2", request="Task 1")
            # Yield to loop so workers dequeue and start executing _blocking_run
            await asyncio.sleep(0.02)

            # Now queue is empty; submit 3 items to fill capacity (max_queue_size=3)
            queued_futures = []
            for i in range(3):
                _, f = await backend.submit_run(
                    run_id=f"run-queue-{i}",
                    request=f"Queued Task {i}",
                )
                queued_futures.append(f)

            # 4th queued submission exceeds capacity -> raises EXECUTION_QUEUE_FULL
            with pytest.raises(OrchestrationError) as exc_info:
                await backend.submit_run(
                    run_id="run-queue-overflow",
                    request="Task overflow",
                )

            assert exc_info.value.code == "EXECUTION_QUEUE_FULL"

        finally:
            block_event.set()
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_process_local_idempotency_concurrent(self, test_setup):
        backend, store, runtime, metrics = test_setup
        await backend.start()

        idempotency_key = "idemp-key-100"
        subject_id = "user-alice"

        # Submit twice concurrently with same idempotency key
        job1, fut1 = await backend.submit_run(
            run_id="run-idemp-1",
            request="Compute report",
            idempotency_key=idempotency_key,
            subject_id=subject_id,
        )

        job2, fut2 = await backend.submit_run(
            run_id="run-idemp-2",
            request="Compute report",
            idempotency_key=idempotency_key,
            subject_id=subject_id,
        )

        res1 = await asyncio.wait_for(fut1, timeout=5.0)
        res2 = await asyncio.wait_for(fut2, timeout=5.0)

        assert res1.run_id == res2.run_id
        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_process_local_idempotency_conflict(self, test_setup):
        backend, store, runtime, metrics = test_setup
        await backend.start()

        idempotency_key = "idemp-key-200"
        subject_id = "user-alice"

        await backend.submit_run(
            run_id="run-orig",
            request="Original request",
            idempotency_key=idempotency_key,
            subject_id=subject_id,
        )

        # Reusing same key with conflicting payload raises IdempotencyConflictError
        with pytest.raises(IdempotencyConflictError):
            await backend.submit_run(
                run_id="run-conflict",
                request="Completely different request payload",
                idempotency_key=idempotency_key,
                subject_id=subject_id,
            )

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_logical_cancellation_queued_run(self, test_setup):
        backend, store, runtime, metrics = test_setup
        await backend.start()

        block_event = asyncio.Event()

        async def _blocking_run(*args, **kwargs):
            await block_event.wait()
            state = OrchestrationState.create_initial(request="unblocked")
            state.metadata.status = ExecutionStatus.COMPLETED
            return state

        runtime.run = AsyncMock(side_effect=_blocking_run)

        try:
            # Submit 2 runs to occupy the 2 workers
            _, f_occ1 = await backend.submit_run(run_id="run-occ-1", request="Task 1")
            _, f_occ2 = await backend.submit_run(run_id="run-occ-2", request="Task 2")
            await asyncio.sleep(0.02)

            # Submit 3rd run that will remain waiting in queue
            run_id = "run-cancel-queued"
            job, future = await backend.submit_run(
                run_id=run_id,
                request="Task to cancel in queue",
            )

            # Cancel 3rd run while still in queue
            cancelled = await backend.cancel_run(run_id, reason="Cancelled in queue")
            assert cancelled is True

            # Unblock running workers; worker will finish occupied task, dequeue 3rd run, and finalize as CANCELLED
            block_event.set()
            await asyncio.gather(f_occ1, f_occ2)

            res = await asyncio.wait_for(future, timeout=5.0)
            assert res.metadata.status == ExecutionStatus.CANCELLED
            assert job.status == JobStatus.CANCELLED

        finally:
            block_event.set()
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_cancellation_paused_hitl_run(self, test_setup):
        backend, store, runtime, metrics = test_setup
        await backend.start()

        run_id = "run-hitl-paused"
        state = OrchestrationState.create_initial(request="HITL test")
        state.run_id = run_id
        state.metadata.status = ExecutionStatus.PAUSED_FOR_HUMAN
        state.requires_human = True
        await store.save_state(state)

        # Cancel paused run
        cancelled = await backend.cancel_run(run_id, reason="User cancelled while review was pending")
        assert cancelled is True

        updated_state = await store.get_state(run_id)
        assert updated_state.metadata.status == ExecutionStatus.CANCELLED

        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_drains_queue(self, test_setup):
        backend, store, runtime, metrics = test_setup
        await backend.start()

        block_event = asyncio.Event()

        async def _blocking_run(*args, **kwargs):
            await block_event.wait()
            state = OrchestrationState.create_initial(request="unblocked")
            state.metadata.status = ExecutionStatus.COMPLETED
            return state

        runtime.run = AsyncMock(side_effect=_blocking_run)

        try:
            # Submit 2 runs to occupy workers
            _, f_occ1 = await backend.submit_run(run_id="run-occ-1", request="Task 1")
            _, f_occ2 = await backend.submit_run(run_id="run-occ-2", request="Task 2")
            await asyncio.sleep(0.02)

            # Submit 2 items that stay in queue
            j1, f1 = await backend.submit_run(run_id="run-shut-1", request="Task 1")
            j2, f2 = await backend.submit_run(run_id="run-shut-2", request="Task 2")

            # Initiate shutdown -> unstarted queued items are drained and marked CANCELLED
            block_event.set()
            await backend.shutdown(timeout_seconds=2.0)

            res1 = await f1
            res2 = await f2
            assert res1.metadata.status == ExecutionStatus.CANCELLED
            assert res2.metadata.status == ExecutionStatus.CANCELLED
        finally:
            block_event.set()
            await backend.shutdown()
