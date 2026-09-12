"""Tests for Server-Sent Events (SSE) priority queueing, backpressure, and subscriber limits."""

import asyncio
import pytest

from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.memory.in_memory import InMemoryStateStore
from enterprise_orchestrator.observability.audit.in_memory import InMemoryAuditSink
from enterprise_orchestrator.observability.metrics.registry import MetricsRegistry
from enterprise_orchestrator.observability.tracing.in_memory import InMemoryTracer
from enterprise_orchestrator.orchestration.runtime import OrchestrationRuntime
from enterprise_orchestrator.services.orchestration_service import OrchestrationService


@pytest.fixture
def service():
    store = InMemoryStateStore()
    runtime = OrchestrationRuntime(
        state_store=store,
        metrics=MetricsRegistry(),
        tracer=InMemoryTracer(),
        audit_sink=InMemoryAuditSink(),
    )
    return OrchestrationService(runtime=runtime, max_subscribers_per_run=2)


class TestSSEHardening:
    """Test priority event preservation and subscriber limits."""

    @pytest.mark.asyncio
    async def test_critical_lifecycle_event_preserved_under_queue_saturation(self, service):
        state = OrchestrationState.create_initial(request="SSE test")
        run_id = state.run_id
        await service.state_store.save_state(state)

        # 1. Start subscription generator
        gen = service.subscribe_events(run_id)
        # Consume the initial 'run_status' event
        initial_event = await gen.__anext__()
        assert initial_event["event"] == "run_status"

        # 2. Get the subscriber's internal queue and fill it to capacity (100 items) with non-critical events
        queue = service._event_queues[run_id][0]
        for i in range(100):
            service._publish_event(run_id, "run_status", {"step": i})

        assert queue.full() is True

        # 3. Publish a critical lifecycle event ('run_completed')
        service._publish_event(run_id, "run_completed", {"response": "Success result"})

        # 4. Drain events from generator and verify 'run_completed' is present and received!
        received_events = []
        try:
            while True:
                ev = await asyncio.wait_for(gen.__anext__(), timeout=0.1)
                received_events.append(ev)
                if ev.get("event") == "run_completed":
                    break
        except (asyncio.TimeoutError, StopAsyncIteration):
            pass

        assert any(e.get("event") == "run_completed" for e in received_events)
        assert any(e.get("data", {}).get("response") == "Success result" for e in received_events)

    @pytest.mark.asyncio
    async def test_max_concurrent_subscribers_limit(self, service):
        state = OrchestrationState.create_initial(request="SSE limit test")
        run_id = state.run_id
        await service.state_store.save_state(state)

        # Connect 2 subscribers (max allowed is 2)
        gen1 = service.subscribe_events(run_id)
        await gen1.__anext__()

        gen2 = service.subscribe_events(run_id)
        await gen2.__anext__()

        # 3rd subscriber exceeds limit
        gen3 = service.subscribe_events(run_id)
        err_event = await gen3.__anext__()
        assert err_event["event"] == "error"
        assert "Maximum concurrent SSE subscribers" in err_event["data"]["message"]
