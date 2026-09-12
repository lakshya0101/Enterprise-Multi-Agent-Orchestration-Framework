"""Tests for state persistence and checkpoint store contracts."""

import pytest

from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.memory.base import Checkpoint
from enterprise_orchestrator.memory.in_memory import InMemoryStateStore


@pytest.mark.asyncio
async def test_in_memory_state_store_save_and_retrieve():
    """Verify state persistence in the in-memory store."""
    store = InMemoryStateStore()
    state = OrchestrationState.create_initial(request="Persist state test")
    state.add_agent_output("agent_x", {"foo": "bar"})

    # Save state
    await store.save_state(state)

    # Fetch state
    retrieved = await store.get_state(state.run_id)
    assert retrieved is not None
    assert retrieved.run_id == state.run_id
    assert retrieved.agent_outputs == {"agent_x": {"foo": "bar"}}

    # Verify deep copy isolation
    state.add_agent_output("agent_x", {"foo": "mutated"})
    assert (await store.get_state(state.run_id)).agent_outputs == {"agent_x": {"foo": "bar"}}


@pytest.mark.asyncio
async def test_in_memory_checkpoints_lifecycle():
    """Verify saving, retrieving, and listing step checkpoints."""
    store = InMemoryStateStore()
    state = OrchestrationState.create_initial(request="Checkpoint workflow")

    cp1 = Checkpoint(run_id=state.run_id, step=1, state=state)
    await store.save_checkpoint(cp1)

    state.increment_step("planner")
    cp2 = Checkpoint(run_id=state.run_id, step=2, state=state)
    await store.save_checkpoint(cp2)

    # Fetch individual checkpoint
    fetched_cp1 = await store.get_checkpoint(cp1.checkpoint_id)
    assert fetched_cp1 is not None
    assert fetched_cp1.step == 1

    # List checkpoints
    checkpoints = await store.list_checkpoints(state.run_id)
    assert len(checkpoints) == 2
    assert checkpoints[0].step == 1
    assert checkpoints[1].step == 2

    # Delete run
    deleted = await store.delete_run(state.run_id)
    assert deleted is True
    assert await store.get_state(state.run_id) is None
    assert len(await store.list_checkpoints(state.run_id)) == 0
