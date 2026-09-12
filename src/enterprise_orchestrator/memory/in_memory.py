"""In-memory state and checkpoint store for testing and local execution."""

from typing import Dict, List, Optional

from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.memory.base import BaseStateStore, Checkpoint


class InMemoryStateStore(BaseStateStore):
    """Thread-safe in-memory store for states and step checkpoints."""

    def __init__(self) -> None:
        self._states: Dict[str, OrchestrationState] = {}
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._run_checkpoints: Dict[str, List[str]] = {}

    async def save_state(self, state: OrchestrationState) -> None:
        """Persist latest state using model copy."""
        self._states[state.run_id] = state.model_copy(deep=True)

    async def get_state(self, run_id: str) -> Optional[OrchestrationState]:
        """Fetch latest state snapshot."""
        state = self._states.get(run_id)
        if state:
            return state.model_copy(deep=True)
        return None

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Store immutable checkpoint."""
        cp_copy = checkpoint.model_copy(deep=True)
        self._checkpoints[cp_copy.checkpoint_id] = cp_copy
        if cp_copy.run_id not in self._run_checkpoints:
            self._run_checkpoints[cp_copy.run_id] = []
        self._run_checkpoints[cp_copy.run_id].append(cp_copy.checkpoint_id)

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Fetch checkpoint by ID."""
        cp = self._checkpoints.get(checkpoint_id)
        if cp:
            return cp.model_copy(deep=True)
        return None

    async def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        """List checkpoints for a given run ID."""
        cp_ids = self._run_checkpoints.get(run_id, [])
        return [self._checkpoints[cid].model_copy(deep=True) for cid in cp_ids if cid in self._checkpoints]

    async def delete_run(self, run_id: str) -> bool:
        """Purge state and associated checkpoints."""
        existed = False
        if run_id in self._states:
            del self._states[run_id]
            existed = True
        if run_id in self._run_checkpoints:
            cp_ids = self._run_checkpoints.pop(run_id)
            for cid in cp_ids:
                self._checkpoints.pop(cid, None)
            existed = True
        return existed
