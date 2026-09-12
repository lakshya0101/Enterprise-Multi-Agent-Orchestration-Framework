"""PostgreSQL persistent state and checkpoint store implementing BaseStateStore."""

import json
from typing import Any, List, Optional

from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.errors.exceptions import ConfigurationError, OrchestrationError
from enterprise_orchestrator.memory.base import BaseStateStore, Checkpoint

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR(64) PRIMARY KEY,
    status VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    state_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    checkpoint_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id_step ON checkpoints(run_id, step ASC);
"""


class PostgreSQLStateStore(BaseStateStore):
    """Production PostgreSQL persistence engine for OrchestrationState and execution checkpoints."""

    def __init__(
        self,
        dsn: Optional[str] = None,
        pool: Optional[Any] = None,
        auto_init_schema: bool = True,
    ) -> None:
        self.dsn = dsn or "postgresql://postgres:postgres@localhost:5432/orchestrator"
        self._pool = pool
        self.auto_init_schema = auto_init_schema
        self._schema_initialized = False

    async def _get_pool(self) -> Any:
        """Lazily acquire or create asyncpg connection pool."""
        if self._pool is not None:
            return self._pool

        try:
            import asyncpg
        except ImportError as e:
            raise ConfigurationError(
                message="The 'asyncpg' package is required for PostgreSQL persistence. "
                "Install it with 'pip install asyncpg' or 'pip install enterprise-multi-agent-orchestration-framework[postgres]'.",
                code="MISSING_POSTGRES_DEPENDENCY",
            ) from e

        try:
            self._pool = await asyncpg.create_pool(dsn=self.dsn, min_size=1, max_size=10)
            return self._pool
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to connect to PostgreSQL database: {e}",
                code="POSTGRES_CONNECTION_ERROR",
                retryable=True,
            ) from e

    async def init_schema(self) -> None:
        """Create database tables and indices if they do not exist."""
        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                await conn.execute(SCHEMA_SQL)
            self._schema_initialized = True
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to initialize PostgreSQL database schema: {e}",
                code="POSTGRES_SCHEMA_INIT_ERROR",
                retryable=True,
            ) from e

    async def _ensure_schema(self) -> None:
        if self.auto_init_schema and not self._schema_initialized:
            await self.init_schema()

    async def save_state(self, state: OrchestrationState) -> None:
        """Persist or update state snapshot using Pydantic JSON serialization."""
        await self._ensure_schema()
        pool = await self._get_pool()
        state_json = state.model_dump_json()

        query = """
        INSERT INTO runs (run_id, status, created_at, updated_at, state_json)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        ON CONFLICT(run_id) DO UPDATE SET
            status = EXCLUDED.status,
            updated_at = EXCLUDED.updated_at,
            state_json = EXCLUDED.state_json;
        """
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    query,
                    state.run_id,
                    state.metadata.status.value,
                    state.metadata.created_at,
                    state.metadata.updated_at,
                    state_json,
                )
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to persist orchestration state in PostgreSQL for run '{state.run_id}': {e}",
                code="POSTGRES_SAVE_STATE_ERROR",
                retryable=True,
            ) from e

    async def get_state(self, run_id: str) -> Optional[OrchestrationState]:
        """Retrieve and deserialize latest state snapshot for a given run ID."""
        await self._ensure_schema()
        pool = await self._get_pool()
        query = "SELECT state_json FROM runs WHERE run_id = $1;"

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, run_id)
                if not row:
                    return None
                val = row["state_json"]
                if isinstance(val, str):
                    return OrchestrationState.model_validate_json(val)
                elif isinstance(val, dict):
                    return OrchestrationState.model_validate(val)
                else:
                    return OrchestrationState.model_validate_json(json.dumps(val))
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to retrieve state from PostgreSQL for run '{run_id}': {e}",
                code="POSTGRES_GET_STATE_ERROR",
                retryable=False,
            ) from e

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Persist an immutable execution step checkpoint."""
        await self._ensure_schema()
        pool = await self._get_pool()
        checkpoint_json = checkpoint.model_dump_json()

        query = """
        INSERT INTO checkpoints (checkpoint_id, run_id, step, created_at, checkpoint_json)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        ON CONFLICT(checkpoint_id) DO UPDATE SET
            checkpoint_json = EXCLUDED.checkpoint_json;
        """
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    query,
                    checkpoint.checkpoint_id,
                    checkpoint.run_id,
                    checkpoint.step,
                    checkpoint.timestamp,
                    checkpoint_json,
                )
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to save checkpoint in PostgreSQL for '{checkpoint.checkpoint_id}': {e}",
                code="POSTGRES_SAVE_CHECKPOINT_ERROR",
                retryable=True,
            ) from e

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Fetch a specific execution checkpoint by ID."""
        await self._ensure_schema()
        pool = await self._get_pool()
        query = "SELECT checkpoint_json FROM checkpoints WHERE checkpoint_id = $1;"

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, checkpoint_id)
                if not row:
                    return None
                val = row["checkpoint_json"]
                if isinstance(val, str):
                    return Checkpoint.model_validate_json(val)
                elif isinstance(val, dict):
                    return Checkpoint.model_validate(val)
                else:
                    return Checkpoint.model_validate_json(json.dumps(val))
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to get checkpoint from PostgreSQL for '{checkpoint_id}': {e}",
                code="POSTGRES_GET_CHECKPOINT_ERROR",
                retryable=False,
            ) from e

    async def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        """List all historical step checkpoints for a run sorted by step ascending."""
        await self._ensure_schema()
        pool = await self._get_pool()
        query = "SELECT checkpoint_json FROM checkpoints WHERE run_id = $1 ORDER BY step ASC, created_at ASC;"

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, run_id)
                checkpoints: List[Checkpoint] = []
                for r in rows:
                    val = r["checkpoint_json"]
                    if isinstance(val, str):
                        checkpoints.append(Checkpoint.model_validate_json(val))
                    elif isinstance(val, dict):
                        checkpoints.append(Checkpoint.model_validate(val))
                    else:
                        checkpoints.append(Checkpoint.model_validate_json(json.dumps(val)))
                return checkpoints
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to list checkpoints from PostgreSQL for run '{run_id}': {e}",
                code="POSTGRES_LIST_CHECKPOINTS_ERROR",
                retryable=False,
            ) from e

    async def delete_run(self, run_id: str) -> bool:
        """Purge state and associated checkpoints for a given run ID."""
        await self._ensure_schema()
        pool = await self._get_pool()
        query = "DELETE FROM runs WHERE run_id = $1;"

        try:
            async with pool.acquire() as conn:
                status_result = await conn.execute(query, run_id)
                # asyncpg returns strings like 'DELETE 1'
                if isinstance(status_result, str) and "DELETE" in status_result:
                    try:
                        count = int(status_result.split()[-1])
                        return count > 0
                    except (ValueError, IndexError):
                        return True
                return True
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to delete run from PostgreSQL for '{run_id}': {e}",
                code="POSTGRES_DELETE_RUN_ERROR",
                retryable=False,
            ) from e

    async def close(self) -> None:
        """Close connection pool if owned."""
        if self._pool is not None and hasattr(self._pool, "close"):
            await self._pool.close()
