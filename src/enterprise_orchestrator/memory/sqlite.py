"""SQLite persistent state and checkpoint store implementing BaseStateStore."""

import os
import sqlite3
from contextlib import contextmanager
from typing import Generator, List, Optional

from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.errors.exceptions import OrchestrationError
from enterprise_orchestrator.memory.base import BaseStateStore, Checkpoint


class SQLiteStateStore(BaseStateStore):
    """Local SQLite state persistence engine for OrchestrationState and execution checkpoints."""

    def __init__(self, db_path: str = "data/orchestrator.db") -> None:
        self.db_path = db_path
        parent_dir = os.path.dirname(os.path.abspath(db_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a transactional SQLite connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema tables and indices if not present."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    checkpoint_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id_step
                ON checkpoints(run_id, step);
                """
            )

    async def save_state(self, state: OrchestrationState) -> None:
        """Persist or update state snapshot using Pydantic JSON serialization."""
        try:
            state_json = state.model_dump_json()
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO runs (run_id, status, created_at, updated_at, state_json)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        status = excluded.status,
                        updated_at = excluded.updated_at,
                        state_json = excluded.state_json;
                    """,
                    (
                        state.run_id,
                        state.metadata.status.value,
                        state.metadata.created_at.isoformat(),
                        state.metadata.updated_at.isoformat(),
                        state_json,
                    ),
                )
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to persist orchestration state for run '{state.run_id}': {e}",
                code="SQLITE_SAVE_STATE_ERROR",
                retryable=True,
            ) from e

    async def get_state(self, run_id: str) -> Optional[OrchestrationState]:
        """Retrieve and deserialize latest state snapshot for a given run ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT state_json FROM runs WHERE run_id = ?;",
                    (run_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return OrchestrationState.model_validate_json(row["state_json"])
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to retrieve state for run '{run_id}': {e}",
                code="SQLITE_GET_STATE_ERROR",
                retryable=False,
            ) from e

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Persist an immutable execution step checkpoint."""
        try:
            checkpoint_json = checkpoint.model_dump_json()
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO checkpoints (checkpoint_id, run_id, step, created_at, checkpoint_json)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(checkpoint_id) DO UPDATE SET
                        checkpoint_json = excluded.checkpoint_json;
                    """,
                    (
                        checkpoint.checkpoint_id,
                        checkpoint.run_id,
                        checkpoint.step,
                        checkpoint.timestamp.isoformat(),
                        checkpoint_json,
                    ),
                )
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to save checkpoint '{checkpoint.checkpoint_id}': {e}",
                code="SQLITE_SAVE_CHECKPOINT_ERROR",
                retryable=True,
            ) from e

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Fetch a specific execution checkpoint by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT checkpoint_json FROM checkpoints WHERE checkpoint_id = ?;",
                    (checkpoint_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return Checkpoint.model_validate_json(row["checkpoint_json"])
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to get checkpoint '{checkpoint_id}': {e}",
                code="SQLITE_GET_CHECKPOINT_ERROR",
                retryable=False,
            ) from e

    async def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        """List all historical step checkpoints for a run sorted by step ascending."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT checkpoint_json FROM checkpoints WHERE run_id = ? ORDER BY step ASC, created_at ASC;",
                    (run_id,),
                )
                rows = cursor.fetchall()
                return [Checkpoint.model_validate_json(r["checkpoint_json"]) for r in rows]
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to list checkpoints for run '{run_id}': {e}",
                code="SQLITE_LIST_CHECKPOINTS_ERROR",
                retryable=False,
            ) from e

    async def delete_run(self, run_id: str) -> bool:
        """Purge state and associated checkpoints for a given run ID."""
        try:
            with self._get_connection() as conn:
                # Delete checkpoints first (if foreign keys not enforced by schema)
                conn.execute("DELETE FROM checkpoints WHERE run_id = ?;", (run_id,))
                cursor = conn.execute("DELETE FROM runs WHERE run_id = ?;", (run_id,))
                return cursor.rowcount > 0
        except Exception as e:
            raise OrchestrationError(
                message=f"Failed to delete run '{run_id}': {e}",
                code="SQLITE_DELETE_RUN_ERROR",
                retryable=False,
            ) from e
