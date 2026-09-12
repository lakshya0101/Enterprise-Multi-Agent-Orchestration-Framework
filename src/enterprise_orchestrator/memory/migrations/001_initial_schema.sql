-- 001_initial_schema.sql
-- Enterprise Multi-Agent Orchestration Framework: PostgreSQL State Persistence

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
