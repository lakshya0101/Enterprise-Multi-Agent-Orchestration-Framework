"""Evaluation domain models for offline quality, fidelity, and reliability scoring."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ScorerResult(BaseModel):
    """Result produced by an individual evaluation scorer."""

    scorer_name: str
    score: float = Field(ge=0.0, le=1.0, description="Normalized score between 0.0 and 1.0")
    passed: bool
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class EvaluationResult(BaseModel):
    """Aggregated evaluation outcome for a workflow execution against benchmark criteria."""

    evaluation_id: str = Field(default_factory=lambda: str(uuid4()))
    benchmark_id: Optional[str] = None
    run_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    scorer_results: List[ScorerResult] = Field(default_factory=list)
    duration_ms: float = 0.0
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkScenario(BaseModel):
    """Declarative specification of a deterministic offline benchmark scenario."""

    benchmark_id: str
    name: str
    description: str
    request: str
    expected_status: str = "completed"  # completed, paused_for_human, failed
    expected_tasks_count: Optional[int] = None
    expected_hitl_decision: Optional[str] = None
    expected_ground_truth_doc_ids: Optional[List[str]] = None
    min_required_score: float = 0.8
    custom_context: Optional[Dict[str, Any]] = None
