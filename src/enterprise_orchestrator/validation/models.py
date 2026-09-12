"""Validation models and result contracts."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from enterprise_orchestrator.core.types import ValidationStatus


class ValidationResult(BaseModel):
    """Structured verdict produced by a validator or critic agent."""

    validation_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique ID for this validation outcome")
    is_valid: bool = Field(..., description="Binary determination of whether output passed validation")
    status: ValidationStatus = Field(default=ValidationStatus.VALID, description="Granular validation verdict")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence or quality score between 0.0 and 1.0")
    feedback: Optional[str] = Field(default=None, description="Actionable critique and guidance for correction")
    issues: List[str] = Field(default_factory=list, description="List of specific flaws, schema violations, or inconsistencies")
    needs_retry: bool = Field(default=False, description="Flag indicating if the failed step should be retried automatically")
    needs_human_review: bool = Field(default=False, description="Flag indicating if ambiguity requires human escalation")
    validator_name: str = Field(default="default_validator", description="Identifier of the validator agent or rule")
    evaluated_task_id: Optional[str] = Field(default=None, description="Task ID that was evaluated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional evaluation metrics or debug metadata")
