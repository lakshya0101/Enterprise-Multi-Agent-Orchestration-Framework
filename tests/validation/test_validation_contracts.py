"""Tests for validation and critic contracts."""

from typing import Any, Dict, Optional

import pytest

from enterprise_orchestrator.core.types import ValidationStatus
from enterprise_orchestrator.validation.base import BaseValidator
from enterprise_orchestrator.validation.models import ValidationResult


class SimpleThresholdValidator(BaseValidator):
    """Fake validator that checks if numeric scores meet a threshold."""

    def __init__(self, min_threshold: float = 0.8) -> None:
        super().__init__(name="threshold_validator", description="Validates output quality score")
        self.min_threshold = min_threshold

    async def validate(
        self,
        target_output: Any,
        context: Optional[Dict[str, Any]] = None,
        criteria: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        score = target_output.get("quality_score", 0.0) if isinstance(target_output, dict) else 0.0
        if score >= self.min_threshold:
            return ValidationResult(
                is_valid=True,
                status=ValidationStatus.VALID,
                score=score,
                validator_name=self.name,
            )
        elif score >= 0.5:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.NEEDS_RETRY,
                score=score,
                issues=["Quality score below acceptable threshold"],
                feedback="Please expand on key points to increase quality score.",
                needs_retry=True,
                validator_name=self.name,
            )
        else:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.NEEDS_HUMAN_REVIEW,
                score=score,
                issues=["Quality score severely degraded"],
                feedback="Uncertain output requires human supervisor review.",
                needs_human_review=True,
                validator_name=self.name,
            )


@pytest.mark.asyncio
async def test_validation_outcomes():
    """Verify different validation paths: Valid, Retry, Human Review."""
    validator = SimpleThresholdValidator(min_threshold=0.8)

    # Valid output
    res_valid = await validator.validate({"quality_score": 0.9})
    assert res_valid.is_valid is True
    assert res_valid.status == ValidationStatus.VALID
    assert res_valid.needs_retry is False

    # Needs retry
    res_retry = await validator.validate({"quality_score": 0.6})
    assert res_retry.is_valid is False
    assert res_retry.status == ValidationStatus.NEEDS_RETRY
    assert res_retry.needs_retry is True
    assert "expand" in (res_retry.feedback or "")

    # Needs human review
    res_human = await validator.validate({"quality_score": 0.3})
    assert res_human.is_valid is False
    assert res_human.status == ValidationStatus.NEEDS_HUMAN_REVIEW
    assert res_human.needs_human_review is True
