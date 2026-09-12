"""Base validator interface contract."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from enterprise_orchestrator.validation.models import ValidationResult


class BaseValidator(ABC):
    """Abstract base class for all validator and critic implementations."""

    def __init__(self, name: str = "base_validator", description: str = "Base Validator") -> None:
        self.name = name
        self.description = description

    @abstractmethod
    async def validate(
        self,
        target_output: Any,
        context: Optional[Dict[str, Any]] = None,
        criteria: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Evaluate target output against validation criteria and return a structured verdict."""
        pass
