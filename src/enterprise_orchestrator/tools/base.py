"""Base tool interface and metadata contracts."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from pydantic import BaseModel, Field

from enterprise_orchestrator.errors.exceptions import ValidationError


class ToolMetadata(BaseModel):
    """Metadata describing tool capabilities, constraints, and requirements."""

    name: str = Field(..., min_length=1, description="Unique programmatic tool identifier")
    description: str = Field(..., min_length=1, description="Clear description of tool purpose for LLM and agent planning")
    version: str = Field(default="0.1.0", description="Tool implementation version")
    timeout_seconds: float = Field(default=30.0, gt=0.0, description="Max execution timeout before cancellation")
    requires_permissions: List[str] = Field(default_factory=list, description="Security permissions needed to invoke this tool")
    tags: List[str] = Field(default_factory=list, description="Classification tags (e.g. search, calculation, database)")


class BaseTool(ABC):
    """Abstract base class for all deterministic and external tools."""

    def __init__(self, metadata: ToolMetadata) -> None:
        self.metadata = metadata

    @property
    def name(self) -> str:
        """Accessor for tool name."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Accessor for tool description."""
        return self.metadata.description

    @property
    @abstractmethod
    def input_schema(self) -> Type[BaseModel]:
        """Pydantic model defining valid input parameters."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> Type[BaseModel]:
        """Pydantic model defining structured output result."""
        pass

    def validate_input(self, raw_input: Dict[str, Any]) -> BaseModel:
        """Validate raw dictionary input against tool input schema."""
        try:
            return self.input_schema.model_validate(raw_input)
        except Exception as e:
            raise ValidationError(
                message=f"Validation failed for tool '{self.name}' input: {e}",
                details={"tool_name": self.name, "raw_input": raw_input},
            ) from e

    @abstractmethod
    async def execute(self, input_data: BaseModel) -> BaseModel:
        """Execute the tool logic using validated input parameters."""
        pass
