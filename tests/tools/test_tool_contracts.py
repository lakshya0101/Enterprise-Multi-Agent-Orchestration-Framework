"""Tests for tool interfaces and tool registry execution contracts."""

from typing import Type

import pytest
from pydantic import BaseModel, Field

from enterprise_orchestrator.errors.exceptions import (
    ToolNotFoundError,
    ValidationError,
)
from enterprise_orchestrator.tools.base import BaseTool, ToolMetadata
from enterprise_orchestrator.tools.registry import ToolRegistry


class CalculatorInput(BaseModel):
    a: float = Field(..., description="First operand")
    b: float = Field(..., description="Second operand")
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")


class CalculatorOutput(BaseModel):
    result: float = Field(..., description="Calculated result")


class CalculatorTool(BaseTool):
    """Deterministic calculator tool implementation for testing."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="calculator",
                description="Performs basic arithmetic operations",
                tags=["math", "utility"],
                timeout_seconds=5.0,
            )
        )

    @property
    def input_schema(self) -> Type[BaseModel]:
        return CalculatorInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return CalculatorOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        data = CalculatorInput.model_validate(input_data)
        if data.operation == "add":
            return CalculatorOutput(result=data.a + data.b)
        elif data.operation == "multiply":
            return CalculatorOutput(result=data.a * data.b)
        elif data.operation == "divide":
            if data.b == 0:
                raise ValueError("Division by zero is not allowed.")
            return CalculatorOutput(result=data.a / data.b)
        raise ValueError(f"Unknown operation: {data.operation}")


@pytest.mark.asyncio
async def test_tool_registry_lifecycle():
    """Verify tool registration, lookup, list, and unregistration."""
    registry = ToolRegistry()
    tool = CalculatorTool()

    assert registry.exists("calculator") is False
    registry.register(tool)
    assert registry.exists("calculator") is True

    # Duplicate registration prevention
    with pytest.raises(ValueError):
        registry.register(tool)

    fetched = registry.get("calculator")
    assert fetched is not None
    assert fetched.name == "calculator"

    tools_list = registry.list_tools()
    assert len(tools_list) == 1
    assert tools_list[0].name == "calculator"

    # Execution via registry
    result = await registry.execute("calculator", {"a": 10, "b": 5, "operation": "add"})
    assert result == {"result": 15.0}

    # Unregister
    assert registry.unregister("calculator") is True
    assert registry.exists("calculator") is False
    assert registry.get("calculator") is None


@pytest.mark.asyncio
async def test_tool_execution_errors():
    """Verify registry behavior on missing tools and invalid parameters."""
    registry = ToolRegistry()
    tool = CalculatorTool()
    registry.register(tool)

    # Missing tool error
    with pytest.raises(ToolNotFoundError):
        await registry.execute("unknown_tool", {})

    # Invalid input parameter validation
    with pytest.raises(ValidationError):
        await registry.execute("calculator", {"a": "invalid_number", "b": 2, "operation": "add"})
