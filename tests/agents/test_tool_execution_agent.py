"""Unit tests for ToolExecutionAgent and ToolRegistry integration."""

from typing import Type

import pytest
from pydantic import BaseModel, Field

from enterprise_orchestrator.agents.result import AgentStatus
from enterprise_orchestrator.agents.tool_execution import ToolExecutionAgent
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import Task
from enterprise_orchestrator.tools.base import BaseTool, ToolMetadata
from enterprise_orchestrator.tools.registry import ToolRegistry


class MultiplyInput(BaseModel):
    x: float = Field(...)
    y: float = Field(...)


class MultiplyOutput(BaseModel):
    product: float = Field(...)


class MultiplyTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(name="multiply_tool", description="Multiplies numbers", timeout_seconds=2.0)
        )

    @property
    def input_schema(self) -> Type[BaseModel]:
        return MultiplyInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return MultiplyOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        data = MultiplyInput.model_validate(input_data)
        return MultiplyOutput(product=data.x * data.y)


@pytest.fixture
def populated_tool_registry():
    registry = ToolRegistry()
    registry.register(MultiplyTool())
    return registry


@pytest.mark.asyncio
async def test_tool_execution_agent_success(populated_tool_registry):
    """Verify happy path execution of registered tool via ToolExecutionAgent."""
    agent = ToolExecutionAgent(tool_registry=populated_tool_registry)
    state = OrchestrationState.create_initial(request="Calculate numbers")
    task = Task(
        title="Execute Multiplication",
        input_data={"tool_name": "multiply_tool", "parameters": {"x": 6, "y": 7}},
    )

    result = await agent.execute(state, task)

    assert result.status == AgentStatus.SUCCESS
    assert result.agent_name == "tool_execution_agent"
    assert result.output["result"] == {"product": 42.0}
    assert state.tool_results["multiply_tool"] == {"product": 42.0}


@pytest.mark.asyncio
async def test_tool_execution_agent_flat_parameters(populated_tool_registry):
    """Verify tool parameter extraction when passed directly in input_data."""
    agent = ToolExecutionAgent(tool_registry=populated_tool_registry)
    state = OrchestrationState.create_initial(request="Calculate numbers")
    task = Task(
        title="Execute Multiplication Flat",
        input_data={"tool_name": "multiply_tool", "x": 5, "y": 4},
    )

    result = await agent.execute(state, task)
    assert result.status == AgentStatus.SUCCESS
    assert result.output["result"] == {"product": 20.0}


@pytest.mark.asyncio
async def test_tool_execution_agent_missing_tool_name(populated_tool_registry):
    """Verify rejection when task input lacks 'tool_name'."""
    agent = ToolExecutionAgent(tool_registry=populated_tool_registry)
    state = OrchestrationState.create_initial(request="Calculate")
    task = Task(title="Invalid Task", input_data={"x": 5})

    result = await agent.execute(state, task)
    assert result.status == AgentStatus.FAILURE
    assert "missing required 'tool_name'" in result.error


@pytest.mark.asyncio
async def test_tool_execution_agent_unknown_tool(populated_tool_registry):
    """Verify failure when requested tool is not in registry."""
    agent = ToolExecutionAgent(tool_registry=populated_tool_registry)
    state = OrchestrationState.create_initial(request="Calculate")
    task = Task(title="Missing Tool", input_data={"tool_name": "weather_api", "city": "NYC"})

    result = await agent.execute(state, task)
    assert result.status == AgentStatus.FAILURE
    assert "Tool not found" in result.error
    assert result.retryable is False


@pytest.mark.asyncio
async def test_tool_execution_agent_invalid_parameters(populated_tool_registry):
    """Verify schema validation failure when invalid parameters are provided."""
    agent = ToolExecutionAgent(tool_registry=populated_tool_registry)
    state = OrchestrationState.create_initial(request="Calculate")
    task = Task(
        title="Invalid Types",
        input_data={"tool_name": "multiply_tool", "parameters": {"x": "not_a_number", "y": 5}},
    )

    result = await agent.execute(state, task)
    assert result.status == AgentStatus.FAILURE
    assert "validation error" in result.error
