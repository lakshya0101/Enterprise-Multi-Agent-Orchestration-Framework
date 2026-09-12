"""Tests for agent interface contracts and fake agent execution."""

from typing import Optional

import pytest

from enterprise_orchestrator.agents.base import AgentMetadata, BaseAgent
from enterprise_orchestrator.agents.result import AgentResult, AgentStatus
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import Task
from enterprise_orchestrator.core.types import AgentRole, TaskType


class FakePlannerAgent(BaseAgent):
    """Fake planner agent implementing BaseAgent for contract testing."""

    def __init__(self) -> None:
        super().__init__(
            metadata=AgentMetadata(
                name="fake_planner",
                role=AgentRole.PLANNER,
                description="Generates mock execution plans",
                capabilities=["decomposition", "task_graph"],
            )
        )

    def validate_input(self, state: OrchestrationState, task: Optional[Task] = None) -> bool:
        return bool(state.request and len(state.request) > 0)

    async def execute(self, state: OrchestrationState, task: Optional[Task] = None) -> AgentResult:
        if not self.validate_input(state, task):
            return AgentResult.failure(agent_name=self.name, error="Empty request", retryable=False)

        plan_data = {
            "tasks": [
                {"title": "Search data", "type": "retrieval"},
                {"title": "Summarize", "type": "synthesis"},
            ]
        }
        return AgentResult.success(
            agent_name=self.name,
            output=plan_data,
            raw_response="Plan generated successfully.",
            execution_time_ms=12.5,
        )


@pytest.mark.asyncio
async def test_agent_execution_success():
    """Verify standard success flow for an agent."""
    agent = FakePlannerAgent()
    assert agent.name == "fake_planner"
    assert agent.role == AgentRole.PLANNER

    state = OrchestrationState.create_initial(request="Plan a market analysis")
    result = await agent.execute(state)

    assert result.status == AgentStatus.SUCCESS
    assert result.agent_name == "fake_planner"
    assert "tasks" in result.output
    assert result.execution_time_ms == 12.5
    assert result.retryable is False


@pytest.mark.asyncio
async def test_agent_execution_failure():
    """Verify failure result flow for an agent."""
    agent = FakePlannerAgent()
    state = OrchestrationState(request="x")
    state.request = ""  # Force invalid input

    result = await agent.execute(state)
    assert result.status == AgentStatus.FAILURE
    assert result.error == "Empty request"


def test_agent_result_factories():
    """Verify result factory methods."""
    res_success = AgentResult.success("agent1", {"a": 1})
    assert res_success.status == AgentStatus.SUCCESS
    assert res_success.output == {"a": 1}

    res_retry = AgentResult.failure("agent1", "Rate limited", retryable=True)
    assert res_retry.status == AgentStatus.RETRYABLE_FAILURE
    assert res_retry.retryable is True

    res_esc = AgentResult.escalate("agent1", "Uncertain calculation", context={"val": -1})
    assert res_esc.status == AgentStatus.ESCALATE_HUMAN
    assert res_esc.requires_human is True
