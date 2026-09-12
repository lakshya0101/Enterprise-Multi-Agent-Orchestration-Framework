"""Tool execution agent bridging task workflows to the sandboxed ToolRegistry."""

import time
from typing import Any, Dict, Optional

from enterprise_orchestrator.agents.base import AgentMetadata, BaseAgent
from enterprise_orchestrator.agents.result import AgentResult
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import Task
from enterprise_orchestrator.core.types import AgentRole
from enterprise_orchestrator.errors.exceptions import (
    OrchestratorError,
    TimeoutError as OrchestratorTimeoutError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ValidationError,
)
from enterprise_orchestrator.tools.registry import ToolRegistry


class ToolExecutionAgent(BaseAgent):
    """Agent responsible for resolving, validating, and executing tools registered in the ToolRegistry."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        metadata = AgentMetadata(
            name="tool_execution_agent",
            role=AgentRole.TOOL_EXECUTOR,
            description="Executes deterministic tools and external integrations registered in the ToolRegistry",
            capabilities=["tool_dispatch", "parameter_validation", "sandboxed_execution"],
            input_schema_desc="Task input containing 'tool_name' and 'parameters' dict",
            output_schema_desc="Dict containing tool execution output",
        )
        super().__init__(metadata=metadata)
        self.tool_registry = tool_registry

    def validate_input(self, state: OrchestrationState, task: Optional[Task] = None) -> bool:
        """Verify that a target tool name is specified in task input."""
        if not task or not task.input_data:
            return False
        tool_name = task.input_data.get("tool_name")
        return bool(tool_name and str(tool_name).strip())

    async def execute(self, state: OrchestrationState, task: Optional[Task] = None) -> AgentResult:
        """Resolve and execute the designated tool via the ToolRegistry."""
        start_time = time.perf_counter()

        if not self.validate_input(state, task):
            return AgentResult.failure(
                agent_name=self.name,
                error="Task input_data missing required 'tool_name' field.",
                retryable=False,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        assert task is not None  # Guaranteed by validate_input
        tool_name = str(task.input_data.get("tool_name")).strip()
        # Accept parameters either nested under 'parameters' or as the full input dict (excluding tool_name)
        params = task.input_data.get("parameters")
        if params is None:
            params = {k: v for k, v in task.input_data.items() if k != "tool_name"}

        if not isinstance(params, dict):
            return AgentResult.failure(
                agent_name=self.name,
                error=f"Tool parameters must be a dictionary, got {type(params).__name__}.",
                retryable=False,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        try:
            result_dict = await self.tool_registry.execute(tool_name, params)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record in state for downstream tasks
            state.add_tool_result(tool_name=tool_name, result=result_dict)

            return AgentResult.success(
                agent_name=self.name,
                output={"tool_name": tool_name, "result": result_dict},
                execution_time_ms=latency_ms,
                metadata={"tool_name": tool_name},
            )

        except ToolNotFoundError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return AgentResult.failure(
                agent_name=self.name,
                error=f"Tool not found: {e.message}",
                retryable=False,
                execution_time_ms=latency_ms,
                metadata={"code": e.code, "tool_name": tool_name},
            )
        except ValidationError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return AgentResult.failure(
                agent_name=self.name,
                error=f"Tool input parameter validation error: {e.message}",
                retryable=False,
                execution_time_ms=latency_ms,
                metadata={"code": e.code, "tool_name": tool_name},
            )
        except OrchestratorTimeoutError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return AgentResult.failure(
                agent_name=self.name,
                error=f"Tool execution timed out: {e.message}",
                retryable=True,
                execution_time_ms=latency_ms,
                metadata={"code": e.code, "tool_name": tool_name},
            )
        except ToolExecutionError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return AgentResult.failure(
                agent_name=self.name,
                error=f"Tool execution failed: {e.message}",
                retryable=e.retryable,
                execution_time_ms=latency_ms,
                metadata={"code": e.code, "tool_name": tool_name},
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return AgentResult.failure(
                agent_name=self.name,
                error=f"Unexpected error executing tool '{tool_name}': {e}",
                retryable=False,
                execution_time_ms=latency_ms,
            )
