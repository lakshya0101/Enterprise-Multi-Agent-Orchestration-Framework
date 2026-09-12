"""Tool registry for registering, discovering, and dispatching tool invocations."""

import asyncio
from typing import Any, Dict, List, Optional

from enterprise_orchestrator.errors.exceptions import (
    TimeoutError as OrchestratorTimeoutError,
)
from enterprise_orchestrator.errors.exceptions import (
    ToolExecutionError,
    ToolNotFoundError,
)
from enterprise_orchestrator.tools.base import BaseTool, ToolMetadata


class ToolRegistry:
    """Central registry managing lifecycle and dispatch for all executable tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance in the registry."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> bool:
        """Remove a tool from the registry. Returns True if removed, False otherwise."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Look up a tool by name."""
        return self._tools.get(tool_name)

    def list_tools(self) -> List[ToolMetadata]:
        """Return metadata for all registered tools."""
        return [tool.metadata for tool in self._tools.values()]

    def exists(self, tool_name: str) -> bool:
        """Check whether a tool exists in the registry."""
        return tool_name in self._tools

    async def execute(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool by name with timeout enforcement and error mapping."""
        tool = self.get(tool_name)
        if not tool:
            raise ToolNotFoundError(tool_name=tool_name)

        validated_input = tool.validate_input(input_data)
        timeout_seconds = tool.metadata.timeout_seconds

        try:
            result = await asyncio.wait_for(
                tool.execute(validated_input),
                timeout=timeout_seconds,
            )
            return result.model_dump()
        except asyncio.TimeoutError as e:
            raise OrchestratorTimeoutError(
                message=f"Tool '{tool_name}' timed out after {timeout_seconds} seconds.",
                timeout_seconds=timeout_seconds,
                details={"tool_name": tool_name},
            ) from e
        except Exception as e:
            if isinstance(e, (ToolNotFoundError, OrchestratorTimeoutError)):
                raise
            raise ToolExecutionError(
                message=f"Tool '{tool_name}' execution failed: {e}",
                tool_name=tool_name,
                retryable=True,
                details={"tool_name": tool_name, "error": str(e)},
            ) from e
