"""Standardized agent result contract."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Execution status returned by an agent."""

    SUCCESS = "success"
    FAILURE = "failure"
    RETRYABLE_FAILURE = "retryable_failure"
    ESCALATE_HUMAN = "escalate_human"
    CANCELLED = "cancelled"


class AgentResult(BaseModel):
    """Standardized outcome envelope returned by any agent execution."""

    agent_name: str = Field(..., description="Name of the agent that produced this result")
    status: AgentStatus = Field(default=AgentStatus.SUCCESS, description="Execution outcome status")
    output: Optional[Dict[str, Any]] = Field(default=None, description="Structured payload produced by agent")
    raw_response: Optional[str] = Field(default=None, description="Raw textual output from LLM/service if applicable")
    error: Optional[str] = Field(default=None, description="Error message if execution was unsuccessful")
    retryable: bool = Field(default=False, description="Whether the supervisor may retry this operation")
    requires_human: bool = Field(default=False, description="Whether this result mandates human escalation")
    execution_time_ms: float = Field(default=0.0, ge=0.0, description="Duration of agent execution in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic, token usage, or tracing metadata")

    @classmethod
    def success(
        cls,
        agent_name: str,
        output: Dict[str, Any],
        raw_response: Optional[str] = None,
        execution_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":
        """Factory method for successful agent results."""
        return cls(
            agent_name=agent_name,
            status=AgentStatus.SUCCESS,
            output=output,
            raw_response=raw_response,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        agent_name: str,
        error: str,
        retryable: bool = False,
        execution_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":
        """Factory method for failed agent results."""
        status = AgentStatus.RETRYABLE_FAILURE if retryable else AgentStatus.FAILURE
        return cls(
            agent_name=agent_name,
            status=status,
            error=error,
            retryable=retryable,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
        )

    @classmethod
    def escalate(
        cls,
        agent_name: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
        execution_time_ms: float = 0.0,
    ) -> "AgentResult":
        """Factory method for escalation results."""
        return cls(
            agent_name=agent_name,
            status=AgentStatus.ESCALATE_HUMAN,
            error=reason,
            requires_human=True,
            output=context,
            execution_time_ms=execution_time_ms,
        )
