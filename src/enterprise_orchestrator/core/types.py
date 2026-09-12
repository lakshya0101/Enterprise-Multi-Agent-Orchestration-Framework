"""Core enumerations and type definitions for the orchestration framework."""

from enum import Enum


class TaskStatus(str, Enum):
    """Lifecycle status of an individual task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Categorization of task purpose for routing and specialized agent dispatch."""

    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    TOOL_EXECUTION = "tool_execution"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    CUSTOM = "custom"


class AgentRole(str, Enum):
    """Standardized roles for specialized agents."""

    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    RETRIEVER = "retriever"
    TOOL_EXECUTOR = "tool_executor"
    VALIDATOR = "validator"
    CUSTOM = "custom"


class ExecutionStatus(str, Enum):
    """Global lifecycle status of an orchestration run."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED_FOR_HUMAN = "paused_for_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationStatus(str, Enum):
    """Verdict of a validator/critic inspection."""

    VALID = "valid"
    INVALID = "invalid"
    NEEDS_RETRY = "needs_retry"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class HumanRequestStatus(str, Enum):
    """State of an escalation request awaiting human input."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    TIMEOUT = "timeout"


class MessageRole(str, Enum):
    """Standard conversational and instruction roles for LLM exchanges."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
