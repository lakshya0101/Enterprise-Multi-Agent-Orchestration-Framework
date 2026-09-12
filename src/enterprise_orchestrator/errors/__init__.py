"""Framework exceptions and error taxonomy."""

from enterprise_orchestrator.errors.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    HumanInterventionRequired,
    OrchestrationError,
    OrchestratorError,
    ProviderError,
    ProviderUnavailableError,
    RetrievalError,
    TaskDependencyError,
    TimeoutError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ValidationError,
    sanitize_message,
)

__all__ = [
    "OrchestratorError",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "ConfigurationError",
    "ProviderError",
    "ProviderUnavailableError",
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "RetrievalError",
    "OrchestrationError",
    "TaskDependencyError",
    "TimeoutError",
    "HumanInterventionRequired",
    "sanitize_message",
]
