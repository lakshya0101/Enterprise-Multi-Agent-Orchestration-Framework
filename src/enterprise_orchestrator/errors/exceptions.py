"""Framework error taxonomy and exception definitions."""

from typing import Any, Dict, Optional


def sanitize_message(message: str, sensitive_patterns: Optional[list[str]] = None) -> str:
    """Sanitize sensitive patterns from error messages to prevent credential leakage."""
    if not message:
        return message
    sanitized = message
    if sensitive_patterns:
        for pattern in sensitive_patterns:
            if pattern and len(pattern) > 4:
                sanitized = sanitized.replace(pattern, "[REDACTED_SECRET]")
    return sanitized


class OrchestratorError(Exception):
    """Base exception for all Enterprise Multi-Agent Orchestration Framework errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ORCHESTRATOR_ERROR",
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.raw_message = message
        self.message = sanitize_message(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error representation to a serializable dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


class ValidationError(OrchestratorError):
    """Raised when state, input, or output validation fails."""

    def __init__(
        self,
        message: str,
        code: str = "VALIDATION_ERROR",
        retryable: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, retryable=retryable, details=details)


class ConfigurationError(OrchestratorError):
    """Raised when environment or framework configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        code: str = "CONFIGURATION_ERROR",
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, retryable=retryable, details=details)


class ProviderError(OrchestratorError):
    """Raised when an LLM provider encounters an error during generation."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        code: str = "PROVIDER_ERROR",
        status_code: Optional[int] = None,
        retryable: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        details["provider_name"] = provider_name
        if status_code is not None:
            details["status_code"] = status_code
        self.provider_name = provider_name
        self.status_code = status_code
        super().__init__(message=message, code=code, retryable=retryable, details=details)


class ProviderUnavailableError(ProviderError):
    """Raised when a specific LLM provider is down, rate-limited, or unresponsive."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            provider_name=provider_name,
            code="PROVIDER_UNAVAILABLE",
            status_code=status_code,
            retryable=True,
            details=details,
        )


class ToolError(OrchestratorError):
    """Base exception for tool registry and execution errors."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        code: str = "TOOL_ERROR",
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        details["tool_name"] = tool_name
        self.tool_name = tool_name
        super().__init__(message=message, code=code, retryable=retryable, details=details)


class ToolNotFoundError(ToolError):
    """Raised when a requested tool does not exist in the registry."""

    def __init__(
        self,
        tool_name: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=f"Tool '{tool_name}' is not registered in the tool registry.",
            tool_name=tool_name,
            code="TOOL_NOT_FOUND",
            retryable=False,
            details=details,
        )


class ToolExecutionError(ToolError):
    """Raised when tool execution fails unexpectedly."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        retryable: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            tool_name=tool_name,
            code="TOOL_EXECUTION_ERROR",
            retryable=retryable,
            details=details,
        )


class RetrievalError(OrchestratorError):
    """Raised when context retrieval from vector store or knowledge base fails."""

    def __init__(
        self,
        message: str,
        code: str = "RETRIEVAL_ERROR",
        retryable: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, retryable=retryable, details=details)


class OrchestrationError(OrchestratorError):
    """Raised when the supervisor encounters a routing or state transition anomaly."""

    def __init__(
        self,
        message: str,
        code: str = "ORCHESTRATION_ERROR",
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, code=code, retryable=retryable, details=details)


class TaskDependencyError(OrchestrationError):
    """Raised when a task is scheduled before its required dependencies are completed."""

    def __init__(
        self,
        task_id: str,
        unmet_dependencies: list[str],
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        details["task_id"] = task_id
        details["unmet_dependencies"] = unmet_dependencies
        super().__init__(
            message=f"Task '{task_id}' cannot execute. Unmet dependencies: {unmet_dependencies}",
            code="TASK_DEPENDENCY_ERROR",
            retryable=False,
            details=details,
        )


class TimeoutError(OrchestratorError):
    """Raised when an operation exceeds its configured deadline."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        code: str = "TIMEOUT_ERROR",
        retryable: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        details["timeout_seconds"] = timeout_seconds
        self.timeout_seconds = timeout_seconds
        super().__init__(message=message, code=code, retryable=retryable, details=details)


class HumanInterventionRequired(OrchestratorError):
    """Raised or captured when execution pauses for human approval or input."""

    def __init__(
        self,
        reason: str,
        request_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        details["request_id"] = request_id
        self.request_id = request_id
        super().__init__(
            message=f"Human intervention required for request '{request_id}': {reason}",
            code="HUMAN_INTERVENTION_REQUIRED",
            retryable=False,
            details=details,
        )


class AuthenticationError(OrchestratorError):
    """Raised when identity authentication fails or credentials are missing/invalid."""

    def __init__(
        self,
        message: str = "Authentication failed: Missing or invalid credentials.",
        code: str = "AUTHENTICATION_FAILED",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            retryable=False,
            details=details,
        )


class AuthorizationError(OrchestratorError):
    """Raised when an authenticated identity lacks required permissions or roles."""

    def __init__(
        self,
        message: str = "Forbidden: Insufficient permissions to perform this action.",
        code: str = "FORBIDDEN",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            retryable=False,
            details=details,
        )
