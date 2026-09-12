"""Tests for error taxonomy, classification, serialization, and secret redaction."""

from enterprise_orchestrator.errors.exceptions import (
    ConfigurationError,
    HumanInterventionRequired,
    OrchestratorError,
    ProviderError,
    ProviderUnavailableError,
    TaskDependencyError,
    TimeoutError,
    ToolError,
    ToolNotFoundError,
    ValidationError,
    sanitize_message,
)


def test_sanitize_message():
    """Verify sensitive token redaction."""
    secret = "AIzaSySecretApiKey12345"
    raw_msg = f"Failed to connect using key {secret} on endpoint"
    sanitized = sanitize_message(raw_msg, sensitive_patterns=[secret])
    assert secret not in sanitized
    assert "[REDACTED_SECRET]" in sanitized


def test_error_hierarchy_and_serialization():
    """Verify error attributes, retryable flags, and to_dict method."""
    err = ProviderUnavailableError(
        message="Groq service rate limited",
        provider_name="groq",
        status_code=429,
        details={"model": "llama-3.3-70b"},
    )
    assert isinstance(err, ProviderError)
    assert isinstance(err, OrchestratorError)
    assert err.retryable is True
    assert err.code == "PROVIDER_UNAVAILABLE"
    assert err.provider_name == "groq"
    assert err.status_code == 429

    err_dict = err.to_dict()
    assert err_dict["error_type"] == "ProviderUnavailableError"
    assert err_dict["retryable"] is True
    assert err_dict["details"]["provider_name"] == "groq"


def test_task_dependency_error():
    """Verify task dependency error details."""
    err = TaskDependencyError(task_id="task-4", unmet_dependencies=["task-1", "task-2"])
    assert err.retryable is False
    assert err.code == "TASK_DEPENDENCY_ERROR"
    assert err.details["unmet_dependencies"] == ["task-1", "task-2"]


def test_validation_and_tool_errors():
    """Verify validation and tool error models."""
    v_err = ValidationError(message="Schema mismatch")
    assert v_err.retryable is True

    t_err = ToolNotFoundError(tool_name="web_search")
    assert t_err.retryable is False
    assert t_err.tool_name == "web_search"

    to_err = TimeoutError(message="Operation deadline exceeded", timeout_seconds=30.0)
    assert to_err.retryable is True
    assert to_err.timeout_seconds == 30.0

    cfg_err = ConfigurationError(message="Missing config file")
    assert cfg_err.retryable is False

    hitl_err = HumanInterventionRequired(reason="Payment confirmation needed", request_id="req-1")
    assert hitl_err.retryable is False
    assert hitl_err.request_id == "req-1"
