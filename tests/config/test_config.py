"""Tests for configuration management and secret safety."""

from enterprise_orchestrator.config.settings import FrameworkSettings


def test_framework_settings_defaults():
    """Verify settings defaults and types."""
    settings = FrameworkSettings()
    assert settings.default_provider == "gemini"
    assert settings.fallback_provider == "groq"
    assert settings.default_timeout_seconds > 0
    assert settings.max_global_retries >= 1


def test_framework_settings_safe_dump():
    """Verify secrets are redacted in safe dump and string representation."""
    settings = FrameworkSettings(
        gemini_api_key="secret-gemini-key",
        groq_api_key="secret-groq-key",
        langchain_api_key="secret-lc-key",
    )
    safe_data = settings.model_dump_safe()
    assert safe_data["gemini_api_key"] == "[REDACTED]"
    assert safe_data["groq_api_key"] == "[REDACTED]"
    assert safe_data["langchain_api_key"] == "[REDACTED]"

    rep = repr(settings)
    assert "secret-gemini-key" not in rep
    assert "secret-groq-key" not in rep
    assert "[REDACTED]" in rep
