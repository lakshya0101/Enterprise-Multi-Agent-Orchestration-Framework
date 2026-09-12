"""Unit tests for SensitiveDataRedactor."""

import pytest

from enterprise_orchestrator.observability.audit.redaction import SensitiveDataRedactor


def test_field_name_redaction() -> None:
    redactor = SensitiveDataRedactor()
    data = {
        "user_id": "usr-123",
        "api_key": "sk-secret1234567890",
        "auth_token": "bearer-xyz",
        "nested": {
            "password": "super-secret-pass",
            "normal_field": "visible-data",
            "client_secret": "my-secret-value",
        },
        "list_items": [
            {"token": "token-in-list", "name": "item1"},
            {"secret": "secret-in-list"},
        ],
    }

    redacted = redactor.redact(data)

    assert redacted["user_id"] == "usr-123"
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["auth_token"] == "[REDACTED]"
    assert redacted["nested"]["password"] == "[REDACTED]"
    assert redacted["nested"]["normal_field"] == "visible-data"
    assert redacted["nested"]["client_secret"] == "[REDACTED]"
    assert redacted["list_items"][0]["token"] == "[REDACTED]"
    assert redacted["list_items"][0]["name"] == "item1"
    assert redacted["list_items"][1]["secret"] == "[REDACTED]"


def test_regex_pattern_redaction() -> None:
    redactor = SensitiveDataRedactor()
    text = "Authorization header: Bearer ya29.a0AfH6SMD_secrettoken and email is test@company.com"
    redacted = redactor.redact_text(text)

    assert "Bearer ya29" not in redacted
    assert "test@company.com" not in redacted
    assert "[REDACTED]" in redacted
