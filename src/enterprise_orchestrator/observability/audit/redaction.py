"""Sensitive data redactor providing multi-layered field-based and pattern-based redaction."""

import re
from typing import Any, Dict, List, Optional, Set


class SensitiveDataRedactor:
    """Enterprise redaction pipeline for audit logs, traces, and metrics."""

    DEFAULT_SENSITIVE_KEYS: Set[str] = {
        "api_key",
        "apikey",
        "api-key",
        "secret",
        "secret_key",
        "password",
        "token",
        "auth_token",
        "access_token",
        "refresh_token",
        "authorization",
        "credential",
        "credentials",
        "private_key",
        "client_secret",
        "session_secret",
        "bearer",
    }

    # Regex patterns for secondary defense
    BEARER_PATTERN = re.compile(r"Bearer\s+([A-Za-z0-9\-\._~\+\/]+=*)", re.IGNORECASE)
    API_KEY_PATTERN = re.compile(r"(?:key|api[_-]?key|token)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{16,})['\"]?", re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")

    def __init__(
        self,
        sensitive_keys: Optional[Set[str]] = None,
        custom_patterns: Optional[List[re.Pattern]] = None,
        mask: str = "[REDACTED]",
    ) -> None:
        self.sensitive_keys = {k.lower() for k in (sensitive_keys or self.DEFAULT_SENSITIVE_KEYS)}
        self.patterns = custom_patterns or [self.BEARER_PATTERN, self.API_KEY_PATTERN, self.EMAIL_PATTERN]
        self.mask = mask

    def redact_text(self, text: str) -> str:
        """Apply regex-based pattern redaction to raw text strings."""
        if not isinstance(text, str):
            return text

        result = text
        for pattern in self.patterns:
            result = pattern.sub(self.mask, result)
        return result

    def redact(self, data: Any) -> Any:
        """Recursively redact sensitive field names and values in arbitrary nested structures."""
        if isinstance(data, dict):
            redacted_dict: Dict[str, Any] = {}
            for k, v in data.items():
                k_lower = str(k).lower().strip()
                if any(k_lower == s_key or k_lower.endswith(f"_{s_key}") for s_key in self.sensitive_keys):
                    redacted_dict[k] = self.mask
                else:
                    redacted_dict[k] = self.redact(v)
            return redacted_dict

        elif isinstance(data, list):
            return [self.redact(item) for item in data]

        elif isinstance(data, tuple):
            return tuple(self.redact(item) for item in data)

        elif isinstance(data, str):
            return self.redact_text(data)

        return data
