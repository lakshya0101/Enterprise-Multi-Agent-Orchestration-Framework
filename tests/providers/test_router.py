"""Unit tests for LLMRouter multi-provider dispatch and fallback behavior."""

from typing import Type, TypeVar
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from enterprise_orchestrator.core.types import MessageRole
from enterprise_orchestrator.errors.exceptions import (
    ConfigurationError,
    ProviderError,
    ProviderUnavailableError,
)
from enterprise_orchestrator.providers.base import BaseLLMProvider
from enterprise_orchestrator.providers.models import (
    LLMRequest,
    LLMResponse,
    Message,
    TokenUsage,
)
from enterprise_orchestrator.providers.router import LLMRouter

T = TypeVar("T", bound=BaseModel)


class MockProvider(BaseLLMProvider):
    """Configurable mock provider for router logic verification."""

    def __init__(self, name: str, default_model: str = "mock-model") -> None:
        super().__init__(provider_name=name, default_model=default_model)
        self.generate_mock = AsyncMock()
        self.generate_structured_mock = AsyncMock()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return await self.generate_mock(request)

    async def generate_structured(self, request: LLMRequest, schema: Type[T]) -> tuple[T, LLMResponse]:
        return await self.generate_structured_mock(request, schema)

    async def health_check(self) -> bool:
        return True


class SampleOutputSchema(BaseModel):
    value: str = Field(...)


@pytest.mark.asyncio
async def test_router_primary_provider_success():
    """Verify router dispatches directly to primary provider when healthy."""
    p_gemini = MockProvider("gemini")
    p_groq = MockProvider("groq")

    p_gemini.generate_mock.return_value = LLMResponse(
        content="Primary Gemini response",
        model="gemini-2.5-flash",
        provider="gemini",
        usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
    )

    router = LLMRouter(
        providers={"gemini": p_gemini, "groq": p_groq},
        default_provider="gemini",
        fallback_provider="groq",
        enable_fallback=True,
    )

    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="Hello")])
    resp = await router.generate(req)

    assert resp.provider == "gemini"
    assert resp.content == "Primary Gemini response"
    assert p_gemini.generate_mock.call_count == 1
    assert p_groq.generate_mock.call_count == 0


@pytest.mark.asyncio
async def test_router_automatic_fallback_on_429_rate_limit():
    """Verify router transparently falls back to secondary provider on retryable failure."""
    p_gemini = MockProvider("gemini")
    p_groq = MockProvider("groq")

    # Gemini fails with 429 Rate Limit
    p_gemini.generate_mock.side_effect = ProviderUnavailableError(
        message="Gemini quota exceeded",
        provider_name="gemini",
        status_code=429,
    )

    # Groq succeeds
    p_groq.generate_mock.return_value = LLMResponse(
        content="Fallback Groq response",
        model="llama-3.3-70b-versatile",
        provider="groq",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
    )

    router = LLMRouter(
        providers={"gemini": p_gemini, "groq": p_groq},
        default_provider="gemini",
        fallback_provider="groq",
        enable_fallback=True,
    )

    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="Hello")])
    resp = await router.generate(req)

    assert resp.provider == "groq"
    assert resp.content == "Fallback Groq response"
    assert p_gemini.generate_mock.call_count == 1
    assert p_groq.generate_mock.call_count == 1
    assert resp.raw_response["router_metadata"]["fallback_used"] is True
    assert resp.raw_response["router_metadata"]["primary_provider"] == "gemini"


@pytest.mark.asyncio
async def test_router_no_fallback_on_non_retryable_error():
    """Verify router immediately bubbles up non-retryable errors without attempting fallback."""
    p_gemini = MockProvider("gemini")
    p_groq = MockProvider("groq")

    # Non-retryable authentication error
    p_gemini.generate_mock.side_effect = ProviderError(
        message="Invalid credentials",
        provider_name="gemini",
        retryable=False,
    )

    router = LLMRouter(
        providers={"gemini": p_gemini, "groq": p_groq},
        default_provider="gemini",
        fallback_provider="groq",
        enable_fallback=True,
    )

    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="Hello")])
    with pytest.raises(ProviderError) as exc_info:
        await router.generate(req)

    assert exc_info.value.retryable is False
    assert p_gemini.generate_mock.call_count == 1
    assert p_groq.generate_mock.call_count == 0


@pytest.mark.asyncio
async def test_router_both_providers_fail():
    """Verify consolidated ProviderError raised if both primary and fallback fail."""
    p_gemini = MockProvider("gemini")
    p_groq = MockProvider("groq")

    p_gemini.generate_mock.side_effect = ProviderUnavailableError("Gemini 503", "gemini", status_code=503)
    p_groq.generate_mock.side_effect = ProviderUnavailableError("Groq 429", "groq", status_code=429)

    router = LLMRouter(
        providers={"gemini": p_gemini, "groq": p_groq},
        default_provider="gemini",
        fallback_provider="groq",
        enable_fallback=True,
    )

    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="Hello")])
    with pytest.raises(ProviderError) as exc_info:
        await router.generate(req)

    assert "All available LLM providers failed" in str(exc_info.value)
    assert p_gemini.generate_mock.call_count == 1
    assert p_groq.generate_mock.call_count == 1


@pytest.mark.asyncio
async def test_router_structured_generation_fallback():
    """Verify fallback works identically for structured output generation."""
    p_gemini = MockProvider("gemini")
    p_groq = MockProvider("groq")

    p_gemini.generate_structured_mock.side_effect = ProviderUnavailableError("Gemini down", "gemini")

    mock_model = SampleOutputSchema(value="Groq structured output")
    mock_resp = LLMResponse(content="{}", model="llama", provider="groq")
    p_groq.generate_structured_mock.return_value = (mock_model, mock_resp)

    router = LLMRouter(
        providers={"gemini": p_gemini, "groq": p_groq},
        default_provider="gemini",
        fallback_provider="groq",
        enable_fallback=True,
    )

    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="Get value")])
    parsed, resp = await router.generate_structured(req, SampleOutputSchema)

    assert parsed.value == "Groq structured output"
    assert resp.provider == "groq"
    assert resp.raw_response["router_metadata"]["fallback_used"] is True
