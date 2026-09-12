"""Tests for LLM Provider contract and fake provider implementation."""

from typing import Type, TypeVar

import pytest
from pydantic import BaseModel, Field

from enterprise_orchestrator.core.types import MessageRole
from enterprise_orchestrator.errors.exceptions import ProviderUnavailableError
from enterprise_orchestrator.providers.base import BaseLLMProvider
from enterprise_orchestrator.providers.models import LLMRequest, LLMResponse, Message, TokenUsage

T = TypeVar("T", bound=BaseModel)


class StructuredSummary(BaseModel):
    """Target test schema for structured generation."""

    title: str = Field(..., description="Summary title")
    bullet_points: list[str] = Field(default_factory=list, description="Extracted key points")


class FakeLLMProvider(BaseLLMProvider):
    """Fake provider for contract testing without external network dependencies."""

    def __init__(self, should_fail: bool = False) -> None:
        super().__init__(provider_name="fake_provider", default_model="fake-model-v1")
        self.should_fail = should_fail

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if self.should_fail:
            raise ProviderUnavailableError(
                message="Fake provider is currently down",
                provider_name=self.provider_name,
                status_code=503,
            )

        return LLMResponse(
            content="This is generated text from the fake provider.",
            model=request.model or self.default_model,
            provider=self.provider_name,
            usage=TokenUsage(prompt_tokens=15, completion_tokens=10, total_tokens=25),
            latency_ms=45.0,
        )

    async def generate_structured(self, request: LLMRequest, schema: Type[T]) -> tuple[T, LLMResponse]:
        if self.should_fail:
            raise ProviderUnavailableError(
                message="Fake provider error during structured gen",
                provider_name=self.provider_name,
                status_code=503,
            )

        mock_data = {
            "title": "Quarterly Highlights",
            "bullet_points": ["Revenue +12%", "Active users +25%"],
        }
        parsed = schema.model_validate(mock_data)
        response = LLMResponse(
            content=str(mock_data),
            structured_data=mock_data,
            model=request.model or self.default_model,
            provider=self.provider_name,
            usage=TokenUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35),
            latency_ms=60.0,
        )
        return parsed, response

    async def health_check(self) -> bool:
        return not self.should_fail


@pytest.mark.asyncio
async def test_fake_provider_text_generation():
    """Verify provider generation contract."""
    provider = FakeLLMProvider()
    req = LLMRequest(
        messages=[Message(role=MessageRole.USER, content="Hello AI")],
        temperature=0.5,
    )
    resp = await provider.generate(req)

    assert resp.provider == "fake_provider"
    assert resp.model == "fake-model-v1"
    assert resp.usage.total_tokens == 25
    assert "fake provider" in resp.content


@pytest.mark.asyncio
async def test_fake_provider_structured_generation():
    """Verify structured generation schema enforcement."""
    provider = FakeLLMProvider()
    req = LLMRequest(
        messages=[Message(role=MessageRole.USER, content="Extract quarterly highlights")],
    )
    parsed, resp = await provider.generate_structured(req, StructuredSummary)

    assert isinstance(parsed, StructuredSummary)
    assert parsed.title == "Quarterly Highlights"
    assert len(parsed.bullet_points) == 2
    assert resp.usage.total_tokens == 35


@pytest.mark.asyncio
async def test_fake_provider_failure_and_health():
    """Verify provider error mapping and health check."""
    provider = FakeLLMProvider(should_fail=True)
    assert await provider.health_check() is False

    req = LLMRequest(messages=[Message(role=MessageRole.USER, content="Test")])
    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.generate(req)

    assert exc_info.value.provider_name == "fake_provider"
    assert exc_info.value.retryable is True
