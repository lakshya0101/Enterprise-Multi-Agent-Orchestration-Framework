"""Base LLM provider interface contract."""

from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

from enterprise_orchestrator.providers.models import LLMRequest, LLMResponse

T = TypeVar("T", bound=BaseModel)


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM service integrations."""

    def __init__(self, provider_name: str, default_model: str) -> None:
        self.provider_name = provider_name
        self.default_model = default_model

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Execute text generation request against the provider."""
        pass

    @abstractmethod
    async def generate_structured(self, request: LLMRequest, schema: Type[T]) -> tuple[T, LLMResponse]:
        """Execute generation and enforce parsing against a target Pydantic schema."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify connectivity and API key validity without incurring high token costs."""
        pass
