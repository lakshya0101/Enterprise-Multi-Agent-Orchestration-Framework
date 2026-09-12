"""LLM Router managing multi-provider dispatch, fallback resilience, and load balancing."""

from typing import Dict, List, Optional, Set, Type, TypeVar

from pydantic import BaseModel

from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.errors.exceptions import (
    ConfigurationError,
    OrchestratorError,
    ProviderError,
    ProviderUnavailableError,
    TimeoutError as OrchestratorTimeoutError,
    ValidationError,
)
from enterprise_orchestrator.providers.base import BaseLLMProvider
from enterprise_orchestrator.providers.models import LLMRequest, LLMResponse

T = TypeVar("T", bound=BaseModel)


class LLMRouter:
    """Central router for resilient LLM dispatch across primary and fallback providers."""

    def __init__(
        self,
        providers: Optional[Dict[str, BaseLLMProvider]] = None,
        default_provider: Optional[str] = None,
        fallback_provider: Optional[str] = None,
        enable_fallback: Optional[bool] = None,
    ) -> None:
        settings = FrameworkSettings()
        self._providers: Dict[str, BaseLLMProvider] = providers or {}
        self.default_provider = default_provider or settings.llm_provider or "gemini"
        self.fallback_provider = fallback_provider or settings.fallback_provider or "groq"
        self.enable_fallback = enable_fallback if enable_fallback is not None else settings.llm_enable_fallback

    def register_provider(self, provider: BaseLLMProvider) -> None:
        """Register a provider instance with the router."""
        self._providers[provider.provider_name] = provider

    def get_provider(self, name: str) -> Optional[BaseLLMProvider]:
        """Fetch a registered provider by name."""
        return self._providers.get(name)

    def list_providers(self) -> List[str]:
        """List names of all registered providers."""
        return list(self._providers.keys())

    def _is_retryable_failure(self, error: Exception) -> bool:
        """Determine if an exception warrants attempting a fallback provider."""
        if isinstance(error, ProviderUnavailableError):
            return True
        if isinstance(error, OrchestratorTimeoutError):
            return True
        if isinstance(error, ProviderError) and error.retryable:
            return True
        return False

    async def generate(
        self,
        request: LLMRequest,
        provider_name: Optional[str] = None,
    ) -> LLMResponse:
        """Route generation request to target provider with automatic fallback on retryable failures."""
        primary_name = provider_name or self.default_provider
        attempted_providers: Set[str] = set()
        current_target = primary_name
        fallback_history: List[Dict[str, str]] = []

        while current_target and current_target not in attempted_providers:
            attempted_providers.add(current_target)
            provider = self.get_provider(current_target)

            if not provider:
                raise ConfigurationError(
                    message=f"Requested LLM provider '{current_target}' is not registered in LLMRouter.",
                    code="PROVIDER_NOT_FOUND",
                    retryable=False,
                )

            try:
                response = await provider.generate(request)

                # If fallback was used, annotate response metadata
                if len(attempted_providers) > 1:
                    response.raw_response = response.raw_response or {}
                    response.raw_response["router_metadata"] = {
                        "fallback_used": True,
                        "primary_provider": primary_name,
                        "active_provider": current_target,
                        "fallback_history": fallback_history,
                    }
                return response

            except Exception as e:
                # If error is non-retryable (e.g. ValidationError, ConfigurationError, non-retryable Auth error), do not fallback
                if not self._is_retryable_failure(e) or not self.enable_fallback:
                    raise

                error_summary = str(e)
                fallback_history.append({"failed_provider": current_target, "error": error_summary})

                # Determine next candidate provider
                if current_target == primary_name and self.fallback_provider and self.fallback_provider != primary_name:
                    current_target = self.fallback_provider
                else:
                    # Find any other available registered provider not yet attempted
                    candidates = [p for p in self._providers if p not in attempted_providers]
                    current_target = candidates[0] if candidates else None

        # If all attempted providers failed
        failed_summary = "; ".join(f"[{item['failed_provider']}: {item['error']}]" for item in fallback_history)
        raise ProviderError(
            message=f"All available LLM providers failed. Attempted: {list(attempted_providers)}. Errors: {failed_summary}",
            provider_name=primary_name,
            code="ALL_PROVIDERS_FAILED",
            retryable=True,
            details={"attempted_providers": list(attempted_providers), "history": fallback_history},
        )

    async def generate_structured(
        self,
        request: LLMRequest,
        schema: Type[T],
        provider_name: Optional[str] = None,
    ) -> tuple[T, LLMResponse]:
        """Route structured generation request with fallback protection."""
        primary_name = provider_name or self.default_provider
        attempted_providers: Set[str] = set()
        current_target = primary_name
        fallback_history: List[Dict[str, str]] = []

        while current_target and current_target not in attempted_providers:
            attempted_providers.add(current_target)
            provider = self.get_provider(current_target)

            if not provider:
                raise ConfigurationError(
                    message=f"Requested LLM provider '{current_target}' is not registered in LLMRouter.",
                    code="PROVIDER_NOT_FOUND",
                    retryable=False,
                )

            try:
                parsed_model, response = await provider.generate_structured(request, schema)

                if len(attempted_providers) > 1:
                    response.raw_response = response.raw_response or {}
                    response.raw_response["router_metadata"] = {
                        "fallback_used": True,
                        "primary_provider": primary_name,
                        "active_provider": current_target,
                        "fallback_history": fallback_history,
                    }
                return parsed_model, response

            except Exception as e:
                # Do not fallback on schema validation programming errors if not retryable
                if not self._is_retryable_failure(e) or not self.enable_fallback:
                    raise

                error_summary = str(e)
                fallback_history.append({"failed_provider": current_target, "error": error_summary})

                if current_target == primary_name and self.fallback_provider and self.fallback_provider != primary_name:
                    current_target = self.fallback_provider
                else:
                    candidates = [p for p in self._providers if p not in attempted_providers]
                    current_target = candidates[0] if candidates else None

        failed_summary = "; ".join(f"[{item['failed_provider']}: {item['error']}]" for item in fallback_history)
        raise ProviderError(
            message=f"All available LLM providers failed for structured output. Attempted: {list(attempted_providers)}. Errors: {failed_summary}",
            provider_name=primary_name,
            code="ALL_PROVIDERS_FAILED",
            retryable=True,
            details={"attempted_providers": list(attempted_providers), "history": fallback_history},
        )
