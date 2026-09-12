"""Manual smoke test utility for live Google Gemini and Groq API verification.

IMPORTANT NOTICE:
This script makes REAL network calls to the external provider APIs.
It consumes free-tier quota when executed.
It is intended ONLY for manual verification and should NOT be run as part of automated CI test suites.
"""

import asyncio
import os
import sys

# Ensure src is in sys.path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pydantic import BaseModel, Field

from enterprise_orchestrator.config.settings import FrameworkSettings
from enterprise_orchestrator.core.types import MessageRole
from enterprise_orchestrator.providers.gemini import GeminiProvider
from enterprise_orchestrator.providers.groq import GroqProvider
from enterprise_orchestrator.providers.models import LLMRequest, Message
from enterprise_orchestrator.providers.router import LLMRouter


class SmokeTestEvaluation(BaseModel):
    summary: str = Field(..., description="Short summary statement")
    status: str = Field(..., description="Execution status e.g. SUCCESS")


async def main() -> None:
    print("=" * 70)
    print("ENTERPRISE MULTI-AGENT ORCHESTRATION FRAMEWORK: LIVE PROVIDER SMOKE TEST")
    print("=" * 70)
    print("⚠️  NOTICE: This script makes REAL network requests using your configured API keys.")
    print("-" * 70)

    settings = FrameworkSettings()
    print(f"[*] Default Provider:   {settings.llm_provider}")
    print(f"[*] Fallback Provider:  {settings.fallback_provider}")
    print(f"[*] Gemini Model:       {settings.gemini_model}")
    print(f"[*] Groq Model:         {settings.groq_model}")
    print(f"[*] Gemini Key Present: {bool(settings.gemini_api_key)}")
    print(f"[*] Groq Key Present:   {bool(settings.groq_api_key)}")
    print("-" * 70)

    # 1. Test Gemini Provider
    if settings.gemini_api_key:
        print("\n[1/3] Testing Google Gemini Provider (Text & Structured Output)...")
        try:
            gemini = GeminiProvider()
            req = LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You are a concise enterprise assistant."),
                    Message(role=MessageRole.USER, content="Reply with exactly: 'Gemini live test verified'"),
                ],
                model=settings.gemini_model,
            )
            resp = await gemini.generate(req)
            print(f"  [+] Text Generation Success! Provider: {resp.provider} | Model: {resp.model}")
            print(f"  [+] Response: {resp.content.strip()}")
            print(f"  [+] Token Usage: {resp.usage.model_dump()} | Latency: {resp.latency_ms:.1f}ms")

            print("  [>] Testing Gemini Structured Output...")
            struct_req = LLMRequest(
                messages=[Message(role=MessageRole.USER, content="Provide summary='Gemini OK', status='SUCCESS'")]
            )
            parsed, s_resp = await gemini.generate_structured(struct_req, SmokeTestEvaluation)
            print(f"  [+] Structured Output Success! Parsed: {parsed.model_dump()}")
        except Exception as e:
            print(f"  [-] Gemini Test Failed: {e}")
    else:
        print("\n[1/3] Skipping Gemini: GEMINI_API_KEY is not set.")

    # 2. Test Groq Provider
    if settings.groq_api_key:
        print("\n[2/3] Testing Groq Provider (Text & Structured Output)...")
        try:
            groq = GroqProvider()
            req = LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You are a concise assistant."),
                    Message(role=MessageRole.USER, content="Reply with exactly: 'Groq live test verified'"),
                ],
                model=settings.groq_model,
            )
            resp = await groq.generate(req)
            print(f"  [+] Text Generation Success! Provider: {resp.provider} | Model: {resp.model}")
            print(f"  [+] Response: {resp.content.strip()}")
            print(f"  [+] Token Usage: {resp.usage.model_dump()} | Latency: {resp.latency_ms:.1f}ms")

            print("  [>] Testing Groq Structured Output...")
            struct_req = LLMRequest(
                messages=[Message(role=MessageRole.USER, content="Provide summary='Groq OK', status='SUCCESS'")]
            )
            parsed, s_resp = await groq.generate_structured(struct_req, SmokeTestEvaluation)
            print(f"  [+] Structured Output Success! Parsed: {parsed.model_dump()}")
        except Exception as e:
            print(f"  [-] Groq Test Failed: {e}")
    else:
        print("\n[2/3] Skipping Groq: GROQ_API_KEY is not set.")

    # 3. Test LLM Router
    print("\n[3/3] Testing LLMRouter Dispatch...")
    providers = {}
    if settings.gemini_api_key:
        providers["gemini"] = GeminiProvider()
    if settings.groq_api_key:
        providers["groq"] = GroqProvider()

    if providers:
        try:
            router = LLMRouter(providers=providers)
            req = LLMRequest(messages=[Message(role=MessageRole.USER, content="Say 'Router OK'")])
            r_resp = await router.generate(req)
            print(f"  [+] Router Dispatched Successfully! Active Provider: {r_resp.provider}")
            print(f"  [+] Content: {r_resp.content.strip()}")
        except Exception as e:
            print(f"  [-] Router Test Failed: {e}")
    else:
        print("  [-] No providers configured for router test.")

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
