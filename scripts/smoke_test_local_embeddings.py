"""Explicit standalone smoke test for LocalSentenceTransformerEmbeddingProvider.

NOTE: This script is intended for manual execution only and is intentionally not run
during pytest to prevent unexpected network calls or model downloads.

Usage:
    python scripts/smoke_test_local_embeddings.py
"""

import asyncio
import sys

from enterprise_orchestrator.retrieval.embeddings import LocalSentenceTransformerEmbeddingProvider


async def main() -> None:
    print("--- Smoke Testing LocalSentenceTransformerEmbeddingProvider ---")
    model_name = "all-MiniLM-L6-v2"
    print(f"Initializing provider with model: {model_name}...")
    
    provider = LocalSentenceTransformerEmbeddingProvider(model_name=model_name)
    
    query = "Enterprise agent governance and compliance"
    print(f"Embedding query: '{query}'...")
    vec = await provider.embed_text(query)
    
    print(f"Success! Generated vector of dimension: {len(vec)}")
    print(f"Sample values (first 5): {vec[:5]}")
    
    docs = [
        "First document on financial regulation.",
        "Second document on software architecture.",
    ]
    print(f"Embedding batch of {len(docs)} documents...")
    doc_vecs = await provider.embed_documents(docs)
    print(f"Success! Generated {len(doc_vecs)} document vectors of dimension {len(doc_vecs[0])}.")
    print("Local SentenceTransformer smoke test completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
