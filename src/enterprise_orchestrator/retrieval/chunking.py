"""Document chunking models and implementations for semantic RAG."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from enterprise_orchestrator.agents.retrieval import Document


class DocumentChunk(BaseModel):
    """Normalized chunk of a parent document for embedding and indexing."""

    id: str = Field(..., description="Deterministic unique identifier for this chunk")
    document_id: str = Field(..., description="ID of the source parent document")
    chunk_index: int = Field(..., ge=0, description="Sequential position index within the document")
    content: str = Field(..., min_length=1, description="Textual body content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Preserved and chunk-level metadata")


class BaseDocumentChunker(ABC):
    """Abstract interface for segmenting documents into indexable chunks."""

    @abstractmethod
    async def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a single document into chunks."""
        pass

    async def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """Split multiple documents into chunks preserving order."""
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            chunks = await self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks


class SimpleTextChunker(BaseDocumentChunker):
    """Deterministic character/word boundary chunker with sliding overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split document content into overlapping text chunks with deterministic IDs."""
        text = document.content.strip() if document.content else ""
        if not text:
            return []

        # Short document case: fits in a single chunk
        if len(text) <= self.chunk_size:
            chunk_metadata = dict(document.metadata)
            chunk_metadata.update({
                "source_doc_id": document.id,
                "chunk_index": 0,
                "start_char": 0,
                "end_char": len(text),
            })
            return [
                DocumentChunk(
                    id=f"{document.id}_chunk_0",
                    document_id=document.id,
                    chunk_index=0,
                    content=text,
                    metadata=chunk_metadata,
                )
            ]

        chunks: List[DocumentChunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = dict(document.metadata)
                chunk_metadata.update({
                    "source_doc_id": document.id,
                    "chunk_index": chunk_idx,
                    "start_char": start,
                    "end_char": end,
                })
                chunks.append(
                    DocumentChunk(
                        id=f"{document.id}_chunk_{chunk_idx}",
                        document_id=document.id,
                        chunk_index=chunk_idx,
                        content=chunk_text,
                        metadata=chunk_metadata,
                    )
                )
                chunk_idx += 1

            if end == len(text):
                break
            start += step

        return chunks
