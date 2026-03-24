"""
Shared embedding functions used by both the RAG tool and the ingestion pipeline.

Kept in one place so model names and embedding logic are never duplicated.
"""

import os

import ollama
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from agent.config import DENSE_MODEL, SPARSE_MODEL

_sparse_encoder: SparseTextEmbedding | None = None


def _sparse() -> SparseTextEmbedding:
    global _sparse_encoder
    if _sparse_encoder is None:
        _sparse_encoder = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_encoder


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client from environment variables."""
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )


def dense_embed(text: str) -> list[float]:
    """Embed text with nomic-embed-text via Ollama (768-dim dense vector)."""
    return ollama.embed(model=DENSE_MODEL, input=text).embeddings[0]


def sparse_embed(text: str) -> SparseVector:
    """Encode text with BM25 via fastembed (sparse vector for keyword matching)."""
    result = list(_sparse().embed([text]))[0]
    return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())
