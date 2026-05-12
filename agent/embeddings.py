"""
Shared embedding functions used by both the RAG tool and the ingestion pipeline.

Kept in one place so model names and embedding logic are never duplicated.

Both dense and sparse encoders run locally via fastembed (ONNX Runtime).
No external API calls, no daemon — the model files are downloaded once into
the Python cache on first use, then run inline in this process.
"""

import os

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from agent.config import DENSE_MODEL, SPARSE_MODEL

_dense_encoder: TextEmbedding | None = None
_sparse_encoder: SparseTextEmbedding | None = None


def _dense() -> TextEmbedding:
    global _dense_encoder
    if _dense_encoder is None:
        _dense_encoder = TextEmbedding(model_name=DENSE_MODEL)
        # Warm-up: first call pays the model-load cost; absorb it now so the
        # first real query is fast.
        list(_dense_encoder.embed(["warmup"]))
    return _dense_encoder


def _sparse() -> SparseTextEmbedding:
    global _sparse_encoder
    if _sparse_encoder is None:
        _sparse_encoder = SparseTextEmbedding(model_name=SPARSE_MODEL)
        list(_sparse_encoder.embed(["warmup"]))
    return _sparse_encoder


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client from environment variables."""
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )


def dense_embed(text: str) -> list[float]:
    """Embed text with the configured dense model (default: BAAI/bge-small-en-v1.5,
    384-dim) via fastembed. Pure local inference — no API call."""
    result = list(_dense().embed([text]))[0]
    return result.tolist()


def sparse_embed(text: str) -> SparseVector:
    """Encode text with BM25 via fastembed (sparse vector for keyword matching)."""
    result = list(_sparse().embed([text]))[0]
    return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())
