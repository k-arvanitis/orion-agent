"""
RAG tool — hybrid semantic + keyword search over ShopNova policy documents.

Query flow:
  1. Embed the query with nomic-embed-text (dense, semantic).
  2. Encode the query with BM25/fastembed (sparse, keyword).
  3. Run both searches in parallel via Qdrant prefetch.
  4. Fuse results with Reciprocal Rank Fusion (RRF).
  5. Return top-k chunks with source citations.

Why hybrid?
  Policy docs contain exact terms (e.g. "30-day return window", "Boleto",
  "CPF") that pure semantic search can miss or rank poorly. BM25 catches
  exact keyword matches; the dense model handles paraphrase and intent.
  RRF merges both ranked lists without needing a learned weighting.
"""

import os

import ollama
from fastembed import SparseTextEmbedding
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

_qdrant_client: QdrantClient | None = None
_sparse_encoder: SparseTextEmbedding | None = None

COLLECTION = "orion-policies"
DENSE_MODEL = "nomic-embed-text"
SPARSE_MODEL = "Qdrant/bm25"
TOP_K = 4
PREFETCH_K = 20  # candidates per searcher before fusion


def _qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
    return _qdrant_client


def _sparse() -> SparseTextEmbedding:
    global _sparse_encoder
    if _sparse_encoder is None:
        _sparse_encoder = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_encoder


def _dense_embed(text: str) -> list[float]:
    return ollama.embed(model=DENSE_MODEL, input=text).embeddings[0]


def _sparse_embed(text: str) -> SparseVector:
    result = list(_sparse().embed([text]))[0]
    return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())


@tool
def search_policies(query: str) -> str:
    """
    Search ShopNova's policy documents for information about returns, warranties,
    shipping rules, payment terms, or any other store policy.

    Uses hybrid search: semantic similarity (dense) + keyword matching (sparse),
    fused with Reciprocal Rank Fusion for best coverage.

    Args:
        query: The customer's question or the topic to search for.

    Returns:
        Relevant policy excerpts with source citations.
    """
    dense_vector = _dense_embed(query)
    sparse_vector = _sparse_embed(query)

    result = _qdrant().query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=PREFETCH_K),
            Prefetch(query=sparse_vector, using="sparse", limit=PREFETCH_K),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=TOP_K,
        with_payload=True,
    )

    if not result.points:
        return "No relevant policy information found."

    results = []
    for hit in result.points:
        p = hit.payload or {}
        source = p.get("source", "unknown")
        heading = p.get("heading", p.get("section", ""))
        content = p.get("content", "")
        header = f"[{source} — {heading}]" if heading else f"[{source}]"
        results.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(results)
