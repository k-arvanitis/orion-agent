"""
RAG tool — hybrid semantic + keyword search over ShopNova policy documents.

Query flow:
  1. Embed the query with nomic-embed-text (dense, semantic).
  2. Encode the query with BM25/fastembed (sparse, keyword).
  3. Run both searches in parallel via Qdrant prefetch.
  4. Fuse results with Reciprocal Rank Fusion (RRF).
  5. Return a structured response: answer text + raw chunks for the trace panel.

Why hybrid?
  Policy docs contain exact terms (e.g. "30-day return window", "Boleto",
  "CPF") that pure semantic search can miss or rank poorly. BM25 catches
  exact keyword matches; the dense model handles paraphrase and intent.
  RRF merges both ranked lists without needing a learned weighting.

Return format:
  JSON string: {"answer": "<text for LLM>", "chunks": [{"source", "heading", "content"}]}
  The graph's tools_node puts "answer" into ToolMessage.content (what the LLM sees)
  and stores "chunks" in graph state (what the UI trace panel shows).
"""

import json
import logging
import os

from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from agent.config import QDRANT_COLLECTION
from agent.embeddings import dense_embed as _dense_embed
from agent.embeddings import get_qdrant_client
from agent.embeddings import sparse_embed as _sparse_embed

logger = logging.getLogger(__name__)

_qdrant_client: QdrantClient | None = None

TOP_K = 4
PREFETCH_K = 20
MAX_CHUNK_CHARS = 600


def _qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = get_qdrant_client()
    return _qdrant_client


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
        JSON string with 'answer' (text for the LLM) and 'chunks' (source metadata).
    """
    logger.info("RAG search: %r", query)

    try:
        dense_vector = _dense_embed(query)
        sparse_vector = _sparse_embed(query)

        result = _qdrant().query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=PREFETCH_K),
                Prefetch(query=sparse_vector, using="sparse", limit=PREFETCH_K),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=TOP_K,
            with_payload=True,
        )
    except Exception:
        logger.error("RAG search failed", exc_info=True)
        return json.dumps({
            "answer": "Policy search is temporarily unavailable. Try the database for order information, or ask the customer to contact support.",
            "chunks": [],
        })

    if not result.points:
        logger.debug("RAG search returned no results for: %r", query)
        return json.dumps({"answer": "No relevant policy information found.", "chunks": []})

    chunks = []
    answer_parts = []
    for hit in result.points:
        p = hit.payload or {}
        source = p.get("source", "unknown")
        heading = p.get("heading", p.get("section", ""))
        content = p.get("content", "")
        header = f"[{source} — {heading}]" if heading else f"[{source}]"
        chunks.append({"source": source, "heading": heading, "content": content})
        answer_parts.append(f"{header}\n{content[:MAX_CHUNK_CHARS]}")

    logger.debug("RAG returned %d chunks", len(chunks))
    return json.dumps({
        "answer": "\n\n---\n\n".join(answer_parts),
        "chunks": chunks,
    })
