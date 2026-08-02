"""Small local Qdrant index used by the key-free support demo.

The production agent uses the configured hybrid Qdrant collection. The demo
keeps a local Qdrant collection over the same bundled policy chunks so policy
retrieval remains real and repeatable without cloud credentials.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from threading import Lock

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

VECTOR_SIZE = 384
COLLECTION_NAME = "orion-demo-policies"
CHUNKS_PATH = Path(__file__).resolve().parents[1] / "data/output/document-chunks.json"

_client: QdrantClient | None = None
_lock = Lock()

_SYNONYMS = {
    "back": ("return",),
    "broken": ("defect", "warranty"),
    "cancel": ("refund", "return"),
    "delivery": ("shipping",),
    "exchange": ("return",),
    "faulty": ("defect", "warranty"),
    "late": ("delay", "shipping"),
    "money": ("refund", "payment"),
    "package": ("parcel", "shipping"),
    "send": ("return",),
}


def _terms(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(_SYNONYMS.get(token, ()))
        if len(token) > 5 and token.endswith("ing"):
            expanded.append(token[:-3])
        elif len(token) > 4 and token.endswith("s"):
            expanded.append(token[:-1])
    return expanded


def _embed(text: str) -> list[float]:
    """Create a deterministic lexical embedding for local vector retrieval."""
    vector = [0.0] * VECTOR_SIZE
    for term, count in Counter(_terms(text)).items():
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % VECTOR_SIZE
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _build_index() -> QdrantClient:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=index,
                vector=_embed(
                    " ".join(
                        (
                            chunk.get("doc_title", ""),
                            chunk.get("section", ""),
                            chunk.get("heading", ""),
                            chunk.get("content", ""),
                        )
                    )
                ),
                payload=chunk,
            )
            for index, chunk in enumerate(chunks, start=1)
        ],
    )
    return client


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = _build_index()
    return _client


def search_policy_documents(query: str, limit: int = 3) -> list[dict]:
    """Return the highest-scoring policy chunks from the local Qdrant index."""
    normalized = query.lower()
    enrichment = ""
    if "return" in normalized or "refund" in normalized or "send back" in normalized:
        enrichment = " general return window standard return period eligibility"
    elif "warranty" in normalized or "broken" in normalized or "defect" in normalized:
        enrichment = " warranty coverage standard manufacturing defect covered period"
    elif "shipping" in normalized or "delivery" in normalized or "parcel" in normalized:
        enrichment = " shipping methods standard delivery timeframe tracking"
    elif "payment" in normalized or "charge" in normalized or "pay" in normalized:
        enrichment = " payment methods authorization refund processing"
    results = _get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=_embed(query + enrichment),
        limit=limit,
        with_payload=True,
        score_threshold=0.04,
    )
    return [
        {
            "source": str((point.payload or {}).get("source", "unknown")),
            "heading": str((point.payload or {}).get("heading", "Policy")),
            "content": str((point.payload or {}).get("content", "")),
            "score": round(float(point.score), 3),
        }
        for point in results.points
    ]
