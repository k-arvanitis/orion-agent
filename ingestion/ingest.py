"""
Embed and ingest document chunks into Qdrant Cloud with hybrid search support.

Each chunk gets two vectors:
  - dense:  nomic-embed-text via Ollama (768-dim, semantic similarity)
  - sparse: BM25 via fastembed (variable-dim, keyword matching)

At query time, Qdrant runs both searches independently, then fuses the ranked
results using Reciprocal Rank Fusion (RRF) — giving the best of semantic and
keyword retrieval.

Usage:
    uv run python ingest.py
    uv run python ingest.py --chunks data/output/document-chunks.json --collection orion-policies
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.config import DENSE_MODEL, DENSE_DIM, SPARSE_MODEL, QDRANT_COLLECTION

DEFAULT_COLLECTION = QDRANT_COLLECTION
DEFAULT_CHUNKS_FILE = "data/output/document-chunks.json"

# Loaded once at module level — model download happens on first use
_sparse_encoder = SparseTextEmbedding(model_name=SPARSE_MODEL)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def dense_embed(text: str) -> list[float]:
    return ollama.embed(model=DENSE_MODEL, input=text).embeddings[0]


def sparse_embed(text: str) -> SparseVector:
    result = list(_sparse_encoder.embed([text]))[0]
    return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------


def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        logger.error("QDRANT_URL and QDRANT_API_KEY must be set in .env")
        sys.exit(1)
    return QdrantClient(url=url, api_key=api_key)


def recreate_collection(client: QdrantClient, collection: str) -> None:
    """Drop and recreate with named dense + sparse vector configs."""
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        client.delete_collection(collection)
        logger.info("Dropped existing collection '%s'", collection)

    client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    logger.info("Created collection '%s' with dense + sparse vectors", collection)


def ingest(chunks: list[dict], client: QdrantClient, collection: str) -> None:
    total = len(chunks)
    points: list[PointStruct] = []

    for i, chunk in enumerate(chunks, start=1):
        logger.debug("Embedding %d/%d: %s", i, total, chunk["heading"])
        print(f"  Embedding {i}/{total}: {chunk['heading']}", end="\r")
        points.append(
            PointStruct(
                id=i,
                vector={
                    "dense": dense_embed(chunk["content"]),
                    "sparse": sparse_embed(chunk["content"]),
                },
                payload={
                    "source": chunk["source"],
                    "doc_title": chunk["doc_title"],
                    "section": chunk["section"],
                    "heading": chunk["heading"],
                    "content": chunk["content"],
                },
            )
        )

    print()
    client.upsert(collection_name=collection, points=points)
    logger.info("Upserted %d points into '%s'", len(points), collection)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Embed and push document chunks into Qdrant (dense + sparse hybrid).",
    )
    parser.add_argument(
        "--chunks",
        metavar="FILE",
        type=Path,
        default=Path(DEFAULT_CHUNKS_FILE),
        help=f"Path to chunks JSON file (default: {DEFAULT_CHUNKS_FILE}).",
    )
    parser.add_argument(
        "--collection",
        metavar="NAME",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION}).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.chunks.exists():
        parser.error(f"Chunks file '{args.chunks}' not found. Run the chunker first.")

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from '{args.chunks}'\n")

    client = get_client()
    recreate_collection(client, args.collection)

    print(f"\nEmbedding and ingesting into '{args.collection}' ...\n")
    ingest(chunks, client, args.collection)

    print(f"\nDone. {len(chunks)} chunks ingested into '{args.collection}'.")


if __name__ == "__main__":
    main()
