"""
Embed and ingest document chunks into Qdrant (local or Cloud) with hybrid search.

Each chunk gets two vectors:
  - dense:  fastembed BAAI/bge-small-en-v1.5 (384-dim, semantic similarity)
  - sparse: BM25 via fastembed (variable-dim, keyword matching)

At query time, Qdrant runs both searches independently, then fuses the ranked
results using Reciprocal Rank Fusion (RRF) — giving the best of semantic and
keyword retrieval.

Incremental by default: each chunk's point ID is a stable hash of its own
`id` field (not a position), and each point's payload carries a content hash.
A chunk whose content hasn't changed since the last run is skipped instead of
re-embedded, and a chunk whose source document disappeared gets its point
deleted — so re-running ingest after editing one document doesn't re-embed
or re-download anything else. Pass --rebuild to drop and recreate instead.

Usage:
    uv run python ingest.py
    uv run python ingest.py --chunks data/output/document-chunks.json \
--collection orion-policies
    uv run python ingest.py --chunks-dir data/output/
    uv run python ingest.py --rebuild
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

logger = logging.getLogger(__name__)

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orion_agent.agent.config import DENSE_DIM, QDRANT_COLLECTION  # noqa: E402
from orion_agent.agent.embeddings import (  # noqa: E402
    dense_embed,
    get_qdrant_client,
    sparse_embed,
)

DEFAULT_COLLECTION = QDRANT_COLLECTION
DEFAULT_CHUNKS_FILE = "data/output/document-chunks.json"

# Fixed, arbitrary namespace so uuid5(id) is stable across processes and runs.
_POINT_ID_NAMESPACE = uuid.UUID("c9c5b1f6-3b8b-4e21-9a9b-6a9a6f6b6a63")


def _point_id(chunk_id: str) -> str:
    """Deterministic point ID from a chunk's own id — stable across ingest
    runs regardless of chunk ordering or how many chunks exist."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------


def get_client() -> QdrantClient:
    if not os.getenv("QDRANT_URL"):
        logger.error("QDRANT_URL must be set in .env")
        sys.exit(1)
    return get_qdrant_client()


def _create_collection(client: QdrantClient, collection: str) -> None:
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
        },
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    logger.info("Created collection '%s' with dense + sparse vectors", collection)


def recreate_collection(client: QdrantClient, collection: str) -> None:
    """Drop and recreate with named dense + sparse vector configs."""
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        client.delete_collection(collection)
        logger.info("Dropped existing collection '%s'", collection)
    _create_collection(client, collection)


def _ensure_collection(client: QdrantClient, collection: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        _create_collection(client, collection)


def _existing_hashes(client: QdrantClient, collection: str) -> dict[str, str]:
    """Return {point_id: content_hash} for every point currently stored."""
    hashes: dict[str, str] = {}
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=collection,
            with_payload=["content_hash"],
            with_vectors=False,
            limit=256,
            offset=next_page,
        )
        for point in points:
            hashes[str(point.id)] = (point.payload or {}).get("content_hash", "")
        if next_page is None:
            break
    return hashes


def ingest(chunks: list[dict], client: QdrantClient, collection: str) -> None:
    """Embed only chunks whose content changed since the last run, upsert
    them, and delete points for chunks that no longer exist in *chunks*."""
    _ensure_collection(client, collection)
    existing_hashes = _existing_hashes(client, collection)

    new_ids = {_point_id(c["id"]) for c in chunks}
    stale_ids = [pid for pid in existing_hashes if pid not in new_ids]
    if stale_ids:
        client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=stale_ids),
        )
        logger.info(
            "Deleted %d stale point(s) no longer in the chunk set", len(stale_ids)
        )

    to_embed = []
    for chunk in chunks:
        pid = _point_id(chunk["id"])
        content_hash = _content_hash(chunk["content"])
        if existing_hashes.get(pid) == content_hash:
            continue
        to_embed.append((pid, content_hash, chunk))

    total = len(to_embed)
    if total == 0:
        print(f"  0 chunks changed — nothing to embed ({len(chunks)} unchanged)")
        return

    points: list[PointStruct] = []
    for i, (pid, content_hash, chunk) in enumerate(to_embed, start=1):
        logger.debug("Embedding %d/%d: %s", i, total, chunk["heading"])
        print(f"  Embedding {i}/{total}: {chunk['heading']}", end="\r")
        points.append(
            PointStruct(
                id=pid,
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
                    "content_hash": content_hash,
                },
            )
        )

    print()
    client.upsert(collection_name=collection, points=points)
    logger.info(
        "Upserted %d changed point(s), skipped %d unchanged, into '%s'",
        len(points),
        len(chunks) - len(points),
        collection,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description=(
            "Embed and push document chunks into Qdrant (dense + sparse hybrid)."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--chunks",
        metavar="FILE",
        type=Path,
        default=Path(DEFAULT_CHUNKS_FILE),
        help=f"Path to a single chunks JSON file (default: {DEFAULT_CHUNKS_FILE}).",
    )
    group.add_argument(
        "--chunks-dir",
        metavar="DIR",
        type=Path,
        help=(
            "Directory of JSON chunk files — all *.json files are merged and ingested."
        ),
    )
    parser.add_argument(
        "--collection",
        metavar="NAME",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the collection instead of an incremental upsert.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.chunks_dir:
        json_files = sorted(args.chunks_dir.glob("*.json"))
        if not json_files:
            parser.error(f"No JSON files found in '{args.chunks_dir}'.")
        chunks = []
        for f in json_files:
            chunks.extend(json.loads(f.read_text(encoding="utf-8")))
        print(
            f"Loaded {len(chunks)} chunks from {len(json_files)} files "
            f"in '{args.chunks_dir}'\n"
        )
    else:
        if not args.chunks.exists():
            parser.error(
                f"Chunks file '{args.chunks}' not found. Run the chunker first."
            )
        chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
        print(f"Loaded {len(chunks)} chunks from '{args.chunks}'\n")

    client = get_client()
    if args.rebuild:
        recreate_collection(client, args.collection)

    print(f"\nSyncing '{args.collection}' ...\n")
    ingest(chunks, client, args.collection)

    print(f"\nDone. {len(chunks)} chunks in the input set for '{args.collection}'.")


if __name__ == "__main__":
    main()
