"""
Central configuration — all tuneable values in one place.

Every constant can be overridden by setting the matching environment variable.
This avoids hardcoded model names scattered across files.
"""

import os

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

# Groq model used for the agent and eval judge.
# Override with: AGENT_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

# Dense embedding model (Ollama) — must match what was used during ingestion.
DENSE_MODEL: str = os.getenv("DENSE_MODEL", "nomic-embed-text")
DENSE_DIM: int = 768

# Sparse embedding model (fastembed BM25) — must match ingestion.
SPARSE_MODEL: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "orion-policies")

# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

# Operator email that receives escalation alerts.
# Override with: OPERATOR_EMAIL=real-team@yourcompany.com
OPERATOR_EMAIL: str = os.getenv("OPERATOR_EMAIL", "support-team@shopnova.com.br")
