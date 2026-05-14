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
# Override with: AGENT_MODEL=openai/gpt-oss-120b
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "qwen/qwen3-32b")

# Reasoning models on Groq (Qwen 3, gpt-oss) emit <think>...</think> blocks
# unless reasoning_format is set. Non-reasoning models reject the parameter,
# so it must only be passed when applicable.
_REASONING_MODEL_PREFIXES = ("qwen/qwen", "openai/gpt-oss")


def chat_groq_kwargs() -> dict:
    """Return extra ChatGroq kwargs required for the configured AGENT_MODEL."""
    if any(AGENT_MODEL.startswith(p) for p in _REASONING_MODEL_PREFIXES):
        return {"reasoning_format": "hidden"}
    return {}

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

# Dense embedding model (fastembed, local) — must match what was used during
# ingestion. fastembed downloads the model into the Python cache on first use
# and runs inference inline via ONNX Runtime — no daemon, no API call, no key.
DENSE_MODEL: str = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
DENSE_DIM: int = 384

# Sparse embedding model (fastembed BM25) — must match ingestion.
SPARSE_MODEL: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "orion-policies")

# ---------------------------------------------------------------------------
# Voice I/O (used by the Streamlit UI's optional voice mode)
# ---------------------------------------------------------------------------

# Groq Whisper variant — turbo for sub-second short-clip transcription.
VOICE_TRANSCRIBE_MODEL: str = os.getenv(
    "VOICE_TRANSCRIBE_MODEL", "whisper-large-v3-turbo"
)

# ElevenLabs TTS model — turbo v2.5 minimises end-to-end response latency.
VOICE_TTS_MODEL: str = os.getenv("VOICE_TTS_MODEL", "eleven_turbo_v2_5")

# ElevenLabs voice ID. Default: "Sarah" — clear, neutral US English.
VOICE_TTS_VOICE_ID: str = os.getenv("VOICE_TTS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

# Operator email that receives escalation alerts.
# Override with: OPERATOR_EMAIL=real-team@yourcompany.com
OPERATOR_EMAIL: str = os.getenv("OPERATOR_EMAIL", "support-team@shopnova.com.br")
