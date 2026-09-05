"""
Central configuration — all tuneable values in one place.

Every constant can be overridden by setting the matching environment variable.
This avoids hardcoded model names scattered across files.
"""

import os

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

# Chat model used for the agent and SQL generation.
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "qwen/qwen3-235b-a22b-2507")

# OpenRouter is preferred when its key is configured. Existing Groq-only
# installations continue to work without changing their environment.
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_PROVIDER: str = os.getenv(
    "LLM_PROVIDER", "openrouter" if OPENROUTER_API_KEY else "groq"
).lower()
LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://openrouter.ai/api/v1")
LLM_REQUEST_TIMEOUT_S: float = float(os.getenv("LLM_REQUEST_TIMEOUT_S", "60"))
OPENROUTER_PROVIDER_PIN: str = os.getenv("OPENROUTER_PROVIDER_PIN", "")

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
# Voice I/O
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
# Checkpointing
# ---------------------------------------------------------------------------

# Path to the SQLite file that persists conversation state across restarts.
# Set CHECKPOINT_DB_PATH to an absolute path in production.
CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.db")

# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

# Incoming webhook URL for the operator Slack channel. Optional — if unset,
# escalate_to_human still works, it just doesn't notify anyone externally.
SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")
