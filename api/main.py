"""
FastAPI surface for the Orion agent.

Endpoints:
  GET  /api/health      → liveness probe + per-dependency status (llm key, qdrant, db)
  GET  /api/support/*   → database-backed customers, products, and conversations
  POST /api/chat        → streamed NDJSON: token events + final trace event
  POST /api/transcribe  → multipart audio upload → {"text": "..."}
  POST /api/tts         → {"text": "..."} → audio/mpeg bytes

The agent core (LangGraph + tools + guard) is unchanged. This module is a thin
HTTP wrapper around `agent.graph.graph` and `agent.voice`.

Per-session conversation history is held by LangGraph's SQLite checkpointer,
keyed by `session_id` (passed as `thread_id`).
"""

import json
import logging
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import text

load_dotenv()

from api.schemas import (  # noqa: E402
    ChatRequest,
    CustomerMessageRequest,
    HealthResponse,
    SupportReplyRequest,
    TranscribeResponse,
    TtsRequest,
)
from api.support_store import (  # noqa: E402
    delete_conversation,
    get_demo_overview,
    list_conversations,
    list_customers,
    list_products,
    lookup_customer,
    mark_conversation_read,
    reset_demo_conversations,
    resolve_conversation_by_support,
    send_customer_message,
    send_support_reply,
)
from orion_agent.agent import voice  # noqa: E402

logger = logging.getLogger(__name__)

# Guard correction marker — must match agent/graph.py guard_node.
GUARD_CORRECTION_MARKER = "Rewrite using only data"

app = FastAPI(title="Orion Agent API", version="0.1.0")


@lru_cache(maxsize=1)
def _get_agent_graph():
    """Load provider-backed agent dependencies only when /api/chat is used."""
    from orion_agent.agent.graph import graph

    return graph

# CORS — Next.js dev server on :3500; API on :8088. Override via env for prod.
_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3500,http://127.0.0.1:3500",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


def _check_llm_key() -> bool:
    from orion_agent.agent.config import GROQ_API_KEY, OPENROUTER_API_KEY

    return bool(OPENROUTER_API_KEY or GROQ_API_KEY)


def _check_qdrant() -> bool:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ.get("QDRANT_API_KEY") or None,
            timeout=3,
        )
        client.get_collections()
        return True
    except Exception:
        logger.warning("Health check: Qdrant unreachable", exc_info=True)
        return False


def _check_database() -> bool:
    try:
        from api.support_store import _get_engine

        with _get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.warning("Health check: support database unreachable", exc_info=True)
        return False


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe plus a per-dependency snapshot for the support UI.

    Each check is independent and caught separately (CLAUDE.md: one failure
    must not crash a full run) — a dead Qdrant cluster still reports the
    database and LLM key as fine, matching how the agent itself degrades.
    """
    dependencies = {
        "llm_key": _check_llm_key(),
        "qdrant": _check_qdrant(),
        "database": _check_database(),
    }
    status = "ok" if all(dependencies.values()) else "degraded"
    return HealthResponse(status=status, dependencies=dependencies)


# ---------------------------------------------------------------------------
# /api/support — persistent demo CRM and conversations
# ---------------------------------------------------------------------------


@app.get("/api/support/customers")
def support_customers() -> list[dict]:
    return list_customers()


@app.get("/api/support/customers/lookup")
def support_customer_lookup(identifier: str) -> dict:
    customer = lookup_customer(identifier)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@app.get("/api/support/products")
def support_products() -> list[dict]:
    return list_products()


@app.get("/api/support/demo/overview")
def support_demo_overview() -> dict:
    return get_demo_overview()


@app.get("/api/support/conversations")
def support_conversations() -> list[dict]:
    return list_conversations()


@app.post("/api/support/conversations/messages")
def support_customer_message(req: CustomerMessageRequest) -> dict:
    try:
        return send_customer_message(req.message, req.conversation_id)
    except LookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/support/conversations/{conversation_id}/reply")
def support_reply(conversation_id: str, req: SupportReplyRequest) -> dict:
    try:
        return send_support_reply(conversation_id, req.message)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/support/conversations/{conversation_id}/read")
def support_mark_read(conversation_id: str) -> dict:
    try:
        return mark_conversation_read(conversation_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/support/conversations/{conversation_id}/finish")
def support_finish_conversation(conversation_id: str) -> dict:
    try:
        return resolve_conversation_by_support(conversation_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/support/conversations/{conversation_id}")
def support_delete_conversation(conversation_id: str) -> dict:
    try:
        delete_conversation(conversation_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": conversation_id}


@app.post("/api/support/demo/reset")
def support_demo_reset() -> list[dict]:
    return reset_demo_conversations()


# ---------------------------------------------------------------------------
# /api/chat — streamed NDJSON
# ---------------------------------------------------------------------------


def _ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _stream_chat(message: str, session_id: str):
    """Generator that yields NDJSON lines: token events then a final trace."""
    try:
        graph = _get_agent_graph()
    except Exception:
        logger.exception("Agent runtime could not be loaded")
        yield _ndjson(
            {
                "type": "error",
                "message": (
                    "The provider-backed agent is not configured. "
                    "The database support demo is still available."
                ),
            }
        )
        return
    config = {"configurable": {"thread_id": session_id}}

    prior_state = graph.get_state(config)
    prior_msg_count = len(prior_state.values.get("messages", []))

    start = time.time()
    try:
        for chunk, metadata in graph.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode="messages",
        ):
            if (
                hasattr(chunk, "content")
                and chunk.content
                and metadata.get("langgraph_node") == "agent"
                and not getattr(chunk, "tool_calls", None)
            ):
                yield _ndjson({"type": "token", "content": chunk.content})
    except Exception as e:
        logger.exception("Agent stream failed")
        yield _ndjson({"type": "error", "message": str(e)})
        return

    elapsed = time.time() - start

    # Final trace from checkpointer state
    state = graph.get_state(config)
    all_messages = state.values.get("messages", [])

    tools_called: list[str] = []
    for m in all_messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                if tc["name"] not in tools_called:
                    tools_called.append(tc["name"])

    new_messages = all_messages[prior_msg_count:]
    guard_fired = any(
        isinstance(m, HumanMessage)
        and isinstance(m.content, str)
        and GUARD_CORRECTION_MARKER in m.content
        for m in new_messages
    )

    trace = {
        "type": "trace",
        "tools": tools_called,
        "sql": (
            state.values.get("last_sql")
            if "query_database" in tools_called
            else None
        ),
        "chunks": (
            state.values.get("last_chunks")
            if "search_policies" in tools_called
            else None
        ),
        "latency": elapsed,
        "guard_fired": guard_fired,
    }
    yield _ndjson(trace)


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_chat(req.message, req.session_id),
        media_type="application/x-ndjson",
    )


# ---------------------------------------------------------------------------
# /api/transcribe — Whisper via Groq
# ---------------------------------------------------------------------------

_ALLOWED_AUDIO_MIME = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/flac",
}
_MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB — Groq Whisper hard cap


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscribeResponse:
    # Browsers send "audio/webm;codecs=opus" — strip parameters before checking.
    base_mime = (file.content_type or "").split(";", 1)[0].strip().lower()
    if base_mime and base_mime not in _ALLOWED_AUDIO_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio mime type: {file.content_type}",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_bytes) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (>25 MB)")

    try:
        text = voice.transcribe(audio_bytes, filename=file.filename or "audio.webm")
    except Exception as e:
        logger.exception("Whisper transcription failed")
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")

    return TranscribeResponse(text=text)


# ---------------------------------------------------------------------------
# /api/tts — ElevenLabs TTS
# ---------------------------------------------------------------------------


@app.post("/api/tts")
def tts(req: TtsRequest) -> Response:
    try:
        audio = voice.synthesize(req.text)
    except Exception as e:
        logger.exception("ElevenLabs synthesis failed")
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}")

    return Response(content=audio, media_type="audio/mpeg")
