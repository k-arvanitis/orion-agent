"""Pydantic request/response models for the FastAPI surface."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(..., min_length=1, max_length=128)


class TranscribeResponse(BaseModel):
    text: str


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class HealthResponse(BaseModel):
    status: str = "ok"


class Chunk(BaseModel):
    source: str
    heading: str
    content: str


class TraceEvent(BaseModel):
    """Sent as the final NDJSON line on the /api/chat stream."""

    tools: list[str]
    sql: str | None = None
    chunks: list[Chunk] | None = None
    latency: float
    guard_fired: bool
