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


class CustomerMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=80)


class SupportReplyRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
