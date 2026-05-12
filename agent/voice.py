"""
Voice I/O wrapper for the Streamlit UI.

Two functions:
  - transcribe(audio_bytes) -> str  : Groq Whisper (whisper-large-v3-turbo)
  - synthesize(text)        -> bytes: ElevenLabs (eleven_turbo_v2_5)

The agent core (RAG, Text2SQL, guard, escalation) is unchanged. This module
is a thin I/O wrapper that produces text in (transcribe) and consumes text
out (synthesize) — voice adds no new reasoning layer, so existing eval
numbers carry over unchanged.

Both clients are lazy-initialised. The first call after process start pays a
small warm-up cost (one tiny request) so the user-facing first request is
fast.
"""

import logging
import os

from elevenlabs.client import ElevenLabs
from groq import Groq

from agent.config import (
    VOICE_TRANSCRIBE_MODEL,
    VOICE_TTS_MODEL,
    VOICE_TTS_VOICE_ID,
)

logger = logging.getLogger(__name__)

_groq_client: Groq | None = None
_eleven_client: ElevenLabs | None = None


def _groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


def _eleven() -> ElevenLabs:
    global _eleven_client
    if _eleven_client is None:
        _eleven_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        # Warm-up: a tiny synthesis call so the first real response
        # doesn't pay the cold-start latency.
        try:
            list(
                _eleven_client.text_to_speech.convert(
                    voice_id=VOICE_TTS_VOICE_ID,
                    text="hi",
                    model_id=VOICE_TTS_MODEL,
                    output_format="mp3_22050_32",
                )
            )
        except Exception:
            logger.debug("ElevenLabs warm-up call failed", exc_info=True)
    return _eleven_client


def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Transcribe audio bytes to text using Groq Whisper.

    Returns the transcribed text (stripped). Raises on API failure — callers
    should catch and degrade gracefully (e.g. fall back to text input).
    """
    logger.info("Whisper transcribe: %d bytes", len(audio_bytes))
    result = _groq().audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=VOICE_TRANSCRIBE_MODEL,
    )
    text = (result.text or "").strip()
    logger.debug("Transcribed: %r", text)
    return text


def synthesize(text: str) -> bytes:
    """Convert text to spoken audio (MP3 bytes) using ElevenLabs.

    Raises on API failure — callers should catch and continue showing the
    text-only response.
    """
    logger.info("ElevenLabs synthesize: %d chars", len(text))
    audio_iter = _eleven().text_to_speech.convert(
        voice_id=VOICE_TTS_VOICE_ID,
        text=text,
        model_id=VOICE_TTS_MODEL,
        output_format="mp3_44100_128",
    )
    return b"".join(audio_iter)
