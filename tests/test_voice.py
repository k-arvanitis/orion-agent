"""
Unit tests for the voice I/O wrapper.

Groq Whisper and ElevenLabs are fully mocked — no API keys or network
required.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("ELEVENLABS_API_KEY", "test-dummy-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-dummy-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


def test_transcribe_returns_text_from_groq():
    from agent import voice

    fake_groq = MagicMock()
    fake_groq.audio.transcriptions.create.return_value = MagicMock(
        text="  what is the status of my order  "
    )

    with patch.object(voice, "_groq_client", fake_groq):
        result = voice.transcribe(b"\x00\x01\x02fake-wav-bytes")

    assert result == "what is the status of my order"
    fake_groq.audio.transcriptions.create.assert_called_once()
    call_kwargs = fake_groq.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["model"] == voice.VOICE_TRANSCRIBE_MODEL
    assert call_kwargs["file"][0] == "audio.wav"


def test_transcribe_handles_none_text():
    from agent import voice

    fake_groq = MagicMock()
    fake_groq.audio.transcriptions.create.return_value = MagicMock(text=None)

    with patch.object(voice, "_groq_client", fake_groq):
        assert voice.transcribe(b"x") == ""


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


def test_synthesize_returns_concatenated_audio_bytes():
    from agent import voice

    fake_eleven = MagicMock()
    fake_eleven.text_to_speech.convert.return_value = iter(
        [b"chunk1-", b"chunk2-", b"chunk3"]
    )

    with patch.object(voice, "_eleven_client", fake_eleven):
        result = voice.synthesize("Hello, this is your order status.")

    assert result == b"chunk1-chunk2-chunk3"
    call_kwargs = fake_eleven.text_to_speech.convert.call_args.kwargs
    assert call_kwargs["voice_id"] == voice.VOICE_TTS_VOICE_ID
    assert call_kwargs["model_id"] == voice.VOICE_TTS_MODEL
    assert "mp3" in call_kwargs["output_format"]


def test_synthesize_propagates_eleven_failure():
    from agent import voice

    fake_eleven = MagicMock()
    fake_eleven.text_to_speech.convert.side_effect = ConnectionError(
        "ElevenLabs unreachable"
    )

    with patch.object(voice, "_eleven_client", fake_eleven):
        try:
            voice.synthesize("anything")
        except ConnectionError:
            return
    raise AssertionError("Expected ConnectionError to propagate")
