"""
Tests for the FastAPI surface.

The agent graph and voice module are mocked — no Qdrant, Groq, ElevenLabs,
fastembed, or DB calls happen during the test run.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("ELEVENLABS_API_KEY", "test-dummy-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-dummy-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def client():
    from api.main import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


def test_health_returns_ok_when_all_dependencies_are_up(client, monkeypatch):
    monkeypatch.setattr("api.main._check_llm_key", lambda: True)
    monkeypatch.setattr("api.main._check_qdrant", lambda: True)
    monkeypatch.setattr("api.main._check_database", lambda: True)

    r = client.get("/api/health")

    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "dependencies": {"llm_key": True, "qdrant": True, "database": True},
    }


def test_health_returns_degraded_when_a_dependency_is_down(client, monkeypatch):
    monkeypatch.setattr("api.main._check_llm_key", lambda: True)
    monkeypatch.setattr("api.main._check_qdrant", lambda: False)
    monkeypatch.setattr("api.main._check_database", lambda: True)

    r = client.get("/api/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["qdrant"] is False


# ---------------------------------------------------------------------------
# /api/chat — streaming NDJSON
# ---------------------------------------------------------------------------


def _fake_stream():
    """Mimic LangGraph stream_mode='messages' — yields (chunk, metadata) tuples."""
    chunk1 = MagicMock(content="Your order ", tool_calls=None)
    chunk2 = MagicMock(content="is on the way.", tool_calls=None)
    meta = {"langgraph_node": "agent"}
    return iter([(chunk1, meta), (chunk2, meta)])


def _fake_state(messages=None, last_sql=None, last_chunks=None):
    """Build a fake graph.get_state() return value."""
    return MagicMock(
        values={
            "messages": messages or [],
            "last_sql": last_sql,
            "last_chunks": last_chunks,
        }
    )


def test_chat_streams_tokens_then_trace(client):
    from langchain_core.messages import AIMessage

    fake_ai = AIMessage(
        content="",
        tool_calls=[{"name": "query_database", "args": {"q": "x"}, "id": "tc1"}],
    )

    fake_graph = MagicMock()
    fake_graph.stream.return_value = _fake_stream()
    fake_graph.get_state.side_effect = [
        _fake_state(messages=[]),  # before turn (prior_msg_count = 0)
        _fake_state(  # after turn
            messages=[fake_ai],
            last_sql="SELECT * FROM orders LIMIT 1",
        ),
    ]

    with patch("api.main._get_agent_graph", return_value=fake_graph):
        with client.stream(
            "POST",
            "/api/chat",
            json={"message": "where is order 123?", "session_id": "s1"},
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("application/x-ndjson")

            lines = [json.loads(line) for line in r.iter_lines() if line]

    # Two token events, one trace event
    token_events = [le for le in lines if le["type"] == "token"]
    trace_events = [le for le in lines if le["type"] == "trace"]
    assert "".join(t["content"] for t in token_events) == "Your order is on the way."
    assert len(trace_events) == 1

    trace = trace_events[0]
    assert trace["tools"] == ["query_database"]
    assert trace["sql"] == "SELECT * FROM orders LIMIT 1"
    assert trace["chunks"] is None  # SQL turn — no chunks
    assert trace["guard_fired"] is False
    assert trace["latency"] > 0


def test_chat_validates_payload(client):
    r = client.post("/api/chat", json={"message": "", "session_id": "s1"})
    assert r.status_code == 422


def test_chat_degrades_cleanly_when_agent_runtime_is_not_configured(client):
    with patch(
        "api.main._get_agent_graph",
        side_effect=RuntimeError("provider configuration missing"),
    ):
        with client.stream(
            "POST",
            "/api/chat",
            json={"message": "where is my order?", "session_id": "s1"},
        ) as response:
            lines = [json.loads(line) for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert lines == [
        {
            "type": "error",
            "message": (
                "The provider-backed agent is not configured. "
                "The database support demo is still available."
            ),
        }
    ]


# ---------------------------------------------------------------------------
# /api/transcribe
# ---------------------------------------------------------------------------


def test_transcribe_returns_text(client):
    with patch("api.main.voice.transcribe", return_value="hello world") as m:
        r = client.post(
            "/api/transcribe",
            files={"file": ("audio.webm", b"\x00\x01fake-audio", "audio/webm")},
        )
    assert r.status_code == 200
    assert r.json() == {"text": "hello world"}
    m.assert_called_once()


def test_transcribe_rejects_empty_file(client):
    r = client.post(
        "/api/transcribe",
        files={"file": ("audio.webm", b"", "audio/webm")},
    )
    assert r.status_code == 400


def test_transcribe_rejects_unsupported_mime(client):
    r = client.post(
        "/api/transcribe",
        files={"file": ("not-audio.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415


def test_transcribe_502_on_whisper_failure(client):
    with patch(
        "api.main.voice.transcribe",
        side_effect=ConnectionError("whisper down"),
    ):
        r = client.post(
            "/api/transcribe",
            files={"file": ("a.webm", b"\x00", "audio/webm")},
        )
    assert r.status_code == 502


# ---------------------------------------------------------------------------
# /api/tts
# ---------------------------------------------------------------------------


def test_tts_returns_mpeg_bytes(client):
    fake_audio = b"ID3\x04fake-mp3-bytes"
    with patch("api.main.voice.synthesize", return_value=fake_audio) as m:
        r = client.post("/api/tts", json={"text": "Hello there."})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/mpeg"
    assert r.content == fake_audio
    m.assert_called_once_with("Hello there.")


def test_tts_502_on_eleven_failure(client):
    with patch("api.main.voice.synthesize", side_effect=RuntimeError("eleven down")):
        r = client.post("/api/tts", json={"text": "anything"})
    assert r.status_code == 502


# ---------------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------------


def test_write_endpoint_open_when_no_api_key_configured(client, monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "")
    monkeypatch.setattr("api.main.reset_demo_conversations", lambda: [], raising=False)

    r = client.post("/api/support/demo/reset")

    assert r.status_code == 200


def test_write_endpoint_rejects_missing_key_when_configured(client, monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "secret123")

    r = client.post("/api/support/demo/reset")

    assert r.status_code == 401


def test_write_endpoint_rejects_wrong_key_when_configured(client, monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "secret123")

    r = client.post("/api/support/demo/reset", headers={"X-API-Key": "wrong"})

    assert r.status_code == 401


def test_write_endpoint_accepts_correct_key_when_configured(client, monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "secret123")
    monkeypatch.setattr("api.main.reset_demo_conversations", lambda: [], raising=False)

    r = client.post("/api/support/demo/reset", headers={"X-API-Key": "secret123"})

    assert r.status_code == 200


def test_read_endpoints_stay_open_even_when_api_key_configured(client, monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "secret123")
    monkeypatch.setattr("api.main.list_customers", lambda: [], raising=False)

    r = client.get("/api/support/customers")

    assert r.status_code == 200


# ---------------------------------------------------------------------------
# /api/chat rate limiting
# ---------------------------------------------------------------------------


def test_chat_rate_limit_blocks_after_the_configured_count(client, monkeypatch):
    from langchain_core.messages import AIMessage

    monkeypatch.setattr("api.main.RATE_LIMIT_PER_MINUTE", 2)
    monkeypatch.setattr(
        "api.main._rate_limit_hits",
        __import__("collections").defaultdict(__import__("collections").deque),
    )

    fake_graph = MagicMock()
    fake_graph.stream.return_value = _fake_stream()
    fake_graph.get_state.side_effect = [
        _fake_state(messages=[]),
        _fake_state(messages=[AIMessage(content="", tool_calls=[])]),
    ] * 3

    with patch("api.main._get_agent_graph", return_value=fake_graph):
        payload = {"message": "hi", "session_id": "rl-1"}
        r1 = client.post("/api/chat", json=payload)
        r2 = client.post("/api/chat", json=payload)
        r3 = client.post("/api/chat", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429


def test_chat_rate_limit_disabled_when_zero(client, monkeypatch):
    from langchain_core.messages import AIMessage

    monkeypatch.setattr("api.main.RATE_LIMIT_PER_MINUTE", 0)

    fake_graph = MagicMock()
    fake_graph.stream.return_value = _fake_stream()
    fake_graph.get_state.side_effect = [
        _fake_state(messages=[]),
        _fake_state(messages=[AIMessage(content="", tool_calls=[])]),
    ] * 5

    with patch("api.main._get_agent_graph", return_value=fake_graph):
        payload = {"message": "hi", "session_id": "rl-2"}
        for _ in range(4):
            r = client.post("/api/chat", json=payload)
            assert r.status_code == 200
