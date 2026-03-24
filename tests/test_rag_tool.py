"""
Unit tests for the RAG tool.

External dependencies (Ollama, Qdrant) are fully mocked — no services required.
Tests cover:
  - Successful search returns structured JSON with answer and chunks
  - Empty results return the "no results" message
  - Qdrant/Ollama failures return a user-friendly fallback in the answer field
  - Chunks metadata is included in the structured response
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-dummy-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client.models import SparseVector

_FAKE_DENSE = [0.1] * 768
_FAKE_SPARSE = SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2])


def _make_hit(source: str, heading: str, content: str) -> MagicMock:
    hit = MagicMock()
    hit.payload = {"source": source, "heading": heading, "content": content}
    return hit


# ---------------------------------------------------------------------------
# Successful search
# ---------------------------------------------------------------------------


@patch("agent.tools.rag_tool._dense_embed", return_value=_FAKE_DENSE)
@patch("agent.tools.rag_tool._sparse_embed", return_value=_FAKE_SPARSE)
@patch("agent.tools.rag_tool._qdrant")
def test_search_returns_json_with_answer_and_chunks(mock_qdrant, mock_sparse, mock_dense):
    mock_qdrant.return_value.query_points.return_value = MagicMock(
        points=[_make_hit("return_policy.md", "30-Day Returns", "You have 30 days to return.")]
    )

    from agent.tools.rag_tool import search_policies
    raw = search_policies.invoke({"query": "return policy"})
    data = json.loads(raw)

    assert "answer" in data
    assert "chunks" in data
    assert "return_policy.md" in data["answer"]
    assert "30 days" in data["answer"]
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["source"] == "return_policy.md"


@patch("agent.tools.rag_tool._dense_embed", return_value=_FAKE_DENSE)
@patch("agent.tools.rag_tool._sparse_embed", return_value=_FAKE_SPARSE)
@patch("agent.tools.rag_tool._qdrant")
def test_search_chunk_metadata_is_complete(mock_qdrant, mock_sparse, mock_dense):
    mock_qdrant.return_value.query_points.return_value = MagicMock(
        points=[_make_hit("shipping_policy.md", "Express", "Express ships in 1 day.")]
    )

    from agent.tools.rag_tool import search_policies
    data = json.loads(search_policies.invoke({"query": "express shipping"}))

    chunk = data["chunks"][0]
    assert chunk["source"] == "shipping_policy.md"
    assert chunk["heading"] == "Express"
    assert "Express ships" in chunk["content"]


# ---------------------------------------------------------------------------
# No results
# ---------------------------------------------------------------------------


@patch("agent.tools.rag_tool._dense_embed", return_value=_FAKE_DENSE)
@patch("agent.tools.rag_tool._sparse_embed", return_value=_FAKE_SPARSE)
@patch("agent.tools.rag_tool._qdrant")
def test_search_no_results_returns_fallback(mock_qdrant, mock_sparse, mock_dense):
    mock_qdrant.return_value.query_points.return_value = MagicMock(points=[])

    from agent.tools.rag_tool import search_policies
    data = json.loads(search_policies.invoke({"query": "something obscure"}))

    assert "No relevant policy information found" in data["answer"]
    assert data["chunks"] == []


# ---------------------------------------------------------------------------
# Service failures
# ---------------------------------------------------------------------------


@patch("agent.tools.rag_tool._dense_embed", side_effect=ConnectionError("Ollama not running"))
def test_ollama_failure_returns_user_friendly_message(mock_dense):
    from agent.tools.rag_tool import search_policies
    data = json.loads(search_policies.invoke({"query": "warranty"}))

    assert "temporarily unavailable" in data["answer"].lower()
    assert data["chunks"] == []


@patch("agent.tools.rag_tool._dense_embed", return_value=_FAKE_DENSE)
@patch("agent.tools.rag_tool._sparse_embed", return_value=_FAKE_SPARSE)
@patch("agent.tools.rag_tool._qdrant")
def test_qdrant_failure_returns_user_friendly_message(mock_qdrant, mock_sparse, mock_dense):
    mock_qdrant.return_value.query_points.side_effect = ConnectionError("Qdrant unreachable")

    from agent.tools.rag_tool import search_policies
    data = json.loads(search_policies.invoke({"query": "shipping policy"}))

    assert "temporarily unavailable" in data["answer"].lower()
    assert data["chunks"] == []
