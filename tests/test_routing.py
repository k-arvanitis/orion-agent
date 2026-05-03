"""
Unit tests for the agent graph routing functions.

Tests cover:
  - should_continue: routes to tools when AIMessage has tool_calls, else to guard
  - after_guard: routes back to agent on hallucination correction, else to END
"""

import os
import sys
from pathlib import Path

# Set dummy env vars before importing graph (ChatGroq reads GROQ_API_KEY at import)
os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-dummy-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from agent.graph import after_guard, should_continue

# ---------------------------------------------------------------------------
# should_continue
# ---------------------------------------------------------------------------


def test_should_continue_returns_tools_when_tool_calls_present():
    tool_call = {
        "name": "search_policies",
        "args": {"query": "return policy"},
        "id": "call1",
    }
    msg = AIMessage(content="", tool_calls=[tool_call])
    state = {"messages": [msg]}
    assert should_continue(state) == "tools"


def test_should_continue_returns_guard_when_no_tool_calls():
    msg = AIMessage(content="Your order was delivered.")
    state = {"messages": [msg]}
    assert should_continue(state) == "guard"


def test_should_continue_returns_guard_for_human_message():
    msg = HumanMessage(content="Where is my order?")
    state = {"messages": [msg]}
    assert should_continue(state) == "guard"


# ---------------------------------------------------------------------------
# after_guard
# ---------------------------------------------------------------------------


def test_after_guard_returns_agent_on_hallucination_correction():
    msg = HumanMessage(content="Rewrite using only data from the tool results.")
    state = {"messages": [msg]}
    assert after_guard(state) == "agent"


def test_after_guard_returns_end_for_clean_ai_message():
    msg = AIMessage(content="Your order was delivered on time.")
    state = {"messages": [msg]}
    assert after_guard(state) == END


def test_after_guard_returns_end_for_normal_human_message():
    msg = HumanMessage(content="What is the return policy?")
    state = {"messages": [msg]}
    assert after_guard(state) == END
