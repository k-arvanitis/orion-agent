"""
Unit tests for the agent graph routing functions.

Tests cover:
  - should_continue: routes to tools when AIMessage has tool_calls, else to guard
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-dummy-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import should_continue


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
