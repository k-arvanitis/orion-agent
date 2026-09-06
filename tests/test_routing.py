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

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from orion_agent.agent.graph import _truncate_history, graph, should_continue


def test_graph_checkpointer_connection_stays_open():
    state = graph.get_state(
        {"configurable": {"thread_id": "checkpointer-lifecycle-test"}}
    )
    assert state.values == {}


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


def _turn(n: int) -> list:
    """One conversation turn: a human message, an AI tool call, its tool
    reply, and the AI's final response — the unit truncation must not split."""
    return [
        HumanMessage(content=f"question {n}"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": f"c{n}"}]),
        ToolMessage(content="result", tool_call_id=f"c{n}", name="t"),
        AIMessage(content=f"answer {n}"),
    ]


def test_truncate_history_leaves_short_conversations_untouched():
    messages = _turn(1) + _turn(2)
    assert _truncate_history(messages) == messages


def test_truncate_history_cuts_on_a_human_message_boundary():
    from orion_agent.agent.graph import MAX_HISTORY_MESSAGES

    turns = [_turn(i) for i in range(20)]  # 80 messages, well over the cap
    messages = [m for turn in turns for m in turn]

    result = _truncate_history(messages)

    assert len(result) <= MAX_HISTORY_MESSAGES
    assert isinstance(result[0], HumanMessage)
    # No turn is split: every tool_calls AIMessage is immediately followed
    # by its matching ToolMessage within the truncated window.
    for i, m in enumerate(result):
        if isinstance(m, AIMessage) and m.tool_calls:
            assert isinstance(result[i + 1], ToolMessage)
            assert result[i + 1].tool_call_id == m.tool_calls[0]["id"]


def test_truncate_history_keeps_the_most_recent_turns():
    turns = [_turn(i) for i in range(20)]
    messages = [m for turn in turns for m in turn]

    result = _truncate_history(messages)

    assert result[-1].content == "answer 19"
