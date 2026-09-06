"""
Orion agent graph.

ReAct loop with a guard post-processing step:

  [agent] → [tools] → [agent] → ... → [guard] → END

The guard runs once the agent produces its final response (no more tool calls).
It strips PII and checks for hallucinated numbers. On hallucination, it injects
a corrective message and re-runs the agent once.

State includes last_chunks and last_sql so trace data is scoped per session
(thread_id) rather than shared as module-level globals.

Usage:
    from orion_agent.agent.graph import graph

    response = graph.invoke(
        {"messages": [{"role": "user", "content": "Where is my order?"}]},
        config={"configurable": {"thread_id": "session-123"}},
    )
    print(response["messages"][-1].content)
"""

import json
import logging
from pathlib import Path
from typing import NotRequired

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

from orion_agent.agent import guard
from orion_agent.agent.config import CHECKPOINT_DB_PATH
from orion_agent.agent.llm import build_chat_model
from orion_agent.agent.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from orion_agent.agent.tools import escalate_to_human, query_database, search_policies

logger = logging.getLogger(__name__)

# Long-running threads (SqliteSaver persists messages across restarts) would
# otherwise grow the prompt without bound. Cut at a HumanMessage boundary so
# an AIMessage-with-tool-calls is never separated from its ToolMessage reply,
# which the chat API rejects.
MAX_HISTORY_MESSAGES = 40


def _truncate_history(messages: list) -> list:
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    window = messages[-MAX_HISTORY_MESSAGES:]
    for i, m in enumerate(window):
        if isinstance(m, HumanMessage):
            return window[i:]
    return window


_llm = build_chat_model(max_tokens=2048)
_llm_with_tools = _llm.bind_tools([search_policies, query_database, escalate_to_human])

_TOOLS_BY_NAME = {
    "search_policies": search_policies,
    "query_database": query_database,
    "escalate_to_human": escalate_to_human,
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class OrionState(MessagesState):
    """Extends MessagesState with per-session tool trace data."""

    last_chunks: NotRequired[list[dict]]
    last_sql: NotRequired[str]
    tools_called: NotRequired[list[str]]
    escalation: NotRequired[dict | None]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def agent_node(state: OrionState) -> dict:
    history = _truncate_history(state["messages"])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    logger.debug("Agent invoked with %d messages in context", len(messages))
    response = _llm_with_tools.invoke(messages)
    if response.tool_calls:
        tools = [tc["name"] for tc in response.tool_calls]
        logger.info("Agent decided to call tools: %s", tools)
    else:
        logger.info("Agent producing final response (no tool calls)")
    return {"messages": [response]}


def tools_node(state: OrionState) -> dict:
    """
    Custom tool node — calls tools and splits structured responses.

    Tools return JSON: {"answer": "...", "chunks": [...], "sql": "..."}.
    The LLM sees only "answer" via ToolMessage.content.
    Trace metadata (chunks, sql) is stored in graph state, scoped to this thread_id.
    """
    last = state["messages"][-1]
    tool_messages = []
    last_chunks: list[dict] = state.get("last_chunks", [])
    last_sql: str = state.get("last_sql", "")
    tools_called: list[str] = list(state.get("tools_called", []))
    escalation: dict | None = state.get("escalation")

    for tool_call in last.tool_calls:
        tool_fn = _TOOLS_BY_NAME[tool_call["name"]]
        raw = tool_fn.invoke(tool_call["args"])
        tools_called.append(tool_call["name"])

        try:
            data = json.loads(raw)
            answer = data.get("answer", raw)
            if "chunks" in data:
                last_chunks = data["chunks"]
            if "sql" in data:
                last_sql = data["sql"]
            if data.get("escalate"):
                escalation = {
                    "subject": data["subject"],
                    "action_needed": data["action_needed"],
                    "reason": data["reason"],
                }
        except (json.JSONDecodeError, TypeError):
            answer = str(raw)

        tool_messages.append(
            ToolMessage(
                content=answer,
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        )

    return {
        "messages": tool_messages,
        "last_chunks": last_chunks,
        "last_sql": last_sql,
        "tools_called": tools_called,
        "escalation": escalation,
    }


def guard_node(state: OrionState) -> dict:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not isinstance(last.content, str):
        return {}
    result = guard.apply(last.content)
    if result.text != last.content:
        logger.debug("Guard stripped PII from response")
        return {"messages": [AIMessage(content=result.text, id=last.id)]}
    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def should_continue(state: OrionState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "guard"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


_builder = StateGraph(OrionState)
_builder.add_node("agent", agent_node)
_builder.add_node("tools", tools_node)
_builder.add_node("guard", guard_node)

_builder.add_edge(START, "agent")
_builder.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", "guard": "guard"}
)
_builder.add_edge("tools", "agent")
_builder.add_edge("guard", END)

_checkpoint_path = Path(CHECKPOINT_DB_PATH)
_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
# Keep the context manager itself alive for the process lifetime. Retaining only
# its ``__enter__`` result lets the temporary manager be garbage-collected,
# which closes the SQLite connection underneath LangGraph.
_checkpointer_context = SqliteSaver.from_conn_string(str(_checkpoint_path))
_checkpointer = _checkpointer_context.__enter__()
graph = _builder.compile(checkpointer=_checkpointer)


def run_turn(thread_id: str, message: str) -> dict:
    """
    Run one customer turn through the agent graph — the entry point used by
    the `/api/support/*` demo path (api/support_store.py), not just /api/chat.

    Per-turn trace fields are reset before invoking so a prior turn's tool
    calls or escalation don't leak into a later, unrelated turn on the same
    thread_id; `messages` (conversation memory) is preserved by the
    checkpointer as normal.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "orion-support-turn",
        "tags": [PROMPT_VERSION],
        "metadata": {"thread_id": thread_id, "prompt_version": PROMPT_VERSION},
    }
    graph.update_state(
        config,
        {"tools_called": [], "last_chunks": [], "last_sql": "", "escalation": None},
    )
    graph.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    state = graph.get_state(config).values
    last = state["messages"][-1]
    content = last.content
    return {
        "response": content if isinstance(content, str) else str(content),
        "tools_called": state.get("tools_called", []),
        "last_chunks": state.get("last_chunks", []),
        "last_sql": state.get("last_sql", ""),
        "escalation": state.get("escalation"),
    }
