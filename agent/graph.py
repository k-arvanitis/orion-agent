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
    from agent.graph import graph

    response = graph.invoke(
        {"messages": [{"role": "user", "content": "Where is my order?"}]},
        config={"configurable": {"thread_id": "session-123"}},
    )
    print(response["messages"][-1].content)
"""

import json
import logging
from typing import NotRequired

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

from agent import guard
from agent.config import AGENT_MODEL
from agent.prompts import SYSTEM_PROMPT
from agent.tools import escalate, query_database, search_policies

logger = logging.getLogger(__name__)

_llm = ChatGroq(model=AGENT_MODEL, temperature=0, max_tokens=2048)
_llm_with_tools = _llm.bind_tools([search_policies, query_database, escalate])

_TOOLS_BY_NAME = {
    "search_policies": search_policies,
    "query_database": query_database,
    "escalate": escalate,
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class OrionState(MessagesState):
    """Extends MessagesState with per-session tool trace data."""
    last_chunks: NotRequired[list[dict]]
    last_sql: NotRequired[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def agent_node(state: OrionState) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
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
    last_chunks: list[dict] = []
    last_sql: str = ""

    for tool_call in last.tool_calls:
        tool_fn = _TOOLS_BY_NAME[tool_call["name"]]
        raw = tool_fn.invoke(tool_call["args"])

        try:
            data = json.loads(raw)
            answer = data.get("answer", raw)
            if "chunks" in data:
                last_chunks = data["chunks"]
            if "sql" in data:
                last_sql = data["sql"]
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
    }


def guard_node(state: OrionState) -> dict:
    messages = state["messages"]
    last = messages[-1]

    if not isinstance(last, AIMessage) or not isinstance(last.content, str):
        return {}

    tool_output = " ".join(
        m.content for m in messages if isinstance(m, ToolMessage) and m.content
    )

    result = guard.apply(last.content, tool_output)

    if result.clean:
        if result.text != last.content:
            logger.debug("Guard stripped PII from response")
            return {"messages": [AIMessage(content=result.text, id=last.id)]}
        return {}

    logger.warning("Guard detected hallucinated numbers: %s", result.hallucinated)
    correction = (
        f"Your previous response contained numbers not found in the tool output: "
        f"{result.hallucinated}. Rewrite using only data from the tool results."
    )
    return {"messages": [HumanMessage(content=correction)]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def should_continue(state: OrionState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "guard"


def after_guard(state: OrionState) -> str:
    last = state["messages"][-1]
    if isinstance(last, HumanMessage) and "Rewrite using only data" in last.content:
        return "agent"
    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


_builder = StateGraph(OrionState)
_builder.add_node("agent", agent_node)
_builder.add_node("tools", tools_node)
_builder.add_node("guard", guard_node)

_builder.add_edge(START, "agent")
_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "guard": "guard"})
_builder.add_edge("tools", "agent")
_builder.add_conditional_edges("guard", after_guard, {"agent": "agent", END: END})

graph = _builder.compile(checkpointer=MemorySaver())
