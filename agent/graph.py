"""
Orion agent graph.

ReAct loop with a guard post-processing step:

  [agent] → [tools] → [agent] → ... → [guard] → END

The guard runs once the agent produces its final response (no more tool calls).
It strips PII and checks for hallucinated numbers. On hallucination, it injects
a corrective message and re-runs the agent once.

Usage:
    from agent.graph import graph

    response = graph.invoke(
        {"messages": [{"role": "user", "content": "Where is my order?"}]},
        config={"configurable": {"thread_id": "session-123"}},
    )
    print(response["messages"][-1].content)
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from agent import guard
from agent.prompts import SYSTEM_PROMPT
from agent.tools import escalate, query_database, search_policies

_llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    max_tokens=1024,
)
_llm_with_tools = _llm.bind_tools([search_policies, query_database, escalate])
_tool_node = ToolNode([search_policies, query_database, escalate])


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def agent_node(state: MessagesState) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response]}


def guard_node(state: MessagesState) -> dict:
    messages = state["messages"]
    last = messages[-1]

    if not isinstance(last, AIMessage) or not isinstance(last.content, str):
        return {}

    # Collect all tool outputs from this turn
    tool_output = " ".join(
        m.content for m in messages if isinstance(m, ToolMessage)
    )

    result = guard.apply(last.content, tool_output)

    if result.clean:
        # Replace last message with PII-stripped version
        if result.text != last.content:
            return {"messages": [AIMessage(content=result.text, id=last.id)]}
        return {}

    # Hallucination detected — inject correction and let agent retry
    correction = (
        f"Your previous response contained numbers not found in the tool output: "
        f"{result.hallucinated}. Rewrite using only data from the tool results."
    )
    return {"messages": [HumanMessage(content=correction)]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "guard"


def after_guard(state: MessagesState) -> str:
    last = state["messages"][-1]
    if isinstance(last, HumanMessage) and "Rewrite using only data" in last.content:
        return "agent"
    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


_builder = StateGraph(MessagesState)
_builder.add_node("agent", agent_node)
_builder.add_node("tools", _tool_node)
_builder.add_node("guard", guard_node)

_builder.add_edge(START, "agent")
_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "guard": "guard"})
_builder.add_edge("tools", "agent")
_builder.add_conditional_edges("guard", after_guard, {"agent": "agent", END: END})

graph = _builder.compile(checkpointer=MemorySaver())
