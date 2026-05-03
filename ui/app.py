"""
Orion — Streamlit chat interface for ShopNova customer support.

Left:    chat window (customer view)
Sidebar: tool trace panel (agent internals — which tools fired, SQL generated,
         policy chunks retrieved, latency)

Usage:
    uv run streamlit run ui/app.py
"""

import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

# Ensure project root is on the path when run from ui/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

import streamlit as st  # noqa: E402

from agent.graph import graph  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ShopNova Support",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "traces" not in st.session_state:
    st.session_state.traces = []

# ---------------------------------------------------------------------------
# Sidebar — tool trace panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔍 Tool Trace")
    st.caption("Agent internals for the last query.")

    if st.session_state.traces:
        trace = st.session_state.traces[-1]

        # Latency
        st.metric("Latency", f"{trace['latency']:.2f}s")

        # Tools fired
        st.markdown("**Tools used**")
        if not trace["tools"]:
            st.info("No tools called — answered from context.")
        else:
            cols = st.columns(len(trace["tools"]))
            labels = {
                "query_database": ("🗄️ SQL", "blue"),
                "search_policies": ("📄 RAG", "green"),
                "escalate": ("🚨 Escalation", "red"),
            }
            for col, tool_name in zip(cols, trace["tools"]):
                label, color = labels.get(tool_name, (tool_name, "gray"))
                col.markdown(
                    f"<span style='background:{color};color:white;padding:3px 8px;"
                    f"border-radius:4px;font-size:0.8em'>{label}</span>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # SQL query
        if trace.get("sql"):
            with st.expander("Generated SQL", expanded=True):
                st.code(trace["sql"], language="sql")

        # RAG chunks
        if trace.get("chunks"):
            with st.expander(
                f"Retrieved Chunks ({len(trace['chunks'])})", expanded=True
            ):
                for i, chunk in enumerate(trace["chunks"], 1):
                    st.markdown(f"**{i}. {chunk['heading']}**")
                    st.caption(f"📄 {chunk['source']}")
                    preview = chunk["content"][:300]
                    if len(chunk["content"]) > 300:
                        preview += "..."
                    st.markdown(
                        "<div style='font-size:0.85em;color:#555;"
                        "border-left:3px solid #ddd;padding-left:8px'>"
                        f"{preview}</div>",
                        unsafe_allow_html=True,
                    )
                    if i < len(trace["chunks"]):
                        st.divider()

    else:
        st.info("Ask a question to see the agent's tool trace here.")

    st.divider()

    # Session controls
    st.markdown("**Session**")
    st.caption(f"`{st.session_state.session_id[:8]}...`")
    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.traces = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------

st.markdown("## 🛍️ ShopNova Customer Support")
st.caption("Powered by Orion AI Agent · LangGraph · Qdrant · Supabase")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent with streaming
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        start = time.time()
        config = {"configurable": {"thread_id": st.session_state.session_id}}

        for chunk, metadata in graph.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
            stream_mode="messages",
        ):
            if (
                hasattr(chunk, "content")
                and chunk.content
                and metadata.get("langgraph_node") == "agent"
                and not getattr(chunk, "tool_calls", None)
            ):
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")

        elapsed = time.time() - start
        placeholder.markdown(full_response)
        response = full_response

    st.session_state.messages.append({"role": "assistant", "content": response})

    # Fetch state from checkpointer — trace data is scoped to this session's thread_id
    state = graph.get_state(config)
    all_messages = state.values.get("messages", [])
    tools_called = []
    for msg in all_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] not in tools_called:
                    tools_called.append(tc["name"])

    trace = {
        "tools": tools_called,
        "sql": state.values.get("last_sql")
        if "query_database" in tools_called
        else None,
        "chunks": state.values.get("last_chunks")
        if "search_policies" in tools_called
        else None,
        "latency": elapsed,
    }
    st.session_state.traces.append(trace)
    st.rerun()
