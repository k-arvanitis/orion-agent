"""
Orion — Streamlit chat interface for ShopNova customer support.

Inputs:    text (chat_input pinned to bottom) OR voice (audio_input above it).
           Voice flow: Groq Whisper transcribes audio → identical agent run →
           ElevenLabs reads the response aloud. Transcript is shown in chat
           so the user can verify what Whisper heard.
Sidebar:   tool trace (which tools fired, SQL generated, policy chunks
           retrieved, latency, hallucination guard fired).

Usage:
    uv run streamlit run ui/app.py
"""

import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# Ensure project root is on the path when run from ui/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

import streamlit as st  # noqa: E402

from agent import voice  # noqa: E402
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
# Sample questions — clients don't know the Olist dataset, so guide them.
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS = [
    (
        "🗄️ Order lookup (SQL)",
        "What is the status of order 416e49799e9260d93c8f636ce6661a55?",
    ),
    (
        "📄 Policy question (RAG)",
        "How long do I have to return a product?",
    ),
    (
        "🔀 Mixed (SQL + RAG)",
        "My order arrived late — am I eligible for a refund?",
    ),
    (
        "🚨 Escalation",
        "This is the third time my order is wrong. I want to speak to a human. "
        "My email is customer@example.com.",
    ),
]

# Guard correction message marker — must match agent/graph.py guard_node.
GUARD_CORRECTION_MARKER = "Rewrite using only data"

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "traces" not in st.session_state:
    st.session_state.traces = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "last_audio_id" not in st.session_state:
    # File-id of the last audio_input value we already transcribed — prevents
    # re-processing the same recording on every Streamlit rerun.
    st.session_state.last_audio_id = None

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

        # Hallucination guard flag
        if trace.get("guard_fired"):
            st.warning(
                "⚠️ Hallucination guard triggered — the agent's first answer "
                "contained numbers not present in the tool output. The agent was "
                "re-prompted and the corrected response is shown."
            )

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

        # RAG chunks — collapsible, source + heading visible
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
        st.session_state.pending_prompt = None
        st.session_state.last_audio_id = None
        st.rerun()

# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------

st.markdown("## 🛍️ ShopNova Customer Support")
st.caption("Powered by Orion AI Agent · LangGraph · Qdrant · Supabase · Voice mode")

# Sample-question panel — only shown when the chat is empty so it doesn't
# clutter the conversation once the user is engaged.
if not st.session_state.messages:
    st.markdown("#### 💡 Try a sample question")
    st.caption(
        "Don't know the dataset? Click any of these to see the agent route "
        "to the right tool."
    )
    sample_cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, (label, question) in zip(sample_cols, SAMPLE_QUESTIONS):
        if col.button(label, use_container_width=True, help=question):
            st.session_state.pending_prompt = question
            st.rerun()
    st.divider()

# Render conversation history. Assistant messages may carry a TTS audio blob
# from voice-mode runs; replay the latest one once with autoplay then mute.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        audio_bytes = msg.get("audio")
        if audio_bytes:
            should_autoplay = msg.get("autoplay_pending", False)
            if should_autoplay:
                msg["autoplay_pending"] = False
            st.audio(audio_bytes, format="audio/mp3", autoplay=should_autoplay)

# ---------------------------------------------------------------------------
# Input row — voice (above) + text (chat_input pinned to bottom)
# ---------------------------------------------------------------------------

st.markdown(
    "<div style='font-size:0.85em;color:#666;margin-top:1em'>"
    "🎤 <b>Voice mode:</b> tap the mic to record (the icon turns red while "
    "listening), tap again to stop. Whisper transcribes, the agent answers, "
    "and ElevenLabs reads the response aloud."
    "</div>",
    unsafe_allow_html=True,
)
audio_input = st.audio_input(
    "Voice input",
    label_visibility="collapsed",
    key="mic",
)

text_prompt = st.chat_input("Type your message — or use the mic above to speak...")

# ---------------------------------------------------------------------------
# Resolve which input fired this turn
# ---------------------------------------------------------------------------

prompt: str | None = None
input_was_voice = False

if text_prompt:
    prompt = text_prompt
elif audio_input is not None and audio_input.file_id != st.session_state.last_audio_id:
    # New recording — transcribe with Whisper
    st.session_state.last_audio_id = audio_input.file_id
    try:
        with st.spinner("🎤 Transcribing with Whisper..."):
            prompt = voice.transcribe(
                audio_input.getvalue(), filename=audio_input.name or "audio.wav"
            )
    except Exception as e:
        st.error(f"Voice transcription failed: {e}")
        prompt = None
    if prompt:
        input_was_voice = True
elif st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ---------------------------------------------------------------------------
# Process the turn
# ---------------------------------------------------------------------------

if prompt:
    # Show user message immediately. Voice transcripts are tagged so the user
    # can see exactly what Whisper heard.
    user_display = f"🎤 {prompt}" if input_was_voice else prompt
    st.session_state.messages.append({"role": "user", "content": user_display})
    with st.chat_message("user"):
        st.markdown(user_display)

    # Run agent with streaming
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        start = time.time()
        config = {"configurable": {"thread_id": st.session_state.session_id}}

        # Baseline: messages already in state before this turn. Used to scope
        # hallucination-guard detection to messages added by this turn only.
        prior_state = graph.get_state(config)
        prior_msg_count = len(prior_state.values.get("messages", []))

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

    # Generate TTS only when the input was voice — typed turns stay silent.
    response_audio: bytes | None = None
    if input_was_voice and response:
        try:
            with st.spinner("🔊 Generating audio response..."):
                response_audio = voice.synthesize(response)
        except Exception as e:
            st.warning(
                f"Voice playback failed (text response is shown above): {e}"
            )
            response_audio = None

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "audio": response_audio,
            "autoplay_pending": bool(response_audio),
        }
    )

    # Fetch state from checkpointer — trace data is scoped to this session's thread_id
    state = graph.get_state(config)
    all_messages = state.values.get("messages", [])
    tools_called = []
    for msg in all_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] not in tools_called:
                    tools_called.append(tc["name"])

    # Detect hallucination-guard correction: guard_node injects a HumanMessage
    # whose content contains the GUARD_CORRECTION_MARKER. Scan only messages
    # added in this turn so a previous turn's correction doesn't falsely flag
    # every subsequent turn.
    new_messages = all_messages[prior_msg_count:]
    guard_fired = any(
        isinstance(m, HumanMessage)
        and isinstance(m.content, str)
        and GUARD_CORRECTION_MARKER in m.content
        for m in new_messages
    )

    trace = {
        "tools": tools_called,
        "sql": state.values.get("last_sql")
        if "query_database" in tools_called
        else None,
        "chunks": state.values.get("last_chunks")
        if "search_policies" in tools_called
        else None,
        "latency": elapsed,
        "guard_fired": guard_fired,
    }
    st.session_state.traces.append(trace)
    st.rerun()
