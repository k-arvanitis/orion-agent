"""Unit tests for the shared chat-model provider factory."""

from unittest.mock import patch

import pytest

from orion_agent.agent import llm


def test_openrouter_builds_openai_compatible_client():
    """OpenRouter should use its OpenAI-compatible endpoint and API key."""
    with (
        patch.object(llm, "LLM_PROVIDER", "openrouter"),
        patch.object(llm, "OPENROUTER_API_KEY", "test-openrouter-key"),
        patch.object(llm, "OPENROUTER_PROVIDER_PIN", "DeepInfra"),
        patch.object(llm, "ChatOpenAI") as chat_openai,
    ):
        llm.build_chat_model(max_tokens=2048)

    chat_openai.assert_called_once_with(
        model=llm.AGENT_MODEL,
        temperature=0,
        timeout=llm.LLM_REQUEST_TIMEOUT_S,
        max_retries=3,
        max_tokens=2048,
        base_url=llm.LLM_API_BASE,
        api_key="test-openrouter-key",
        extra_body={
            "provider": {"order": ["DeepInfra"], "allow_fallbacks": False}
        },
        stream_usage=True,
    )


def test_groq_remains_available_as_fallback():
    """Explicit Groq configuration should still construct a Groq client."""
    with (
        patch.object(llm, "LLM_PROVIDER", "groq"),
        patch.object(llm, "GROQ_API_KEY", "test-groq-key"),
        patch.object(llm, "ChatGroq") as chat_groq,
    ):
        llm.build_chat_model()

    assert chat_groq.call_args.kwargs["api_key"] == "test-groq-key"


def test_openrouter_requires_a_key():
    """A missing OpenRouter key should fail with a useful configuration error."""
    with (
        patch.object(llm, "LLM_PROVIDER", "openrouter"),
        patch.object(llm, "OPENROUTER_API_KEY", ""),
        pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"),
    ):
        llm.build_chat_model()
