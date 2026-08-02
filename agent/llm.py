"""Shared chat-model construction for Orion's supported LLM providers."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from agent.config import (
    AGENT_MODEL,
    GROQ_API_KEY,
    LLM_API_BASE,
    LLM_PROVIDER,
    LLM_REQUEST_TIMEOUT_S,
    OPENROUTER_API_KEY,
    OPENROUTER_PROVIDER_PIN,
    chat_groq_kwargs,
)


def _openrouter_extra_body() -> dict[str, Any] | None:
    """Return an optional OpenRouter provider-routing preference."""
    if not OPENROUTER_PROVIDER_PIN:
        return None
    return {
        "provider": {
            "order": [OPENROUTER_PROVIDER_PIN],
            "allow_fallbacks": False,
        }
    }


def build_chat_model(*, max_tokens: int | None = None) -> BaseChatModel:
    """Build the configured chat model for agent and Text2SQL calls."""
    common: dict[str, Any] = {
        "model": AGENT_MODEL,
        "temperature": 0,
        "timeout": LLM_REQUEST_TIMEOUT_S,
        "max_retries": 3,
    }
    if max_tokens is not None:
        common["max_tokens"] = max_tokens

    if LLM_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY."
            )
        return ChatOpenAI(
            **common,
            base_url=LLM_API_BASE,
            api_key=OPENROUTER_API_KEY,
            extra_body=_openrouter_extra_body(),
            stream_usage=True,
        )

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError("LLM_PROVIDER=groq requires GROQ_API_KEY.")
        return ChatGroq(
            **common,
            api_key=GROQ_API_KEY,
            **chat_groq_kwargs(),
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}; use 'openrouter' or 'groq'."
    )
