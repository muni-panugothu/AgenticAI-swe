"""Configuration-driven LLM provider selection (Topic 42b).

Environment contract:
- LLM_PROVIDER: ``groq`` (default) | ``openai`` (reserved; not implemented)
- GROQ_API_KEY / existing Groq client (required for groq)
- GROQ_MODEL or chatbot.MODEL (default model id)
- Future OpenAI (not implemented in 42b):
  - OPENAI_API_KEY
  - OPENAI_MODEL
  - optional OPENAI_BASE_URL

Selecting ``openai`` raises until a dedicated adapter is added — never
silently fall through to Groq with the wrong credentials.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from app.services.llm_provider.groq_adapter import GroqProvider


def get_provider_name() -> str:
    raw = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    return raw or "groq"


def get_default_model() -> str:
    return (
        os.getenv("GROQ_MODEL")
        or os.getenv("LLM_MODEL")
        or "openai/gpt-oss-20b"
    )


@lru_cache(maxsize=1)
def get_llm_provider() -> Any:
    """Return the process-wide provider adapter.

    Uses the existing Groq client singleton from ``app.chatbot`` for groq.
    """
    name = get_provider_name()
    if name == "openai":
        raise RuntimeError(
            "LLM_PROVIDER=openai is reserved for a future OpenAI adapter "
            "(Topic 42b does not implement it). Set LLM_PROVIDER=groq."
        )
    if name != "groq":
        raise RuntimeError(f"Unsupported LLM_PROVIDER={name!r}")

    from app import chatbot

    model = os.getenv("GROQ_MODEL") or getattr(chatbot, "MODEL", get_default_model())

    def _client_getter():
        return chatbot.client

    return GroqProvider(
        client_getter=_client_getter,
        default_model=model,
    )


def reset_llm_provider_cache() -> None:
    """Test helper — clear cached provider."""
    get_llm_provider.cache_clear()
