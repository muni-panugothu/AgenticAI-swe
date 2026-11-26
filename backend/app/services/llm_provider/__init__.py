"""Topic 42b — provider-independent LLM access.

Current: Groq adapter. Future: OpenAI adapter (not implemented here).
Agent loop uses ``get_llm_provider()`` / ``completion_to_sdk_shape`` only.
"""

from __future__ import annotations

from app.services.llm_provider.base import (
    LLMProvider,
    StreamOutputCommittedError,
    completion_to_sdk_shape,
)
from app.services.llm_provider.factory import get_llm_provider, get_provider_name

__all__ = [
    "LLMProvider",
    "StreamOutputCommittedError",
    "completion_to_sdk_shape",
    "get_llm_provider",
    "get_provider_name",
]
