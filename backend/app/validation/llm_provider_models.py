"""Normalized LLM provider contracts (Topic 42b).

Provider adapters map SDK-specific streaming objects into these types.
Agent code must not depend on Groq/OpenAI SDK event classes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ProviderName = Literal["groq", "openai"]


class ModelMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ModelRequest(BaseModel):
    """Provider-independent chat completion request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1, max_length=128)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    max_tokens: int = Field(default=1024, ge=1, le=128_000)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stream: bool = False
    # Reserved for future OpenAI Responses API / extra options (ignored by Groq).
    extra: dict[str, Any] = Field(default_factory=dict)


class TextDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=0, max_length=32_000)


class ToolCallDelta(BaseModel):
    """Fragment of a streamed tool call — never executable alone."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    id: str | None = None
    name: str | None = None
    arguments_fragment: str = ""


class CompletedToolCall(BaseModel):
    """Fully accumulated tool call (arguments still untrusted until validated)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    arguments: str = Field(default="{}", max_length=100_000)
    type: str = "function"


class UsageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class StreamCompletion(BaseModel):
    """Normalized completed assistant message after a stream or non-stream call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str | None = None
    tool_calls: list[CompletedToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: UsageMetadata | None = None
    # True if any user-visible text was already forwarded during streaming.
    text_committed: bool = False


class ProviderError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1, max_length=64)
    message_safe: str = Field(min_length=1, max_length=500)
    retryable: bool = False
    # True when the failure happened after user-visible text was emitted.
    output_committed: bool = False


ProviderEventType = Literal[
    "text_delta",
    "tool_call_delta",
    "usage",
    "completion",
    "error",
]
