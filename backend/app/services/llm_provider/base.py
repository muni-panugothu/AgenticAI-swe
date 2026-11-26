"""Provider protocol and SDK-shape bridge for the existing agent loop."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Iterator, Protocol

from app.validation.llm_provider_models import (
    ModelRequest,
    ProviderError,
    StreamCompletion,
    TextDelta,
    ToolCallDelta,
    UsageMetadata,
)


class StreamOutputCommittedError(Exception):
    """Provider failed after user-visible text was already emitted.

    Callers must not retry the same request (Topic 38 / 42b).
    """

    def __init__(
        self,
        message: str = "Provider failed after streaming output",
        *,
        partial_text: str = "",
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_text = partial_text or ""
        self.cause = cause


class LLMProvider(Protocol):
    """Minimal provider surface — Groq now, OpenAI later."""

    name: str

    def complete(self, request: ModelRequest) -> StreamCompletion:
        """Non-streaming completion (REST ``/api/chat`` path)."""

    def stream(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        emit_text: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> StreamCompletion:
        """Streaming completion; optionally forward text deltas.

        When ``emit_text`` is False, text is accumulated but not forwarded
        (tool-enabled intermediate steps). Tool-call fragments are never
        executable until the returned ``StreamCompletion`` is complete.
        ``should_stop`` enables cooperative cancellation between chunks.
        """


def completion_to_sdk_shape(completion: StreamCompletion) -> SimpleNamespace:
    """Adapt StreamCompletion to the attribute shape the agent loop expects.

    Mimics ``response.choices[0].message`` / ``response.usage`` without
    importing Groq types into agent state.
    """
    tool_calls = []
    for tc in completion.tool_calls:
        tool_calls.append(
            SimpleNamespace(
                id=tc.id,
                type=tc.type or "function",
                function=SimpleNamespace(
                    name=tc.name,
                    arguments=tc.arguments or "{}",
                ),
            )
        )
    message = SimpleNamespace(
        role="assistant",
        content=completion.content,
        tool_calls=tool_calls or None,
        tool_call_id=None,
        name=None,
    )
    usage = None
    if completion.usage is not None:
        usage = SimpleNamespace(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            total_tokens=completion.usage.total_tokens,
        )
    choice = SimpleNamespace(
        message=message,
        finish_reason=completion.finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def iter_noop() -> Iterator[None]:
    return iter(())
