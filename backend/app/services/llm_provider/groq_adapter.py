"""Groq Chat Completions adapter (Topic 42b).

Maps Groq SDK stream chunks → normalized StreamCompletion / text deltas.
Does not execute tools. Does not stream ``reasoning`` or tool arguments.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from app.services.llm_provider.base import StreamOutputCommittedError
from app.validation.llm_provider_models import (
    CompletedToolCall,
    ModelRequest,
    StreamCompletion,
    UsageMetadata,
)

logger = logging.getLogger("agentic.llm.groq")

# Parameters we intentionally do not send (unsupported / unsafe for this model).
_UNSUPPORTED_REQUEST_KEYS = frozenset(
    {
        "response_format",  # not required; JSON mode not used for agent loop
        "reasoning_format",
        "include_reasoning",
    }
)


def _request_to_groq_kwargs(request: ModelRequest, *, stream: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": list(request.messages),
        "max_tokens": int(request.max_tokens),
        "stream": bool(stream),
    }
    if request.tools:
        kwargs["tools"] = request.tools
        if request.tool_choice is not None and request.tool_choice != "none":
            kwargs["tool_choice"] = request.tool_choice
    # Note: OpenAI-style stream_options / response_format are not sent by
    # default — Groq 1.7 support varies by model; unsupported params are
    # reported via logs rather than silently attached.
    for key in request.extra:
        if key in _UNSUPPORTED_REQUEST_KEYS:
            logger.info("groq_skip_unsupported_param param=%s", key)
        elif key == "stream_options" and stream:
            kwargs["stream_options"] = request.extra["stream_options"]
    return kwargs


class _ToolCallAccumulator:
    """Accumulate fragmented tool-call deltas; never execute."""

    def __init__(self) -> None:
        self._by_index: dict[int, dict[str, str]] = {}

    def add(self, index: int, *, id: str | None, name: str | None, arguments: str | None) -> None:
        slot = self._by_index.setdefault(
            index, {"id": "", "name": "", "arguments": ""}
        )
        if id:
            slot["id"] = str(id)
        # Name and arguments arrive as fragments; always append (never execute).
        if name:
            slot["name"] += str(name)
        if arguments:
            slot["arguments"] += str(arguments)

    def completed(self) -> list[CompletedToolCall]:
        out: list[CompletedToolCall] = []
        for index in sorted(self._by_index):
            slot = self._by_index[index]
            name = (slot.get("name") or "").strip()
            if not name:
                continue
            call_id = (slot.get("id") or "").strip() or f"call_{index}"
            args = slot.get("arguments") or "{}"
            out.append(
                CompletedToolCall(
                    id=call_id,
                    name=name,
                    arguments=args,
                )
            )
        return out


def _usage_from_chunk(usage_obj: Any) -> UsageMetadata | None:
    if usage_obj is None:
        return None
    total = getattr(usage_obj, "total_tokens", None)
    return UsageMetadata(
        prompt_tokens=getattr(usage_obj, "prompt_tokens", None),
        completion_tokens=getattr(usage_obj, "completion_tokens", None),
        total_tokens=total if isinstance(total, int) else None,
    )


class GroqProvider:
    """Groq implementation of the LLMProvider protocol."""

    name = "groq"

    def __init__(
        self,
        client: Any = None,
        *,
        default_model: str,
        client_getter: Callable[[], Any] | None = None,
    ) -> None:
        # Prefer client_getter so tests can patch ``chatbot.client`` safely.
        self._client = client
        self._client_getter = client_getter
        self.default_model = default_model

    def _get_client(self) -> Any:
        if self._client_getter is not None:
            return self._client_getter()
        if self._client is not None:
            return self._client
        from app import chatbot

        return chatbot.client

    def complete(self, request: ModelRequest) -> StreamCompletion:
        kwargs = _request_to_groq_kwargs(request, stream=False)
        kwargs.pop("stream_options", None)
        kwargs["stream"] = False
        response = self._get_client().chat.completions.create(**kwargs)
        return self._completion_from_response(response, text_committed=False)

    def stream(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        emit_text: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> StreamCompletion:
        kwargs = _request_to_groq_kwargs(request, stream=True)
        text_parts: list[str] = []
        text_committed = False
        tool_acc = _ToolCallAccumulator()
        usage: UsageMetadata | None = None
        finish_reason: str | None = None
        event_stream = None

        def _forward(piece: str) -> None:
            nonlocal text_committed
            if not piece:
                return
            text_parts.append(piece)
            if emit_text and on_text_delta is not None:
                on_text_delta(piece)
                text_committed = True

        def _close_stream() -> None:
            if event_stream is None:
                return
            close = getattr(event_stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.warning("groq_stream_close_failed", exc_info=False)

        try:
            try:
                event_stream = self._get_client().chat.completions.create(**kwargs)
            except Exception as create_exc:
                # Safe fallback: streaming not accepted before any output.
                if text_committed:
                    raise StreamOutputCommittedError(
                        "Groq stream failed after output",
                        partial_text="".join(text_parts),
                        cause=create_exc,
                    ) from create_exc
                logger.warning(
                    "groq_stream_create_failed fallback_non_stream err=%s",
                    type(create_exc).__name__,
                )
                return self.complete(request.model_copy(update={"stream": False}))

            for chunk in event_stream:
                if should_stop is not None and should_stop():
                    _close_stream()
                    from app.services.request_cancellation import RequestCancelled

                    raise RequestCancelled("provider_stream_cancelled")
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = _usage_from_chunk(chunk_usage) or usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                fr = getattr(choice, "finish_reason", None)
                if fr:
                    finish_reason = str(fr)
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                # Never stream internal reasoning.
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    _forward(content)
                tool_calls = getattr(delta, "tool_calls", None) or []
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    tool_acc.add(
                        int(getattr(tc, "index", 0) or 0),
                        id=getattr(tc, "id", None),
                        name=getattr(fn, "name", None) if fn is not None else None,
                        arguments=(
                            getattr(fn, "arguments", None) if fn is not None else None
                        ),
                    )
        except StreamOutputCommittedError:
            _close_stream()
            raise
        except Exception as exc:
            _close_stream()
            from app.services.request_cancellation import RequestCancelled

            if isinstance(exc, RequestCancelled):
                raise
            partial = "".join(text_parts)
            if text_committed or partial:
                raise StreamOutputCommittedError(
                    "Groq stream interrupted after output",
                    partial_text=partial,
                    cause=exc,
                ) from exc
            raise

        _close_stream()
        completed_tools = tool_acc.completed()
        content = "".join(text_parts) if text_parts else None
        # If the model only issued tool calls, content may be empty — OK.
        return StreamCompletion(
            content=content,
            tool_calls=completed_tools,
            finish_reason=finish_reason,
            usage=usage,
            text_committed=text_committed,
        )

    def _completion_from_response(
        self, response: Any, *, text_committed: bool
    ) -> StreamCompletion:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return StreamCompletion(
                content=None,
                tool_calls=[],
                finish_reason=None,
                usage=_usage_from_chunk(getattr(response, "usage", None)),
                text_committed=text_committed,
            )
        message = choices[0].message
        raw_tools = getattr(message, "tool_calls", None) or []
        tool_calls: list[CompletedToolCall] = []
        for tc in raw_tools:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None) if fn is not None else None
            if not name:
                continue
            tool_calls.append(
                CompletedToolCall(
                    id=str(getattr(tc, "id", "") or "call_0"),
                    name=str(name),
                    arguments=str(getattr(fn, "arguments", None) or "{}"),
                    type=str(getattr(tc, "type", None) or "function"),
                )
            )
        content = getattr(message, "content", None)
        if content is not None and not isinstance(content, str):
            content = str(content) if content else None
        raw_finish = getattr(choices[0], "finish_reason", None)
        finish_reason = raw_finish if isinstance(raw_finish, str) else None
        return StreamCompletion(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=_usage_from_chunk(getattr(response, "usage", None)),
            text_committed=text_committed,
        )
