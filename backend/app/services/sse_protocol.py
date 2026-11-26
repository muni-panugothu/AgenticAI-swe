"""Topic 42a/42b — reliable SSE framing for agent chat streams.

Topic 42a: post-completion chunking from a finished ``AgentTurnResult``.
Topic 42b: live ``token`` frames from real provider text deltas, then one
``done`` (or ``error``) built from the trusted turn result.

Event contract (JSON in ``data:``; ``type`` discriminator preserved):
- meta — first event; session/routing fields (+ additive status when known)
- token — reply text (live deltas or post-completion chunks)
- done — exactly one terminal success event
- error — terminal failure; never followed by done

Heartbeat / disconnect (unchanged in 42b — Topic 42c):
- Client disconnect does not cancel synchronous ``chat_with_ai`` work in a
  worker thread; B6 cleanup runs when that thread's ``finally`` exits.
- No Last-Event-ID / reconnect replay in this stage.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Mapping  # Mapping used by live meta citations

from app.validation.agent_output_models import AgentTurnResult

logger = logging.getLogger("agentic.sse")

DEFAULT_TOKEN_CHUNK_SIZE = 48

# Safe public error codes only — never raw exception text.
SAFE_ERROR_MESSAGES: dict[str, str] = {
    "CHAT_ERROR": "Chat processing failed",
    "STREAM_ERROR": "Stream processing failed",
    "RAG_RETRIEVAL_FAILED": "Document retrieval is temporarily unavailable",
    "CHAT_RATE_LIMIT_EXCEEDED": "Chat rate limit exceeded",
    "CHAT_RATE_LIMIT_UNAVAILABLE": "Chat rate limit temporarily unavailable",
    "ABORTED": "The request was aborted before completion.",
    "CANCELLED": "The request was cancelled.",
}

HEARTBEAT_INTERVAL_SECONDS = 15.0


def _json_payload(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def format_sse_frame(
    *,
    event: str,
    data: Mapping[str, Any],
    event_id: int,
) -> str:
    """Format one SSE event with id, event name, JSON data, and blank line."""
    if event_id < 1:
        raise ValueError("event_id must be >= 1")
    name = str(event).strip() or "message"
    # Multi-line data is not used; payload is a single JSON object line.
    return (
        f"id: {event_id}\n"
        f"event: {name}\n"
        f"data: {_json_payload(data)}\n"
        f"\n"
    )


def format_sse_comment(text: str = "ping") -> str:
    """SSE comment frame (ignored by EventSource / typical parsers)."""
    safe = "".join(ch for ch in str(text) if ch not in "\r\n")[:80] or "ping"
    return f": {safe}\n\n"


def citations_for_stream(turn: AgentTurnResult) -> list[dict[str, Any]]:
    """Explicit stream serializer (camelCase). Does not alter REST citations."""
    return [
        {
            "index": c.index,
            "documentId": c.document_id,
            "source": c.source,
            "section": c.section,
            "chunkIndex": c.chunk_index,
            "excerpt": c.excerpt,
            "score": c.score,
        }
        for c in turn.citations
    ]


def safe_error_message(code: str, fallback: str | None = None) -> str:
    return SAFE_ERROR_MESSAGES.get(
        code,
        fallback or SAFE_ERROR_MESSAGES["STREAM_ERROR"],
    )


def format_error_event(
    *,
    event_id: int,
    code: str,
    message: str | None = None,
) -> str:
    """Terminal error event — callers must not emit done afterward."""
    payload = {
        "type": "error",
        "error": message or safe_error_message(code),
        "code": code,
    }
    return format_sse_frame(event=payload["type"], data=payload, event_id=event_id)


def iter_agent_turn_sse_events(
    *,
    turn: AgentTurnResult,
    session_id: str,
    action: str = "general",
    context_used: bool = False,
    rag_skipped: bool = True,
    model: str = "openai/gpt-oss-20b",
    provider: str = "groq",
    rest_citations: list[Any] | None = None,
    chunk_size: int = DEFAULT_TOKEN_CHUNK_SIZE,
) -> Iterator[str]:
    """Yield framed SSE events for one completed AgentTurnResult.

    ``rest_citations`` is the server RAG list used on REST ``/api/chat`` and
    is echoed on ``meta.citations`` for wire compatibility. Stream-normalized
    citations are additive under ``meta.streamCitations`` (camelCase).
    """
    if chunk_size < 1:
        chunk_size = DEFAULT_TOKEN_CHUNK_SIZE

    next_id = 1
    terminal_emitted = False

    def _next() -> int:
        nonlocal next_id
        eid = next_id
        next_id += 1
        return eid

    # Trusted error turns → single terminal error (no done).
    if turn.response_type == "error" or turn.completion_status == "failed":
        code = turn.error_code or "CHAT_ERROR"
        msg = turn.error_message_safe or safe_error_message(code)
        yield format_error_event(event_id=_next(), code=code, message=msg)
        return

    meta: dict[str, Any] = {
        "type": "meta",
        "sessionId": session_id,
        "action": action or "general",
        "contextUsed": bool(context_used),
        "ragSkipped": bool(rag_skipped),
        "model": model,
        "provider": provider,
        # Preserve prior stream/REST citation objects for Expo/web.
        "citations": list(rest_citations) if rest_citations is not None else [],
        # Additive Topic 41a/42a fields (clients may ignore).
        "responseType": turn.response_type,
        "completionStatus": turn.completion_status,
        "uncertain": bool(turn.uncertain),
        "clarificationRequired": bool(turn.clarification_required),
        "partial": bool(turn.partial),
        "streamCitations": citations_for_stream(turn),
    }
    yield format_sse_frame(event="meta", data=meta, event_id=_next())

    reply = turn.reply if turn.reply is not None else ""
    if reply:
        for i in range(0, len(reply), chunk_size):
            token = reply[i : i + chunk_size]
            yield format_sse_frame(
                event="token",
                data={"type": "token", "token": token},
                event_id=_next(),
            )

    if terminal_emitted:
        return

    done: dict[str, Any] = {
        "type": "done",
        "sessionId": session_id,
        "model": model,
        "provider": provider,
        "completionStatus": turn.completion_status,
        "responseType": turn.response_type,
        "partial": bool(turn.partial),
    }
    yield format_sse_frame(event="done", data=done, event_id=_next())
    terminal_emitted = True


def iter_sse_with_guard(
    frames: Iterator[str],
    *,
    start_event_id: int = 1,
) -> Iterator[str]:
    """Wrap a frame iterator; on unexpected failure emit one safe error.

    Does not log full SSE payloads. Does not emit done after error.
    """
    emitted_terminal = False
    next_id = max(1, start_event_id)
    try:
        for frame in frames:
            # Detect terminal event types without logging payload contents.
            if "\nevent: done\n" in frame or "\nevent: error\n" in frame:
                if emitted_terminal:
                    logger.warning("sse_duplicate_terminal_suppressed")
                    return
                emitted_terminal = True
            yield frame
            if emitted_terminal and "\nevent: error\n" in frame:
                return
    except Exception:
        logger.warning("sse_emit_failed", exc_info=False)
        if not emitted_terminal:
            yield format_error_event(
                event_id=next_id,
                code="STREAM_ERROR",
            )


def build_early_meta_frame(
    *,
    event_id: int,
    session_id: str,
    action: str = "general",
    context_used: bool = False,
    rag_skipped: bool = True,
    model: str = "openai/gpt-oss-20b",
    provider: str = "groq",
    rest_citations: list[Any] | None = None,
) -> str:
    """First SSE event for live streaming (status filled on done)."""
    meta: dict[str, Any] = {
        "type": "meta",
        "sessionId": session_id,
        "action": action or "general",
        "contextUsed": bool(context_used),
        "ragSkipped": bool(rag_skipped),
        "model": model,
        "provider": provider,
        "citations": list(rest_citations) if rest_citations is not None else [],
        "streamCitations": citations_for_stream(
            # Empty turn citations — server RAG list is on citations above.
            AgentTurnResult(reply="", citations=[])
        )
        if not rest_citations
        else [
            {
                "index": i,
                "documentId": str(
                    c.get("documentId") or c.get("document_id") or ""
                ),
                "source": c.get("source") or c.get("filename"),
                "section": c.get("section"),
                "chunkIndex": int(c.get("chunkIndex") or c.get("chunk_index") or 0),
                "excerpt": str(c.get("excerpt") or "")[:4000],
                "score": float(c.get("score") or 0.0),
            }
            for i, c in enumerate(rest_citations)
            if isinstance(c, Mapping)
            and (c.get("documentId") or c.get("document_id"))
        ],
    }
    return format_sse_frame(event="meta", data=meta, event_id=event_id)


def build_token_frame(*, event_id: int, token: str) -> str:
    return format_sse_frame(
        event="token",
        data={"type": "token", "token": token},
        event_id=event_id,
    )


def build_done_frame(
    *,
    event_id: int,
    turn: AgentTurnResult,
    session_id: str,
    model: str = "openai/gpt-oss-20b",
    provider: str = "groq",
) -> str:
    done: dict[str, Any] = {
        "type": "done",
        "sessionId": session_id,
        "model": model,
        "provider": provider,
        "completionStatus": turn.completion_status,
        "responseType": turn.response_type,
        "partial": bool(turn.partial),
        "uncertain": bool(turn.uncertain),
        "clarificationRequired": bool(turn.clarification_required),
    }
    return format_sse_frame(event="done", data=done, event_id=event_id)


def build_heartbeat_frame(*, event_id: int, ts: float | None = None) -> str:
    """Additive lifecycle ping — not user-visible text; clients may ignore."""
    import time as _time

    payload = {
        "type": "heartbeat",
        "ts": float(ts if ts is not None else _time.time()),
    }
    return format_sse_frame(event="heartbeat", data=payload, event_id=event_id)


def build_cancelled_frame(*, event_id: int, reason: str = "client_disconnect") -> str:
    """Terminal cancellation — never followed by done."""
    del reason  # reason is for logs only; public payload stays generic
    return format_error_event(
        event_id=event_id,
        code="CANCELLED",
        message=safe_error_message("CANCELLED"),
    )
