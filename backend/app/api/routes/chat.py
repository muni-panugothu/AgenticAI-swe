"""Authenticated chat API — identity from JWT only."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Annotated, Any, AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import AuthenticatedUser, get_current_user, reject_untrusted_user_id_in_body
from app.chatbot import chat_with_ai
from app.services.agent_output import (
    RAG_ABSTAIN_REPLY,
    build_abstain,
    build_from_agent_reply,
)
from app.services.chat_rate_limit_service import (
    ChatRateLimitExceeded,
    enforce_chat_rate_limit,
)
from app.services.llm_provider.factory import get_provider_name
from app.services.rag_document_service import is_document_question, retrieve
from app.services.request_cancellation import CancellationToken, RequestCancelled
from app.services.sse_protocol import (
    HEARTBEAT_INTERVAL_SECONDS,
    build_cancelled_frame,
    build_done_frame,
    build_early_meta_frame,
    build_heartbeat_frame,
    build_token_frame,
    format_error_event,
    iter_agent_turn_sse_events,
    iter_sse_with_guard,
)
from app.validation.agent_output_models import AgentTurnResult

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger("agentic.rag")

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_PROVIDER = "groq"


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=32000)
    history: list[ChatHistoryItem] = Field(default_factory=list)
    action: str | None = None
    useRag: bool = True
    sessionId: str | None = None
    preferProvider: str | None = None
    preferModel: str | None = None


class ChatResponseBody(BaseModel):
    reply: str
    sessionId: str | None = None
    action: str = "general"
    contextUsed: bool = False
    ragSkipped: bool = True
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    citations: list[Any] = Field(default_factory=list)


@dataclass(frozen=True)
class PreparedChatTurn:
    turn: AgentTurnResult
    session_id: str
    action: str
    context_used: bool
    rag_skipped: bool
    rest_citations: list[Any]
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER


@dataclass(frozen=True)
class ChatTurnSetup:
    session_id: str
    action: str
    rag_result: dict[str, Any] | None
    context_used: bool
    rag_skipped: bool
    rest_citations: list[Any]
    abstain_turn: AgentTurnResult | None = None
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER


def _resolve_provider_label() -> str:
    try:
        return get_provider_name()
    except Exception:
        return DEFAULT_PROVIDER


async def _enforce_chat_rate_limit(user_id: str) -> None:
    try:
        await run_in_threadpool(enforce_chat_rate_limit, user_id)
    except ChatRateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Chat rate limit exceeded",
                "code": "CHAT_RATE_LIMIT_EXCEEDED",
                "retry_after_seconds": exc.retry_after_seconds,
            },
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc
    except Exception as exc:
        logger.warning("Chat rate-limit check failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Chat rate limit temporarily unavailable",
                "code": "CHAT_RATE_LIMIT_UNAVAILABLE",
            },
        ) from exc


async def prepare_chat_setup(
    body: ChatRequest,
    user: AuthenticatedUser,
) -> ChatTurnSetup:
    reject_untrusted_user_id_in_body(body.model_dump())
    await _enforce_chat_rate_limit(user.id)

    request_id = body.sessionId or str(uuid4())
    action = body.action or "general"
    provider = _resolve_provider_label()
    rag_result = None
    if body.useRag and is_document_question(body.message):
        from app.services.memory_scope_service import get_active_project_scope

        project_scope = await run_in_threadpool(
            get_active_project_scope,
            user.id,
        )
        project_id = project_scope.get("project_id")
        conversation_history = [
            {"role": item.role, "content": item.content}
            for item in body.history
        ]
        try:
            rag_result = await run_in_threadpool(
                retrieve,
                body.message,
                user_id=user.id,
                tenant_id=user.tenant_id,
                project_id=project_id,
                mode="hybrid",
                conversation_history=conversation_history,
            )
        except Exception as exc:
            logger.warning("RAG retrieval failed (%s)", type(exc).__name__)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Document retrieval is temporarily unavailable",
                    "code": "RAG_RETRIEVAL_FAILED",
                },
            ) from exc
        if not rag_result.get("has_readable_documents", False):
            rag_result = None
        elif not rag_result.get("candidates"):
            return ChatTurnSetup(
                session_id=request_id,
                action=action,
                rag_result=None,
                context_used=False,
                rag_skipped=False,
                rest_citations=[],
                abstain_turn=build_abstain(RAG_ABSTAIN_REPLY),
                provider=provider,
            )

    server_citations = list(rag_result["citations"]) if rag_result else []
    return ChatTurnSetup(
        session_id=request_id,
        action=action,
        rag_result=rag_result,
        context_used=bool(rag_result and rag_result.get("candidates")),
        rag_skipped=rag_result is None,
        rest_citations=server_citations,
        provider=provider,
    )


async def prepare_chat_turn(
    body: ChatRequest,
    user: AuthenticatedUser,
) -> PreparedChatTurn:
    setup = await prepare_chat_setup(body, user)
    if setup.abstain_turn is not None:
        return PreparedChatTurn(
            turn=setup.abstain_turn,
            session_id=setup.session_id,
            action=setup.action,
            context_used=setup.context_used,
            rag_skipped=setup.rag_skipped,
            rest_citations=setup.rest_citations,
            model=setup.model,
            provider=setup.provider,
        )

    try:
        reply = await run_in_threadpool(
            chat_with_ai,
            body.message,
            user_id=user.id,
            request_id=setup.session_id,
            tenant_id=user.tenant_id,
            rag_context=(
                setup.rag_result["context"] if setup.rag_result else None
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Chat processing failed",
                "code": "CHAT_ERROR",
            },
        ) from exc

    turn = build_from_agent_reply(
        str(reply),
        citations=setup.rest_citations,
    )
    return PreparedChatTurn(
        turn=turn,
        session_id=setup.session_id,
        action=setup.action,
        context_used=setup.context_used,
        rag_skipped=setup.rag_skipped,
        rest_citations=setup.rest_citations,
        model=setup.model,
        provider=setup.provider,
    )


def prepared_to_response_body(prepared: PreparedChatTurn) -> ChatResponseBody:
    return ChatResponseBody(
        reply=prepared.turn.reply,
        sessionId=prepared.session_id,
        action=prepared.action,
        contextUsed=prepared.context_used,
        ragSkipped=prepared.rag_skipped,
        model=prepared.model,
        provider=prepared.provider,
        citations=prepared.rest_citations,
    )


@router.post("/chat", response_model=ChatResponseBody)
async def post_chat(
    body: ChatRequest,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> ChatResponseBody:
    prepared = await prepare_chat_turn(body, user)
    return prepared_to_response_body(prepared)


@router.post("/chat/stream")
async def post_chat_stream(
    request: Request,
    body: ChatRequest,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
):
    """SSE with live deltas, heartbeats, and cooperative cancel (42b/42c)."""
    setup = await prepare_chat_setup(body, user)

    if setup.abstain_turn is not None:
        prepared = PreparedChatTurn(
            turn=setup.abstain_turn,
            session_id=setup.session_id,
            action=setup.action,
            context_used=setup.context_used,
            rag_skipped=setup.rag_skipped,
            rest_citations=setup.rest_citations,
            model=setup.model,
            provider=setup.provider,
        )

        def abstain_gen():
            frames = iter_agent_turn_sse_events(
                turn=prepared.turn,
                session_id=prepared.session_id,
                action=prepared.action,
                context_used=prepared.context_used,
                rag_skipped=prepared.rag_skipped,
                model=prepared.model,
                provider=prepared.provider,
                rest_citations=prepared.rest_citations,
            )
            yield from iter_sse_with_guard(frames)

        return StreamingResponse(
            abstain_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    cancel_token = CancellationToken(
        request_id=setup.session_id,
        user_id=user.id,
    )
    event_q: queue.Queue = queue.Queue()

    def _worker() -> None:
        emitted: list[str] = []
        try:
            def on_delta(piece: str) -> None:
                if piece:
                    emitted.append(piece)
                    event_q.put(("token", piece))

            reply = chat_with_ai(
                body.message,
                user_id=user.id,
                request_id=setup.session_id,
                tenant_id=user.tenant_id,
                rag_context=(
                    setup.rag_result["context"] if setup.rag_result else None
                ),
                on_text_delta=on_delta,
                cancellation=cancel_token,
            )
            text = str(reply or "")
            already = "".join(emitted)
            if text and not already:
                event_q.put(("token", text))
            elif text.startswith(already) and len(text) > len(already):
                event_q.put(("token", text[len(already) :]))
            turn = build_from_agent_reply(
                text,
                citations=setup.rest_citations,
            )
            event_q.put(("turn", turn))
        except RequestCancelled:
            event_q.put(("cancelled", cancel_token.reason))
        except Exception:
            logger.warning("chat_stream_agent_failed", exc_info=False)
            event_q.put(("error", "CHAT_ERROR"))

    async def event_generator() -> AsyncIterator[str]:
        next_id = 1
        yield build_early_meta_frame(
            event_id=next_id,
            session_id=setup.session_id,
            action=setup.action,
            context_used=setup.context_used,
            rag_skipped=setup.rag_skipped,
            model=setup.model,
            provider=setup.provider,
            rest_citations=setup.rest_citations,
        )
        next_id += 1

        worker = threading.Thread(
            target=_worker, name=f"chat-stream-{setup.session_id[:8]}", daemon=True
        )
        worker.start()
        terminal = False
        last_hb = time.monotonic()
        try:
            while not terminal:
                if await request.is_disconnected():
                    cancel_token.cancel("client_disconnect")
                    yield build_cancelled_frame(
                        event_id=next_id, reason="client_disconnect"
                    )
                    terminal = True
                    break

                try:
                    kind, payload = event_q.get(timeout=0.25)
                except queue.Empty:
                    now = time.monotonic()
                    if now - last_hb >= HEARTBEAT_INTERVAL_SECONDS:
                        yield build_heartbeat_frame(event_id=next_id)
                        next_id += 1
                        last_hb = now
                    if not worker.is_alive() and event_q.empty():
                        if cancel_token.is_cancelled():
                            yield build_cancelled_frame(
                                event_id=next_id, reason=cancel_token.reason
                            )
                        else:
                            yield format_error_event(
                                event_id=next_id, code="STREAM_ERROR"
                            )
                        terminal = True
                    continue

                if kind == "token":
                    yield build_token_frame(event_id=next_id, token=str(payload))
                    next_id += 1
                elif kind == "turn":
                    turn = payload
                    assert isinstance(turn, AgentTurnResult)
                    yield build_done_frame(
                        event_id=next_id,
                        turn=turn,
                        session_id=setup.session_id,
                        model=setup.model,
                        provider=setup.provider,
                    )
                    terminal = True
                elif kind == "cancelled":
                    yield build_cancelled_frame(
                        event_id=next_id, reason=str(payload)
                    )
                    terminal = True
                elif kind == "error":
                    yield format_error_event(
                        event_id=next_id,
                        code=str(payload or "CHAT_ERROR"),
                    )
                    terminal = True
        finally:
            if not cancel_token.is_cancelled() and not worker.is_alive():
                pass
            elif not terminal:
                cancel_token.cancel("stream_generator_exit")
            worker.join(timeout=30)

    async def guarded() -> AsyncIterator[str]:
        emitted_terminal = False
        try:
            async for frame in event_generator():
                if "\nevent: done\n" in frame or "\nevent: error\n" in frame:
                    if emitted_terminal:
                        return
                    emitted_terminal = True
                yield frame
                if emitted_terminal and "\nevent: error\n" in frame:
                    return
        except Exception:
            logger.warning("sse_emit_failed", exc_info=False)
            if not emitted_terminal:
                yield format_error_event(event_id=1, code="STREAM_ERROR")

    return StreamingResponse(
        guarded(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/action/{action_id}")
async def post_chat_action(
    action_id: str,
    body: ChatRequest,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> ChatResponseBody:
    body.action = action_id
    return await post_chat(body, user)
