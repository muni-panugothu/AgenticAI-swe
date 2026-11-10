"""Authenticated chat API — identity from JWT only."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import AuthenticatedUser, get_current_user, reject_untrusted_user_id_in_body
from app.chatbot import chat_with_ai

router = APIRouter(prefix="/api", tags=["chat"])


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
    model: str = "openai/gpt-oss-20b"
    provider: str = "groq"
    citations: list[Any] = Field(default_factory=list)


@router.post("/chat", response_model=ChatResponseBody)
async def post_chat(
    body: ChatRequest,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> ChatResponseBody:
    reject_untrusted_user_id_in_body(body.model_dump())
    request_id = body.sessionId or str(uuid4())
    try:
        reply = chat_with_ai(
            body.message,
            user_id=user.id,
            request_id=request_id,
            tenant_id=user.tenant_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Chat processing failed",
                "code": "CHAT_ERROR",
            },
        ) from exc
    return ChatResponseBody(
        reply=str(reply),
        sessionId=request_id,
        action=body.action or "general",
        contextUsed=False,
        ragSkipped=True,
    )


@router.post("/chat/stream")
async def post_chat_stream(
    body: ChatRequest,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
):
    """SSE envelope matching the Expo client (meta → token chunks → done)."""
    import json

    from fastapi.responses import PlainTextResponse

    payload = await post_chat(body, user)
    reply = payload.reply
    session = payload.sessionId or ""
    chunk_size = 48
    parts: list[str] = []
    meta = {
        "type": "meta",
        "sessionId": session,
        "action": payload.action,
        "contextUsed": payload.contextUsed,
        "ragSkipped": payload.ragSkipped,
        "model": payload.model,
        "provider": payload.provider,
    }
    parts.append(f"data: {json.dumps(meta)}\n\n")
    for i in range(0, len(reply), chunk_size):
        token = reply[i : i + chunk_size]
        parts.append(f"data: {json.dumps({'type': 'token', 'token': token})}\n\n")
    parts.append(
        f"data: {json.dumps({'type': 'done', 'sessionId': session, 'model': payload.model, 'provider': payload.provider})}\n\n"
    )
    return PlainTextResponse(content="".join(parts), media_type="text/event-stream")


@router.post("/chat/action/{action_id}")
async def post_chat_action(
    action_id: str,
    body: ChatRequest,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> ChatResponseBody:
    body.action = action_id
    return await post_chat(body, user)
