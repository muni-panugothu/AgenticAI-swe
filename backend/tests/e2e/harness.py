"""Reusable offline API E2E harness for Batch B.

Preferred flow:
  TestClient → authenticated /api/chat[/stream]
  → real prepare_chat_setup / prepare_chat_turn
  → real chat_with_ai agent loop
  → real prepare_tool_invocation / execute_tool
  → ScriptedFakeProvider + isolated external fakes
  → API response / SSE events

Never replaces the agent loop with a predetermined reply.
Never calls live Groq / Tavily / production Mongo.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.main import app
from app.services.llm_provider.factory import reset_llm_provider_cache
from app.validation.llm_provider_models import (
    CompletedToolCall,
    ModelRequest,
    StreamCompletion,
)

os.environ.setdefault("JWT_SECRET", "e2e-batch-b-jwt-secret-key-32chars!!")

ScriptStep = StreamCompletion | Callable[[ModelRequest], StreamCompletion]


def text_completion(
    content: str,
    *,
    finish_reason: str = "stop",
    chunks: list[str] | None = None,
) -> StreamCompletion:
    return StreamCompletion(
        content=content,
        tool_calls=[],
        finish_reason=finish_reason,
        text_committed=False,
    )


def tool_completion(
    name: str,
    arguments: dict[str, Any] | str,
    *,
    call_id: str | None = None,
) -> StreamCompletion:
    args = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False)
    )
    return StreamCompletion(
        content=None,
        tool_calls=[
            CompletedToolCall(
                id=call_id or f"tc_{name}_{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=args,
            )
        ],
        finish_reason="tool_calls",
        text_committed=False,
    )


def multi_tool_completion(
    calls: list[tuple[str, dict[str, Any] | str]],
) -> StreamCompletion:
    tool_calls = []
    for i, (name, arguments) in enumerate(calls):
        args = (
            arguments
            if isinstance(arguments, str)
            else json.dumps(arguments, ensure_ascii=False)
        )
        tool_calls.append(
            CompletedToolCall(
                id=f"tc_multi_{i}_{name}",
                name=name,
                arguments=args,
            )
        )
    return StreamCompletion(
        content=None,
        tool_calls=tool_calls,
        finish_reason="tool_calls",
        text_committed=False,
    )


class ScriptedFakeProvider:
    """Deterministic LLM stand-in. Does not bypass tool selection/execution."""

    name = "fake"

    def __init__(self, script: list[ScriptStep] | None = None):
        self.script: list[ScriptStep] = list(script or [])
        self.index = 0
        self.complete_calls = 0
        self.stream_calls = 0
        self.requests: list[ModelRequest] = []
        self._lock = threading.Lock()

    def append(self, step: ScriptStep) -> None:
        self.script.append(step)

    def _next(self, request: ModelRequest) -> StreamCompletion:
        with self._lock:
            self.requests.append(request)
            if self.index >= len(self.script):
                return text_completion(
                    "I could not complete the request with the available evidence."
                )
            step = self.script[self.index]
            self.index += 1
        if callable(step):
            result = step(request)
            if not isinstance(result, StreamCompletion):
                raise TypeError("script callable must return StreamCompletion")
            return result
        return step

    def complete(self, request: ModelRequest) -> StreamCompletion:
        self.complete_calls += 1
        return self._next(request)

    def stream(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        emit_text: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> StreamCompletion:
        self.stream_calls += 1
        completion = self._next(request)
        committed = False
        if emit_text and on_text_delta and completion.content:
            # Chunk for incremental SSE realism without inventing content.
            text = completion.content
            parts = [text[i : i + 12] for i in range(0, len(text), 12)] or [text]
            for piece in parts:
                if should_stop is not None and should_stop():
                    from app.services.request_cancellation import RequestCancelled

                    raise RequestCancelled("provider_stream_cancelled")
                on_text_delta(piece)
                committed = True
        return StreamCompletion(
            content=completion.content,
            tool_calls=list(completion.tool_calls),
            finish_reason=completion.finish_reason,
            usage=completion.usage,
            text_committed=committed,
        )


@dataclass
class ToolCallRecord:
    name: str
    arguments: Any
    kwargs: dict[str, Any]
    result: Any


@dataclass
class E2ERecorder:
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    provider: ScriptedFakeProvider | None = None
    chat_with_ai_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tool_calls]

    @property
    def provider_invocations(self) -> int:
        if not self.provider:
            return 0
        return self.provider.complete_calls + self.provider.stream_calls


@dataclass
class E2EChatResult:
    status_code: int
    body: dict[str, Any]
    raw_text: str
    session_id: str | None
    reply: str
    citations: list[Any]
    context_used: bool
    rag_skipped: bool


@dataclass
class E2EStreamResult:
    status_code: int
    events: list[dict[str, Any]]
    raw_text: str
    tokens: str
    meta: dict[str, Any] | None
    done: dict[str, Any] | None
    error: dict[str, Any] | None


def parse_sse(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        if not block.strip() or block.lstrip().startswith(":"):
            continue
        eid = event_name = data_raw = None
        for line in block.split("\n"):
            if line.startswith("id:"):
                eid = line[3:].strip()
            elif line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_raw = line[5:].strip()
        if data_raw is None:
            continue
        try:
            data = json.loads(data_raw)
        except json.JSONDecodeError:
            data = {"raw": data_raw}
        events.append({"id": eid, "event": event_name, "data": data})
    return events


@dataclass
class ApiE2ESession:
    client: TestClient
    headers: dict[str, str]
    user: dict[str, Any]
    recorder: E2ERecorder
    provider: ScriptedFakeProvider
    memory_store: dict[str, Any]
    project_id: str | None
    tenant_id: str | None

    def post_chat(
        self,
        message: str,
        *,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        use_rag: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> E2EChatResult:
        payload: dict[str, Any] = {
            "message": message,
            "history": history or [],
            "useRag": use_rag,
        }
        if session_id:
            payload["sessionId"] = session_id
        if extra:
            payload.update(extra)
        res = self.client.post("/api/chat", headers=self.headers, json=payload)
        body: dict[str, Any] = {}
        try:
            body = res.json()
        except Exception:
            body = {}
        return E2EChatResult(
            status_code=res.status_code,
            body=body if isinstance(body, dict) else {},
            raw_text=res.text,
            session_id=(body or {}).get("sessionId"),
            reply=str((body or {}).get("reply") or ""),
            citations=list((body or {}).get("citations") or []),
            context_used=bool((body or {}).get("contextUsed")),
            rag_skipped=bool((body or {}).get("ragSkipped", True)),
        )

    def post_stream(
        self,
        message: str,
        *,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        use_rag: bool = True,
    ) -> E2EStreamResult:
        payload: dict[str, Any] = {
            "message": message,
            "history": history or [],
            "useRag": use_rag,
        }
        if session_id:
            payload["sessionId"] = session_id
        res = self.client.post(
            "/api/chat/stream", headers=self.headers, json=payload
        )
        events = parse_sse(res.text) if res.status_code == 200 else []
        tokens = "".join(
            str(e["data"].get("token", ""))
            for e in events
            if e.get("event") == "token"
        )
        meta = next((e["data"] for e in events if e.get("event") == "meta"), None)
        done = next((e["data"] for e in events if e.get("event") == "done"), None)
        error = next((e["data"] for e in events if e.get("event") == "error"), None)
        return E2EStreamResult(
            status_code=res.status_code,
            events=events,
            raw_text=res.text,
            tokens=tokens,
            meta=meta,
            done=done,
            error=error,
        )


@contextmanager
def api_e2e_session(
    *,
    user_id: str | None = None,
    tenant_id: str = "ten_batch_b",
    project_id: str | None = "proj_batch_b",
    email: str | None = None,
    script: list[ScriptStep] | None = None,
    ltm: dict[str, Any] | None = None,
    rag_result: dict[str, Any] | None = None,
    force_document_question: bool | None = None,
    wrap_execute_tool: bool = True,
    fake_web_search: Callable[..., Any] | None = None,
    fake_clock: str | None = "2026-11-26T04:52:00+00:00",
    hitl: bool = False,
    approvals_collection: Any | None = None,
    permissions_collection: Any | None = None,
    extra_patches: list[Any] | None = None,
) -> Iterator[ApiE2ESession]:
    """Open an authenticated API session with a scripted fake LLM provider."""
    from app import chatbot
    from app.services import approval_service, tool_permission_service

    uid = user_id or f"usr_bb_{uuid.uuid4().hex[:10]}"
    mail = email or f"{uid}@batch-b.test"
    user = {
        "_id": uid,
        "email": mail,
        "status": "active",
        "tenant_id": tenant_id,
        "display_name": f"BatchB {uid}",
    }
    token = create_access_token(
        user_id=uid,
        email=mail,
        tenant_id=tenant_id,
        session_id=f"sess_{uid}",
    )
    headers = {"Authorization": f"Bearer {token}"}
    provider = ScriptedFakeProvider(script or [text_completion("OK.")])
    recorder = E2ERecorder(provider=provider)
    memory_store = ltm or {"memories": [], "conflicts": [], "summaries": []}

    env_backup: dict[str, str | None] = {}
    for key in (
        "AGENT_HITL_ENABLED",
        "AGENT_TOOL_PERMISSIONS_ENFORCE",
        "AGENT_SECURITY_POLICY_VERSION",
    ):
        env_backup[key] = os.environ.get(key)

    if hitl:
        os.environ["AGENT_HITL_ENABLED"] = "true"
        os.environ["AGENT_TOOL_PERMISSIONS_ENFORCE"] = "true"
        os.environ["AGENT_SECURITY_POLICY_VERSION"] = "1"
        if approvals_collection is not None:
            approval_service.configure_approval_collection(approvals_collection)
        if permissions_collection is not None:
            tool_permission_service.configure_permission_collection(
                permissions_collection
            )
            tool_permission_service.invalidate_permission_cache()

    def _recording_execute(name, args, **kwargs):
        result = _real_execute(name, args, **kwargs)
        recorder.tool_calls.append(
            ToolCallRecord(
                name=name,
                arguments=args,
                kwargs=dict(kwargs),
                result=result,
            )
        )
        return result

    _real_execute = chatbot.execute_tool

    def _wrapped_chat_with_ai(*args, **kwargs):
        recorder.chat_with_ai_calls.append(
            {
                "message": args[0] if args else kwargs.get("message"),
                "user_id": kwargs.get("user_id"),
                "request_id": kwargs.get("request_id"),
                "has_rag": bool(kwargs.get("rag_context")),
            }
        )
        return _real_chat_with_ai(*args, **kwargs)

    _real_chat_with_ai = chatbot.chat_with_ai

    patches: list[Any] = [
        patch("app.api.deps.get_user_by_id", return_value=user),
        patch("app.api.deps.auth_service.get_active_session", return_value=True),
        patch("app.api.routes.chat.enforce_chat_rate_limit"),
        patch(
            "app.services.memory_scope_service.get_active_project_scope",
            return_value={"project_id": project_id, "tenant_id": tenant_id},
        ),
        # chat_with_ai imports get_llm_provider from the package each call.
        patch(
            "app.services.llm_provider.get_llm_provider",
            return_value=provider,
        ),
        patch(
            "app.services.llm_provider.factory.get_llm_provider",
            return_value=provider,
        ),
        patch.object(chatbot, "load_long_term_memory", return_value=memory_store),
        patch.object(chatbot, "restore_task_context_from_redis"),
        patch.object(chatbot, "save_task_context_to_redis"),
        # Keep explicit memory helpers off unless a scenario opts into real tools.
        patch.object(chatbot, "try_save_explicit_memory", return_value=False),
        patch.object(chatbot, "try_update_explicit_memory", return_value=None),
        patch.object(chatbot, "try_delete_explicit_memory", return_value=None),
        patch.object(chatbot, "try_inspect_memory_metadata", return_value=None),
        # Count agent-loop entries without replacing the loop.
        patch(
            "app.api.routes.chat.chat_with_ai",
            side_effect=_wrapped_chat_with_ai,
        ),
    ]

    if wrap_execute_tool:
        patches.append(
            patch.object(chatbot, "execute_tool", side_effect=_recording_execute)
        )

    if force_document_question is not None:
        patches.append(
            patch(
                "app.api.routes.chat.is_document_question",
                return_value=force_document_question,
            )
        )

    if rag_result is not None:
        patches.append(
            patch("app.api.routes.chat.retrieve", return_value=rag_result)
        )

    if fake_clock is not None:
        def _fake_time_tool(**_kwargs):
            # Match time_tool contract: plain formatted timestamp string.
            return str(fake_clock)

        patches.append(
            patch.dict(
                chatbot.tools,
                {"get_current_time": _fake_time_tool},
                clear=False,
            )
        )

    if fake_web_search is not None:
        patches.append(
            patch.dict(
                chatbot.tools,
                {"web_search": fake_web_search},
                clear=False,
            )
        )

    if hitl:
        patches.append(
            patch(
                "app.services.agent_security_policy.hitl_enabled",
                return_value=True,
            )
        )

    if extra_patches:
        patches.extend(extra_patches)

    reset_llm_provider_cache()
    started = [p.start() for p in patches]
    del started
    client = TestClient(app)
    try:
        chatbot._get_conversation_store(uid).clear()
        yield ApiE2ESession(
            client=client,
            headers=headers,
            user=user,
            recorder=recorder,
            provider=provider,
            memory_store=memory_store,
            project_id=project_id,
            tenant_id=tenant_id,
        )
    finally:
        for p in reversed(patches):
            p.stop()
        reset_llm_provider_cache()
        try:
            chatbot._get_conversation_store(uid).clear()
        except Exception:
            pass
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if hitl:
            approval_service.configure_approval_collection(None)
            tool_permission_service.configure_permission_collection(None)
            tool_permission_service.invalidate_permission_cache()


def assert_no_unauthorized_tools(
    recorder: E2ERecorder, *, allowed: set[str] | frozenset[str]
) -> None:
    unexpected = [n for n in recorder.tool_names if n not in allowed]
    if unexpected:
        raise AssertionError(f"Unexpected tools invoked: {unexpected}")
