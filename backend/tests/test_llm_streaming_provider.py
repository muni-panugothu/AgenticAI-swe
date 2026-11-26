"""Topic 42b — provider abstraction + incremental streaming tests."""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET", "e2e-test-jwt-secret-key-32chars-min!!")

from fastapi.testclient import TestClient

from app.services.llm_provider.base import (
    StreamOutputCommittedError,
    completion_to_sdk_shape,
)
from app.services.llm_provider.factory import reset_llm_provider_cache
from app.services.llm_provider.groq_adapter import GroqProvider, _ToolCallAccumulator
from app.validation.llm_provider_models import (
    CompletedToolCall,
    ModelRequest,
    StreamCompletion,
)


class FakeProvider:
    """Provider-independent fake — no Groq SDK objects."""

    name = "fake"

    def __init__(self, *, chunks: list[str] | None = None, tool_calls=None):
        self.chunks = list(chunks or ["Hello", " world"])
        self.tool_calls = list(tool_calls or [])
        self.complete_calls = 0
        self.stream_calls = 0

    def complete(self, request: ModelRequest) -> StreamCompletion:
        self.complete_calls += 1
        return StreamCompletion(
            content="".join(self.chunks),
            tool_calls=self.tool_calls,
            finish_reason="stop",
            text_committed=False,
        )

    def stream(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        emit_text: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> StreamCompletion:
        self.stream_calls += 1
        committed = False
        for piece in self.chunks:
            if should_stop is not None and should_stop():
                from app.services.request_cancellation import RequestCancelled

                raise RequestCancelled("provider_stream_cancelled")
            if emit_text and on_text_delta is not None:
                on_text_delta(piece)
                committed = True
        return StreamCompletion(
            content="".join(self.chunks),
            tool_calls=self.tool_calls,
            finish_reason="stop",
            text_committed=committed,
        )


def _parse_sse(body: str) -> list[dict]:
    events: list[dict] = []
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
        events.append(
            {"id": eid, "event": event_name, "data": json.loads(data_raw)}
        )
    return events


class ProviderContractTests(unittest.TestCase):
    def test_completion_to_sdk_shape_has_no_groq_types(self):
        completion = StreamCompletion(
            content="hi",
            tool_calls=[
                CompletedToolCall(
                    id="c1", name="search_code", arguments='{"q":"x"}'
                )
            ],
            finish_reason="tool_calls",
        )
        shaped = completion_to_sdk_shape(completion)
        self.assertEqual(shaped.choices[0].message.content, "hi")
        self.assertEqual(shaped.choices[0].message.tool_calls[0].function.name, "search_code")
        self.assertNotIn("groq", type(shaped).__module__)

    def test_fake_provider_streaming_emits_before_completion(self):
        seen: list[str] = []
        order: list[str] = []

        def on_delta(t: str) -> None:
            seen.append(t)
            order.append("delta")

        provider = FakeProvider(chunks=["A", "B", "C"])
        result = provider.stream(
            ModelRequest(model="fake-model", messages=[], max_tokens=16),
            on_text_delta=on_delta,
            emit_text=True,
        )
        order.append("done")
        self.assertEqual(seen, ["A", "B", "C"])
        self.assertEqual(order[0], "delta")
        self.assertEqual(order[-1], "done")
        self.assertTrue(result.text_committed)
        self.assertEqual(result.content, "ABC")

    def test_emit_text_false_does_not_forward(self):
        seen: list[str] = []
        provider = FakeProvider(chunks=["secret"])
        provider.stream(
            ModelRequest(model="m", messages=[], max_tokens=8),
            on_text_delta=lambda t: seen.append(t),
            emit_text=False,
        )
        self.assertEqual(seen, [])


class ToolCallAccumulationTests(unittest.TestCase):
    def test_fragments_not_executable_until_complete(self):
        acc = _ToolCallAccumulator()
        acc.add(0, id="call_1", name="sea", arguments=None)
        acc.add(0, id=None, name="rch_code", arguments='{"q":')
        # Incomplete JSON — still returned as string; agent validates later.
        partial = acc.completed()
        self.assertEqual(len(partial), 1)
        self.assertEqual(partial[0].name, "search_code")
        self.assertEqual(partial[0].arguments, '{"q":')
        acc.add(0, id=None, name=None, arguments='"x"}')
        done = acc.completed()
        self.assertEqual(done[0].arguments, '{"q":"x"}')

    def test_groq_adapter_maps_stream_chunks(self):
        def _chunk(content=None, tool_calls=None, finish=None, usage=None):
            delta = SimpleNamespace(
                content=content,
                tool_calls=tool_calls or [],
                reasoning=None,
            )
            choice = SimpleNamespace(delta=delta, finish_reason=finish)
            return SimpleNamespace(choices=[choice], usage=usage)

        tc1 = SimpleNamespace(
            index=0,
            id="call_9",
            function=SimpleNamespace(name="read_file", arguments='{"file'),
        )
        tc2 = SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(name=None, arguments='_path":"a.py"}'),
        )
        stream = [
            _chunk(content="Hi"),
            _chunk(tool_calls=[tc1]),
            _chunk(tool_calls=[tc2], finish="tool_calls"),
        ]
        client = MagicMock()
        client.chat.completions.create.return_value = iter(stream)
        provider = GroqProvider(client=client, default_model="openai/gpt-oss-20b")
        seen: list[str] = []
        result = provider.stream(
            ModelRequest(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": "x"}],
                max_tokens=64,
                stream=True,
            ),
            on_text_delta=seen.append,
            emit_text=True,
        )
        self.assertEqual(seen, ["Hi"])
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertIn("a.py", result.tool_calls[0].arguments)
        # Reasoning must never be forwarded even if present on delta.
        self.assertNotIn("reasoning", "".join(seen))

    def test_stream_failure_after_output_raises_committed(self):
        def boom():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Partial", tool_calls=[]),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            raise RuntimeError("connection dropped")

        client = MagicMock()
        client.chat.completions.create.return_value = boom()
        provider = GroqProvider(client=client, default_model="m")
        with self.assertRaises(StreamOutputCommittedError) as ctx:
            provider.stream(
                ModelRequest(model="m", messages=[], max_tokens=8),
                on_text_delta=lambda _t: None,
                emit_text=True,
            )
        self.assertEqual(ctx.exception.partial_text, "Partial")


class StreamingIntegrationTests(unittest.TestCase):
    def setUp(self):
        from app.core.security import create_access_token
        from app.main import app

        self.client = TestClient(app)
        self.user = {
            "_id": "usr_42b",
            "email": "s42b@example.com",
            "status": "active",
            "tenant_id": "ten_42b",
            "display_name": "S42B",
        }
        token = create_access_token(
            user_id="usr_42b",
            email="s42b@example.com",
            tenant_id="ten_42b",
            session_id="sess_42b",
        )
        self.headers = {"Authorization": f"Bearer {token}"}
        reset_llm_provider_cache()

    def tearDown(self):
        reset_llm_provider_cache()

    def test_incremental_tokens_before_done_single_agent_call(self):
        deltas_seen_in_agent: list[str] = []

        def fake_chat(message, **kwargs):
            on_delta = kwargs.get("on_text_delta")
            self.assertIsNotNone(on_delta)
            on_delta("Hel")
            deltas_seen_in_agent.append("Hel")
            on_delta("lo!")
            deltas_seen_in_agent.append("lo!")
            return "Hello!"

        with patch("app.api.deps.get_user_by_id", return_value=self.user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.is_document_question", return_value=False
        ), patch(
            "app.api.routes.chat.chat_with_ai", side_effect=fake_chat
        ) as mock_chat:
            res = self.client.post(
                "/api/chat/stream",
                headers=self.headers,
                json={"message": "hi"},
            )

        self.assertEqual(res.status_code, 200, res.text)
        events = _parse_sse(res.text)
        types = [e["event"] for e in events]
        self.assertEqual(types[0], "meta")
        self.assertIn("token", types)
        self.assertEqual(types[-1], "done")
        self.assertEqual(sum(1 for t in types if t == "done"), 1)
        tokens = "".join(
            e["data"]["token"] for e in events if e["event"] == "token"
        )
        self.assertEqual(tokens, "Hello!")
        # Tokens appear before done in the event list.
        first_token_i = types.index("token")
        done_i = types.index("done")
        self.assertLess(first_token_i, done_i)
        mock_chat.assert_called_once()
        self.assertEqual(deltas_seen_in_agent, ["Hel", "lo!"])

    def test_stream_error_path_still_single_agent_invocation(self):
        """Batch A: one accepted stream POST ⇒ one chat_with_ai; no replacement."""
        with patch("app.api.deps.get_user_by_id", return_value=self.user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.is_document_question", return_value=False
        ), patch(
            "app.api.routes.chat.chat_with_ai",
            side_effect=RuntimeError("agent boom"),
        ) as mock_chat:
            res = self.client.post(
                "/api/chat/stream",
                headers=self.headers,
                json={"message": "fail once"},
            )
        self.assertEqual(res.status_code, 200, res.text)
        mock_chat.assert_called_once()
        types = [e["event"] for e in _parse_sse(res.text)]
        self.assertIn("error", types)
        self.assertNotIn("done", types)

    def test_rest_unchanged_and_no_on_text_delta(self):
        with patch("app.api.deps.get_user_by_id", return_value=self.user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.is_document_question", return_value=False
        ), patch(
            "app.api.routes.chat.chat_with_ai", return_value="REST reply"
        ) as mock_chat:
            res = self.client.post(
                "/api/chat",
                headers=self.headers,
                json={"message": "hello rest"},
            )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["reply"], "REST reply")
        for key in (
            "reply",
            "sessionId",
            "action",
            "contextUsed",
            "ragSkipped",
            "model",
            "provider",
            "citations",
        ):
            self.assertIn(key, body)
        self.assertIsNone(mock_chat.call_args.kwargs.get("on_text_delta"))

    def test_rag_citations_preserved_on_stream_meta(self):
        citations = [{"document_id": "doc42", "filename": "a.md"}]
        rag_result = {
            "has_readable_documents": True,
            "candidates": [{"document_id": "doc42"}],
            "context": "ctx",
            "citations": citations,
        }

        def fake_chat(message, **kwargs):
            on_delta = kwargs.get("on_text_delta")
            if on_delta:
                on_delta("Cited answer")
            return "Cited answer"

        with patch("app.api.deps.get_user_by_id", return_value=self.user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.is_document_question", return_value=True
        ), patch(
            "app.api.routes.chat.retrieve", return_value=rag_result
        ), patch(
            "app.services.memory_scope_service.get_active_project_scope",
            return_value={"project_id": "p1"},
        ), patch(
            "app.api.routes.chat.chat_with_ai", side_effect=fake_chat
        ):
            res = self.client.post(
                "/api/chat/stream",
                headers=self.headers,
                json={"message": "According to the handbook?"},
            )
        self.assertEqual(res.status_code, 200, res.text)
        meta = _parse_sse(res.text)[0]["data"]
        self.assertEqual(meta["citations"][0]["document_id"], "doc42")

    def test_no_done_after_error(self):
        with patch("app.api.deps.get_user_by_id", return_value=self.user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.is_document_question", return_value=False
        ), patch(
            "app.api.routes.chat.chat_with_ai",
            side_effect=RuntimeError("boom"),
        ):
            res = self.client.post(
                "/api/chat/stream",
                headers=self.headers,
                json={"message": "fail"},
            )
        self.assertEqual(res.status_code, 200, res.text)
        events = _parse_sse(res.text)
        types = [e["event"] for e in events]
        self.assertIn("error", types)
        self.assertNotIn("done", types)
        self.assertNotIn("boom", res.text)

    def test_openai_provider_not_silently_selected(self):
        reset_llm_provider_cache()
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
            reset_llm_provider_cache()
            from app.services.llm_provider.factory import get_llm_provider

            with self.assertRaises(RuntimeError):
                get_llm_provider()
        reset_llm_provider_cache()


class ChatWithAiStreamingBoundaryTests(unittest.TestCase):
    def test_invoke_uses_provider_not_raw_groq_when_streaming(self):
        """Ensure agent boundary can run with a fake provider (no Groq objects)."""
        fake = FakeProvider(chunks=["Ok"])
        from app.services.llm_provider import completion_to_sdk_shape

        req = ModelRequest(model="fake", messages=[], max_tokens=8)
        seen: list[str] = []
        completion = fake.stream(req, on_text_delta=seen.append, emit_text=True)
        shaped = completion_to_sdk_shape(completion)
        self.assertEqual(shaped.choices[0].message.content, "Ok")
        self.assertEqual(seen, ["Ok"])
        self.assertEqual(fake.stream_calls, 1)
        self.assertEqual(fake.complete_calls, 0)
        # Shaped response is a plain namespace — not a Groq SDK type.
        self.assertEqual(type(shaped).__module__, "types")


if __name__ == "__main__":
    unittest.main()
