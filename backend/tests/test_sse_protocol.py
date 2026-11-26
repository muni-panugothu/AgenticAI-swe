"""Topic 42a — SSE protocol framing and chat stream integration tests."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.services.agent_output import (
    build_answer,
    build_error,
    build_from_agent_reply,
)
from app.services.sse_protocol import (
    citations_for_stream,
    format_error_event,
    format_sse_frame,
    iter_agent_turn_sse_events,
    iter_sse_with_guard,
)
from app.validation.agent_output_models import AgentTurnResult


def _parse_sse(body: str) -> list[dict]:
    """Parse framed SSE into {id, event, data} dicts."""
    events: list[dict] = []
    for block in body.split("\n\n"):
        if not block.strip() or block.strip().startswith(":"):
            continue
        eid = None
        event_name = None
        data_raw = None
        for line in block.split("\n"):
            if line.startswith("id:"):
                eid = line[3:].strip()
            elif line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_raw = line[5:].strip()
        if data_raw is None:
            continue
        payload = json.loads(data_raw)
        events.append({"id": eid, "event": event_name, "data": payload})
    return events


class SseFramingTests(unittest.TestCase):
    def test_frame_has_id_event_data_and_blank_line(self):
        frame = format_sse_frame(
            event="meta",
            data={"type": "meta", "sessionId": "s1"},
            event_id=1,
        )
        self.assertTrue(frame.endswith("\n\n"))
        self.assertIn("id: 1\n", frame)
        self.assertIn("event: meta\n", frame)
        self.assertIn("data: ", frame)
        self.assertIn('"type":"meta"', frame)

    def test_meta_before_tokens_and_single_done(self):
        turn = build_answer("Hello world")
        frames = list(
            iter_agent_turn_sse_events(
                turn=turn,
                session_id="sess-a",
                action="general",
                context_used=False,
                rag_skipped=True,
                rest_citations=[],
                chunk_size=5,
            )
        )
        events = _parse_sse("".join(frames))
        self.assertEqual(events[0]["event"], "meta")
        self.assertEqual(events[0]["data"]["type"], "meta")
        token_events = [e for e in events if e["event"] == "token"]
        self.assertGreaterEqual(len(token_events), 2)
        done_events = [e for e in events if e["event"] == "done"]
        self.assertEqual(len(done_events), 1)
        self.assertEqual(events[-1]["event"], "done")
        # Stable increasing ids within the request
        ids = [int(e["id"]) for e in events]
        self.assertEqual(ids, list(range(1, len(events) + 1)))

    def test_empty_reply_still_emits_meta_and_done(self):
        turn = build_answer("")
        events = _parse_sse(
            "".join(
                iter_agent_turn_sse_events(
                    turn=turn,
                    session_id="empty",
                    rest_citations=[],
                )
            )
        )
        self.assertEqual([e["event"] for e in events], ["meta", "done"])
        self.assertEqual(events[0]["data"]["sessionId"], "empty")

    def test_long_reply_chunking(self):
        text = "A" * 120
        turn = build_answer(text)
        events = _parse_sse(
            "".join(
                iter_agent_turn_sse_events(
                    turn=turn,
                    session_id="long",
                    rest_citations=[],
                    chunk_size=48,
                )
            )
        )
        tokens = [e["data"]["token"] for e in events if e["event"] == "token"]
        self.assertEqual("".join(tokens), text)
        self.assertEqual(len(tokens), 3)

    def test_structured_status_from_agent_turn_result(self):
        reply = (
            "Draft.\n\nNote: This request involved multiple steps "
            "(1 completed, 1 not fully finished). Treat the answer above as partial "
            "where work could not be fully verified."
        )
        turn = build_from_agent_reply(reply)
        self.assertEqual(turn.response_type, "partial")
        events = _parse_sse(
            "".join(
                iter_agent_turn_sse_events(
                    turn=turn,
                    session_id="partial",
                    rest_citations=[],
                )
            )
        )
        meta = events[0]["data"]
        self.assertEqual(meta["responseType"], "partial")
        self.assertEqual(meta["completionStatus"], "partial")
        self.assertTrue(meta["partial"])
        done = events[-1]["data"]
        self.assertEqual(done["completionStatus"], "partial")
        self.assertEqual(done["responseType"], "partial")

    def test_stream_citations_from_server_rag_not_llm_text(self):
        rest = [{"document_id": "doc-srv", "filename": "a.md"}]
        turn = build_from_agent_reply(
            'See {"documentId":"hacked-from-llm"}',
            citations=rest,
        )
        events = _parse_sse(
            "".join(
                iter_agent_turn_sse_events(
                    turn=turn,
                    session_id="cite",
                    rest_citations=rest,
                    context_used=True,
                    rag_skipped=False,
                )
            )
        )
        meta = events[0]["data"]
        self.assertEqual(meta["citations"], rest)
        self.assertEqual(meta["citations"][0]["document_id"], "doc-srv")
        stream_cites = meta["streamCitations"]
        self.assertEqual(stream_cites[0]["documentId"], "doc-srv")
        self.assertNotIn("hacked-from-llm", json.dumps(meta))

    def test_error_turn_is_terminal_without_done(self):
        turn = build_error(
            "Something went wrong.",
            error_code="CHAT_ERROR",
            error_message_safe="Chat processing failed",
        )
        events = _parse_sse(
            "".join(
                iter_agent_turn_sse_events(
                    turn=turn,
                    session_id="err",
                    rest_citations=[],
                )
            )
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "error")
        self.assertEqual(events[0]["data"]["code"], "CHAT_ERROR")
        self.assertNotIn("done", [e["event"] for e in events])

    def test_guard_emits_safe_error_without_raw_exception(self):
        def bad_frames():
            yield format_sse_frame(
                event="meta",
                data={"type": "meta", "sessionId": "x"},
                event_id=1,
            )
            raise RuntimeError("secret JWT eyJhbGciOiJIUzI1NiIs stack trace /tmp/key")

        body = "".join(iter_sse_with_guard(bad_frames()))
        self.assertNotIn("JWT", body)
        self.assertNotIn("stack", body.lower())
        self.assertNotIn("eyJhbGci", body)
        self.assertNotIn("/tmp/key", body)
        events = _parse_sse(body)
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(events[-1]["data"]["code"], "STREAM_ERROR")
        self.assertNotIn("done", [e["event"] for e in events])

    def test_error_event_helper_payload(self):
        frame = format_error_event(event_id=9, code="STREAM_ERROR")
        events = _parse_sse(frame)
        self.assertEqual(events[0]["id"], "9")
        self.assertEqual(events[0]["data"]["type"], "error")
        self.assertNotIn("Exception", events[0]["data"]["error"])

    def test_citations_for_stream_camel_case(self):
        turn = AgentTurnResult(
            reply="x",
            citations=[
                {
                    "index": 0,
                    "document_id": "d1",
                    "chunk_index": 2,
                    "excerpt": "hi",
                    "score": 0.5,
                }
            ],
        )
        cites = citations_for_stream(turn)
        self.assertEqual(cites[0]["documentId"], "d1")
        self.assertEqual(cites[0]["chunkIndex"], 2)


class SseIntegrationTests(unittest.TestCase):
    def setUp(self):
        import os

        os.environ.setdefault(
            "JWT_SECRET", "e2e-test-jwt-secret-key-32chars-min!!"
        )
        from app.core.security import create_access_token
        from app.main import app

        self.client = TestClient(app)
        self.user = {
            "_id": "usr_sse",
            "email": "sse@example.com",
            "status": "active",
            "tenant_id": "ten_sse",
            "display_name": "SSE",
        }
        token = create_access_token(
            user_id="usr_sse",
            email="sse@example.com",
            tenant_id="ten_sse",
            session_id="sess_sse",
        )
        self.headers = {"Authorization": f"Bearer {token}"}

    def test_stream_success_sequence_and_single_agent_call(self):
        with patch("app.api.deps.get_user_by_id", return_value=self.user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.is_document_question", return_value=False
        ), patch(
            "app.api.routes.chat.chat_with_ai",
            return_value="Stream hello from agent",
        ) as mock_chat:
            res = self.client.post(
                "/api/chat/stream",
                headers=self.headers,
                json={"message": "hi there"},
            )

        self.assertEqual(res.status_code, 200, res.text)
        self.assertIn("text/event-stream", res.headers.get("content-type", ""))
        events = _parse_sse(res.text)
        self.assertEqual(events[0]["event"], "meta")
        # Live stream (42b): final classification is on done; meta may omit it.
        done_events = [e for e in events if e["event"] == "done"]
        self.assertEqual(len(done_events), 1)
        self.assertEqual(done_events[0]["data"]["responseType"], "answer")
        self.assertEqual(done_events[0]["data"]["completionStatus"], "complete")
        tokens = "".join(
            e["data"]["token"] for e in events if e["event"] == "token"
        )
        self.assertEqual(tokens, "Stream hello from agent")
        mock_chat.assert_called_once()

    def test_rest_shape_and_citations_unchanged(self):
        citations = [{"document_id": "doc1", "filename": "h.md"}]
        rag_result = {
            "has_readable_documents": True,
            "candidates": [{"document_id": "doc1"}],
            "context": "Retention is 90 days.",
            "citations": citations,
        }
        with patch("app.api.deps.get_user_by_id", return_value=self.user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.is_document_question", return_value=True
        ), patch(
            "app.api.routes.chat.retrieve", return_value=rag_result
        ), patch(
            "app.services.memory_scope_service.get_active_project_scope",
            return_value={"project_id": "proj_sse"},
        ), patch(
            "app.api.routes.chat.chat_with_ai",
            return_value="Retention is 90 days.",
        ) as mock_chat:
            rest = self.client.post(
                "/api/chat",
                headers=self.headers,
                json={"message": "According to the handbook, retention?"},
            )
            stream = self.client.post(
                "/api/chat/stream",
                headers=self.headers,
                json={"message": "According to the handbook, retention?"},
            )

        self.assertEqual(rest.status_code, 200, rest.text)
        body = rest.json()
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
        self.assertEqual(body["citations"][0]["document_id"], "doc1")
        self.assertNotIn("documentId", body["citations"][0])

        self.assertEqual(stream.status_code, 200, stream.text)
        meta = _parse_sse(stream.text)[0]["data"]
        self.assertEqual(meta["citations"][0]["document_id"], "doc1")
        self.assertEqual(meta["streamCitations"][0]["documentId"], "doc1")
        # One agent call per request (REST + stream = 2 total).
        self.assertEqual(mock_chat.call_count, 2)

    def test_stream_auth_required(self):
        res = self.client.post("/api/chat/stream", json={"message": "no auth"})
        self.assertIn(res.status_code, (401, 403))

    def test_agent_failure_after_meta_is_terminal_error_event(self):
        """Topic 42b: stream opens after auth/RAG; agent failures → SSE error."""
        with patch("app.api.deps.get_user_by_id", return_value=self.user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.is_document_question", return_value=False
        ), patch(
            "app.api.routes.chat.chat_with_ai",
            side_effect=RuntimeError("internal boom with secret"),
        ):
            res = self.client.post(
                "/api/chat/stream",
                headers=self.headers,
                json={"message": "fail please"},
            )

        self.assertEqual(res.status_code, 200, res.text)
        events = _parse_sse(res.text)
        self.assertEqual(events[0]["event"], "meta")
        self.assertEqual(events[-1]["event"], "error")
        self.assertNotIn("done", [e["event"] for e in events])
        self.assertNotIn("secret", res.text)
        self.assertIn("CHAT_ERROR", res.text)

    def test_rag_failure_before_stream_is_http_error(self):
        with patch("app.api.deps.get_user_by_id", return_value=self.user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.is_document_question", return_value=True
        ), patch(
            "app.services.memory_scope_service.get_active_project_scope",
            return_value={"project_id": None},
        ), patch(
            "app.api.routes.chat.retrieve",
            side_effect=RuntimeError("rag secret failure"),
        ):
            res = self.client.post(
                "/api/chat/stream",
                headers=self.headers,
                json={"message": "According to the handbook document?"},
            )
        self.assertEqual(res.status_code, 503)
        dumped = res.text
        self.assertNotIn("secret", dumped)
        self.assertIn("RAG_RETRIEVAL_FAILED", dumped)

    def test_unknown_additive_events_do_not_break_web_parser_contract(self):
        """Web parseStreamDataLine returns null for unknown types — mirrored here."""
        unknown = {"type": "heartbeat", "ts": 1}
        self.assertNotIn(unknown["type"], {"meta", "token", "done", "error"})


class ExpoStyleClientParseTests(unittest.TestCase):
    """Mirrors Expo multi-line SSE consumption (data: line only)."""

    def test_ignores_unknown_and_reads_multiline_frames(self):
        turn = build_answer("AB")
        body = "".join(
            iter_agent_turn_sse_events(
                turn=turn,
                session_id="expo",
                rest_citations=[],
                chunk_size=1,
            )
        )
        # Inject an additive unknown event between tokens.
        extra = format_sse_frame(
            event="structured",
            data={"type": "structured", "responseType": "answer"},
            event_id=99,
        )
        parts = body.split("\n\n")
        # Insert after meta
        rebuilt = "\n\n".join([parts[0], extra.rstrip("\n"), *parts[1:]]) + "\n\n"

        full = ""
        types: list[str] = []
        for block in rebuilt.split("\n\n"):
            json_line = None
            for raw in block.split("\n"):
                if raw.startswith("data:"):
                    json_line = raw[5:].strip()
            if not json_line:
                continue
            evt = json.loads(json_line)
            if evt.get("type") in {"meta", "token", "done", "error"}:
                types.append(evt["type"])
                if evt["type"] == "token":
                    full += evt["token"]
            # unknown ignored
        self.assertEqual(types[0], "meta")
        self.assertEqual(full, "AB")
        self.assertEqual(types[-1], "done")
        self.assertNotIn("structured", types)


if __name__ == "__main__":
    unittest.main()
