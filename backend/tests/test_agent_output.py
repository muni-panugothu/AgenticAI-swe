"""Topic 41a — trusted AgentTurnResult foundation tests."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.api.routes.chat import ChatResponseBody
from app.services.agent_output import (
    RAG_ABSTAIN_REPLY,
    build_aborted,
    build_abstain,
    build_answer,
    build_clarification,
    build_error,
    build_from_agent_reply,
    build_partial,
    citations_from_rag,
    reject_untrusted_model_payload,
    turn_result_to_chat_fields,
)
from app.validation.agent_output_models import AgentTurnResult, CitationRef


class AgentOutputModelTests(unittest.TestCase):
    def test_valid_normal_answer(self):
        result = build_answer("Hello world")
        self.assertEqual(result.reply, "Hello world")
        self.assertEqual(result.response_type, "answer")
        self.assertEqual(result.completion_status, "complete")
        self.assertFalse(result.partial)
        self.assertFalse(result.clarification_required)

    def test_clarification_response(self):
        result = build_clarification("Which file did you mean?")
        self.assertEqual(result.response_type, "clarification")
        self.assertEqual(result.completion_status, "clarification_needed")
        self.assertTrue(result.clarification_required)
        self.assertTrue(result.uncertain)

    def test_rag_abstention(self):
        result = build_abstain()
        self.assertEqual(result.reply, RAG_ABSTAIN_REPLY)
        self.assertEqual(result.response_type, "abstain")
        self.assertEqual(result.completion_status, "complete")
        self.assertEqual(result.citations, [])

    def test_partial_and_b6_incomplete(self):
        reply = (
            "Here is a draft.\n\nNote: This request involved multiple steps "
            "(1 completed, 1 not fully finished). Treat the answer above as partial "
            "where work could not be fully verified."
        )
        result = build_from_agent_reply(reply)
        self.assertEqual(result.response_type, "partial")
        self.assertTrue(result.partial)
        self.assertEqual(result.completion_status, "partial")
        self.assertEqual(result.reply, reply)

    def test_budget_exhaustion(self):
        reply = "The AI service is temporarily rate-limited (TPM). Please wait."
        result = build_from_agent_reply(reply, budget_exhausted=True)
        self.assertEqual(result.response_type, "partial")
        self.assertTrue(result.partial)
        result2 = build_from_agent_reply(reply)
        self.assertEqual(result2.response_type, "partial")

    def test_aborted_response(self):
        result = build_aborted("Generation stopped.")
        self.assertEqual(result.completion_status, "aborted")
        self.assertEqual(result.error_code, "ABORTED")
        self.assertTrue(result.partial)

    def test_server_owned_citations(self):
        raw = [
            {
                "index": 0,
                "documentId": "doc-1",
                "source": "guide.pdf",
                "section": "Intro",
                "chunkIndex": 2,
                "excerpt": "Hello",
                "score": 0.91,
            }
        ]
        result = build_answer("Based on the doc.", citations=raw)
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].document_id, "doc-1")
        self.assertEqual(result.citations[0].chunk_index, 2)

    def test_llm_fake_citations_not_trusted_from_text(self):
        # Builder never scrapes citations out of reply text.
        reply = 'According to [1] doc-evil, see {"documentId":"hacked"}'
        result = build_from_agent_reply(reply, citations=None)
        self.assertEqual(result.citations, [])

    def test_invalid_response_type_rejected(self):
        with self.assertRaises(ValidationError):
            AgentTurnResult(reply="x", response_type="nonsense")  # type: ignore[arg-type]

    def test_invalid_completion_status_rejected(self):
        with self.assertRaises(ValidationError):
            AgentTurnResult(reply="x", completion_status="done")  # type: ignore[arg-type]

    def test_unknown_trusted_fields_rejected(self):
        with self.assertRaises(ValidationError):
            AgentTurnResult.model_validate(
                {"reply": "ok", "user_id": "attacker", "tool_allowlist": ["x"]}
            )

    def test_missing_reply_handled(self):
        # None → empty string via builders; model requires reply key when validated raw.
        result = build_from_agent_reply(None)
        self.assertEqual(result.reply, "")
        with self.assertRaises(ValidationError):
            AgentTurnResult.model_validate({})

    def test_no_mutable_default_leakage(self):
        a = build_answer("a")
        b = build_answer("b")
        self.assertIsNot(a.citations, b.citations)
        # frozen models cannot be mutated
        with self.assertRaises(Exception):
            a.reply = "mutated"  # type: ignore[misc]

    def test_original_reply_text_preserved(self):
        text = "Exact reply ✨ with spaces  "
        self.assertEqual(build_answer(text).reply, text)
        self.assertEqual(build_from_agent_reply(text).reply, text)

    def test_api_response_shape_unchanged(self):
        # Public API keeps server RAG citation dicts as-is (snake_case keys).
        server_citations = [
            {"document_id": "d1", "filename": "guide.md", "index": 0}
        ]
        turn = build_answer("Hi", citations=server_citations)
        body = ChatResponseBody(
            reply=turn.reply,
            sessionId="sess-1",
            action="general",
            contextUsed=True,
            ragSkipped=False,
            citations=server_citations,
        )
        dumped = body.model_dump()
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
            self.assertIn(key, dumped)
        self.assertEqual(dumped["reply"], "Hi")
        self.assertEqual(dumped["citations"][0]["document_id"], "d1")
        # Internal SSE helper may normalize; must not be required for API parity.
        fields = turn_result_to_chat_fields(turn)
        self.assertEqual(fields["reply"], "Hi")
        self.assertEqual(fields["citations"][0]["documentId"], "d1")

    def test_reject_untrusted_authorization_payload(self):
        with self.assertRaises(ValueError):
            reject_untrusted_model_payload({"user_id": "u1", "reply": "x"})
        reject_untrusted_model_payload({"summary": "harmless"})

    def test_structured_output_cannot_set_b6_fields(self):
        with self.assertRaises(ValidationError):
            AgentTurnResult.model_validate(
                {
                    "reply": "done",
                    "b6_complete": True,
                    "active_task_id": "t1",
                    "task_status": "completed",
                }
            )

    def test_citations_from_rag_skips_bad_rows(self):
        refs = citations_from_rag(
            [
                {"documentId": "ok", "index": 0},
                {"index": 1},  # missing document id
                "not-a-dict",
            ]
        )
        self.assertEqual(len(refs), 1)
        self.assertIsInstance(refs[0], CitationRef)

    def test_error_builder(self):
        result = build_error(
            "Something went wrong.",
            error_code="CHAT_ERROR",
            error_message_safe="Chat processing failed",
        )
        self.assertEqual(result.response_type, "error")
        self.assertEqual(result.completion_status, "failed")
        self.assertEqual(result.error_code, "CHAT_ERROR")

    def test_build_partial_explicit(self):
        result = build_partial("Partial evidence only.")
        self.assertEqual(result.response_type, "partial")
        self.assertTrue(result.partial)


class AgentOutputIntegrationSmokeTests(unittest.TestCase):
    def test_chat_with_ai_turn_adapter_preserves_string_path(self):
        from app import chatbot
        from unittest.mock import patch

        with patch.object(chatbot, "chat_with_ai", return_value="Adapter reply"):
            turn = chatbot.chat_with_ai_turn("hi", user_id="u-41a")
        self.assertEqual(turn.reply, "Adapter reply")
        self.assertEqual(turn.response_type, "answer")


if __name__ == "__main__":
    unittest.main()
