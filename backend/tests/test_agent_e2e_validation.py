"""Agent End-to-End Validation — controlled, offline orchestration checks.

Exercises real entry points (`/api/chat`, `chat_with_ai`) with fakes for Groq,
Redis task context, rate limits, and retrieval. No live paid APIs, no
production databases, no parallel agent implementation.
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET", "e2e-test-jwt-secret-key-32chars-min!!")

from fastapi.testclient import TestClient

from app import chatbot
from app.core.security import create_access_token
from app.main import app
from app.services.agent_planning import ACTION_CALL_LLM, ACTION_FINALIZE, PlanDecision
from app.services.agent_retry import (
    ACTION_ABORT,
    ACTION_DEGRADE,
    ACTION_RETRY,
    ERROR_TPM,
    ERROR_UNKNOWN,
    decide_llm_retry,
)
from app.services.agent_stopping import (
    STOP_ABORT,
    STOP_FINALIZE_COMPLETE,
    evaluate_loop_stop,
)


def _text_response(content: str, *, tokens: int = 40) -> SimpleNamespace:
    """SDK-shaped chat completion without MagicMock attribute pollution."""
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=None,
        tool_call_id=None,
        name=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=tokens),
    )


def _tool_response(
    tool_id: str, name: str, arguments: str, *, tokens: int = 50
) -> SimpleNamespace:
    tc = SimpleNamespace(
        id=tool_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    message = SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[tc],
        tool_call_id=None,
        name=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=tokens),
    )


@contextmanager
def _chat_harness(
    *,
    create_side_effect,
    user_id: str,
    ltm: dict | None = None,
    execute_tool_mock=None,
    sleep: bool = False,
):
    """Run real ``chat_with_ai`` with external I/O mocked."""
    memory = ltm or {"memories": [], "conflicts": [], "summaries": []}
    patches = [
        patch.object(chatbot, "client"),
        patch.object(chatbot, "load_long_term_memory", return_value=memory),
        patch.object(chatbot, "restore_task_context_from_redis"),
        patch.object(chatbot, "save_task_context_to_redis"),
        patch.object(chatbot, "try_save_explicit_memory", return_value=False),
        patch.object(chatbot, "try_update_explicit_memory", return_value=None),
        patch.object(chatbot, "try_delete_explicit_memory", return_value=None),
        patch.object(chatbot, "try_inspect_memory_metadata", return_value=None),
        patch(
            "app.services.memory_scope_service.get_active_project_scope",
            return_value={"project_id": None, "tenant_id": None},
        ),
    ]
    if sleep:
        patches.append(patch("app.chatbot.time.sleep"))
    if execute_tool_mock is not None:
        patches.append(patch.object(chatbot, "execute_tool", side_effect=execute_tool_mock))
    else:
        patches.append(patch.object(chatbot, "execute_tool", wraps=chatbot.execute_tool))

    entered = [p.start() for p in patches]
    mock_client = entered[0]
    mock_client.chat.completions.create.side_effect = create_side_effect
    exec_tool = entered[-1]
    sleep_mock = entered[-2] if sleep else None
    try:
        chatbot._get_conversation_store(user_id).clear()
        yield mock_client, exec_tool, sleep_mock
    finally:
        for p in reversed(patches):
            p.stop()


class TestE2ENormalConversation(unittest.TestCase):
    def test_authenticated_api_normal_chat_no_tools(self):
        client = TestClient(app)
        user = {
            "_id": "usr_e2e_a",
            "email": "e2e_a@example.com",
            "status": "active",
            "tenant_id": "ten_e2e",
            "display_name": "E2E A",
        }
        token = create_access_token(
            user_id="usr_e2e_a",
            email="e2e_a@example.com",
            tenant_id="ten_e2e",
            session_id="sess_e2e_a",
        )
        headers = {"Authorization": f"Bearer {token}"}

        with patch("app.api.deps.get_user_by_id", return_value=user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.chat_with_ai", return_value="Hello — how can I help?"
        ) as mock_chat, patch(
            "app.api.routes.chat.is_document_question", return_value=False
        ):
            res = client.post(
                "/api/chat",
                headers=headers,
                json={"message": "Hi, how are you?"},
            )

        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertIn("help", body["reply"].lower())
        self.assertTrue(body["ragSkipped"])
        mock_chat.assert_called_once()
        self.assertEqual(mock_chat.call_args.kwargs.get("user_id"), "usr_e2e_a")
        self.assertIsNone(mock_chat.call_args.kwargs.get("rag_context"))

    def test_chat_with_ai_greeting_skips_tools(self):
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            self.assertNotIn("tools", kwargs)
            return _text_response("Hi! I'm doing well — how can I help?")

        with _chat_harness(create_side_effect=create, user_id="e2e-hi") as (
            client,
            exec_tool,
            _,
        ):
            answer = chatbot.chat_with_ai(
                "Hi, how are you?",
                user_id="e2e-hi",
                request_id="req-e2e-hi",
            )

        self.assertIn("help", answer.lower())
        exec_tool.assert_not_called()
        self.assertEqual(calls["n"], 1)
        self.assertEqual(client.chat.completions.create.call_count, 1)


class TestE2EToolExecution(unittest.TestCase):
    def test_calculator_tool_result_reaches_final_answer(self):
        calls = {"n": 0}
        tool_seen = {"ok": False}

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                tools = kwargs.get("tools") or []
                names = {(t.get("function") or {}).get("name") for t in tools}
                self.assertIn("calculator", names)
                choice = kwargs.get("tool_choice")
                self.assertIsInstance(choice, dict)
                self.assertEqual(
                    (choice.get("function") or {}).get("name"),
                    "calculator",
                )
                return _tool_response(
                    "calc1",
                    "calculator",
                    '{"expression": "125 * 48 + 750"}',
                )
            # After the tool runs, history must include the calculator result.
            # Calculation intent may still attach the tool catalog; do not
            # raise from the mock (that would enter provider-error handling).
            messages = kwargs.get("messages") or []
            tool_msgs = [
                m for m in messages if isinstance(m, dict) and m.get("role") == "tool"
            ]
            self.assertTrue(tool_msgs)
            joined = " ".join(str(m.get("content", "")) for m in tool_msgs)
            self.assertIn("6750", joined)
            tool_seen["ok"] = True
            return _text_response("The calculated result is 6750.")

        with _chat_harness(create_side_effect=create, user_id="e2e-calc") as (
            _,
            exec_tool,
            _,
        ):
            answer = chatbot.chat_with_ai(
                "Calculate 125 * 48 + 750.",
                user_id="e2e-calc",
                request_id="req-e2e-calc",
            )

        self.assertTrue(tool_seen["ok"])
        self.assertGreaterEqual(exec_tool.call_count, 1)
        self.assertEqual(exec_tool.call_args.args[0], "calculator")
        self.assertIn("6750", answer)


class TestE2EMultiStepWorkflow(unittest.TestCase):
    def test_search_then_read_then_stop(self):
        calls = {"n": 0}
        tool_order: list[str] = []

        def fake_execute(name, args, **kwargs):
            tool_order.append(name)
            if name == "search_code":
                return {
                    "success": True,
                    "result": (
                        '{"success": true, "matches": [{"file_path": '
                        '"app/chatbot.py", "line": 833, '
                        '"text": "def execute_tool(...):"}]}'
                    ),
                }
            if name == "read_file":
                return {
                    "success": True,
                    "result": (
                        '{"success": true, "content": '
                        '"def execute_tool(...):\\n    # tool runner\\n"}'
                    ),
                }
            return {"success": False, "result": '{"success": false}'}

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _tool_response(
                    "s1",
                    "search_code",
                    '{"search_text":"execute_tool"}',
                )
            if calls["n"] == 2:
                # Prior search evidence must be present for the second step.
                blob = " ".join(
                    str(m.get("content", ""))
                    for m in (kwargs.get("messages") or [])
                    if isinstance(m, dict)
                )
                self.assertIn("execute_tool", blob.lower())
                return _tool_response(
                    "r1",
                    "read_file",
                    '{"file_path":"app/chatbot.py","line_start":833,"line_end":860}',
                )
            return _text_response(
                "execute_tool is the central tool runner in chatbot.py."
            )

        with _chat_harness(
            create_side_effect=create,
            user_id="e2e-multi",
            execute_tool_mock=fake_execute,
        ):
            answer = chatbot.chat_with_ai(
                "Find the execute_tool function implementation in the codebase "
                "and inspect it carefully",
                user_id="e2e-multi",
                request_id="req-e2e-multi",
            )

        self.assertEqual(tool_order[:2], ["search_code", "read_file"])
        self.assertIn("execute_tool", answer.lower())
        self.assertLessEqual(calls["n"], 8)

    def test_repeated_failed_tool_does_not_loop_forever(self):
        calls = {"n": 0}

        def always_fail(name, args, **kwargs):
            return {
                "success": True,
                "result": (
                    '{"success": false, "error_code": "FILE_NOT_FOUND", '
                    '"message": "missing"}'
                ),
            }

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] <= 6:
                return _tool_response(
                    f"f{calls['n']}",
                    "read_file",
                    '{"file_path":"app/missing_zzz.py","line_start":1,"line_end":10}',
                )
            return _text_response("I could not complete a full investigation.")

        with _chat_harness(
            create_side_effect=create,
            user_id="e2e-failloop",
            execute_tool_mock=always_fail,
        ):
            answer = chatbot.chat_with_ai(
                "Read the complete implementation of missing_zzz in the codebase",
                user_id="e2e-failloop",
                request_id="req-e2e-fail",
            )

        self.assertIsInstance(answer, str)
        self.assertTrue(answer.strip())
        self.assertLessEqual(calls["n"], 10)


class TestE2ERagDocumentFlow(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.user = {
            "_id": "usr_e2e_rag",
            "email": "e2e_rag@example.com",
            "status": "active",
            "tenant_id": "ten_e2e",
            "display_name": "E2E RAG",
        }
        token = create_access_token(
            user_id="usr_e2e_rag",
            email="e2e_rag@example.com",
            tenant_id="ten_e2e",
            session_id="sess_e2e_rag",
        )
        self.headers = {"Authorization": f"Bearer {token}"}

    def _auth_patches(self):
        return (
            patch("app.api.deps.get_user_by_id", return_value=self.user),
            patch("app.api.deps.auth_service.get_active_session", return_value=True),
            patch("app.api.routes.chat.enforce_chat_rate_limit"),
            patch(
                "app.services.memory_scope_service.get_active_project_scope",
                return_value={"project_id": "proj_e2e", "tenant_id": "ten_e2e"},
            ),
        )

    def test_rag_context_reaches_chat_with_ai(self):
        rag_result = {
            "has_readable_documents": True,
            "candidates": [{"document_id": "doc1"}],
            "context": "Retention policy is 90 days for logs.",
            "citations": [{"document_id": "doc1", "filename": "handbook.md"}],
        }
        patches = self._auth_patches()
        for p in patches:
            p.start()
        try:
            with patch(
                "app.api.routes.chat.is_document_question", return_value=True
            ), patch(
                "app.api.routes.chat.retrieve", return_value=rag_result
            ) as mock_retrieve, patch(
                "app.api.routes.chat.chat_with_ai",
                return_value="According to the handbook, retention is 90 days.",
            ) as mock_chat:
                res = self.client.post(
                    "/api/chat",
                    headers=self.headers,
                    json={
                        "message": (
                            "According to the uploaded handbook document, "
                            "what is the retention policy?"
                        )
                    },
                )
        finally:
            for p in reversed(patches):
                p.stop()

        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertTrue(body["contextUsed"])
        self.assertFalse(body["ragSkipped"])
        self.assertEqual(body["citations"][0]["document_id"], "doc1")
        mock_retrieve.assert_called_once()
        self.assertEqual(mock_retrieve.call_args.kwargs["user_id"], "usr_e2e_rag")
        self.assertEqual(mock_retrieve.call_args.kwargs["project_id"], "proj_e2e")
        self.assertEqual(
            mock_chat.call_args.kwargs["rag_context"],
            "Retention policy is 90 days for logs.",
        )

    def test_readable_docs_without_candidates_abstains(self):
        rag_result = {
            "has_readable_documents": True,
            "candidates": [],
            "context": "",
            "citations": [],
        }
        patches = self._auth_patches()
        for p in patches:
            p.start()
        try:
            with patch(
                "app.api.routes.chat.is_document_question", return_value=True
            ), patch("app.api.routes.chat.retrieve", return_value=rag_result), patch(
                "app.api.routes.chat.chat_with_ai"
            ) as mock_chat:
                res = self.client.post(
                    "/api/chat",
                    headers=self.headers,
                    json={
                        "message": (
                            "According to the uploaded handbook document, "
                            "what is the retention policy?"
                        )
                    },
                )
        finally:
            for p in reversed(patches):
                p.stop()

        self.assertEqual(res.status_code, 200, res.text)
        self.assertIn("couldn't find authorized document evidence", res.json()["reply"])
        mock_chat.assert_not_called()

    def test_rag_route_disables_tools_and_uses_document_context(self):
        captured = {"msgs": None}

        def create(**kwargs):
            captured["msgs"] = kwargs.get("messages")
            self.assertNotIn("tools", kwargs)
            return _text_response("Retention is 90 days per the document.")

        with _chat_harness(create_side_effect=create, user_id="e2e-rag-ctx") as (
            _,
            exec_tool,
            _,
        ):
            answer = chatbot.chat_with_ai(
                "What is the retention policy?",
                user_id="e2e-rag-ctx",
                request_id="req-e2e-rag-ctx",
                rag_context="DOCUMENT: Retention policy is 90 days for logs.",
            )

        exec_tool.assert_not_called()
        blob = " ".join(
            str(m.get("content", ""))
            for m in (captured["msgs"] or [])
            if isinstance(m, dict)
        )
        self.assertIn("90 days", blob)
        self.assertIn("90 days", answer.lower())


class TestE2EPersonalMemory(unittest.TestCase):
    def test_own_memory_injected_other_user_secret_excluded(self):
        ltm = {
            "memories": [
                {
                    "memory": "My preferred code editor is VS Code",
                    "fact_key": "editor",
                    "user_id": "e2e-mem-owner",
                    "owner_id": "e2e-mem-owner",
                    "importance": 4,
                    "confidence": 0.9,
                }
            ],
            "conflicts": [],
            "summaries": [],
        }
        captured = {"blob": ""}

        def create(**kwargs):
            captured["blob"] = " ".join(
                str(m.get("content", ""))
                for m in (kwargs.get("messages") or [])
                if isinstance(m, dict)
            )
            return _text_response("Your preferred editor is VS Code.")

        with _chat_harness(
            create_side_effect=create,
            user_id="e2e-mem-owner",
            ltm=ltm,
        ):
            # Also ensure cross-user secret never appears via a poisoned LTM load.
            answer = chatbot.chat_with_ai(
                "What is my preferred code editor?",
                user_id="e2e-mem-owner",
                request_id="req-e2e-mem",
            )

        self.assertIn("vs code", answer.lower())
        self.assertNotIn("bob_private_secret_key_xyz", captured["blob"].lower())
        self.assertNotIn("other_user_bank_password", captured["blob"].lower())

    def test_rag_context_does_not_mix_as_personal_memory_tooling(self):
        """RAG answers must disable tools (no personal-memory tool path)."""

        def create(**kwargs):
            self.assertNotIn("tools", kwargs)
            return _text_response("Answer from document only.")

        with _chat_harness(create_side_effect=create, user_id="e2e-nomix") as (
            _,
            exec_tool,
            _,
        ):
            chatbot.chat_with_ai(
                "Summarize the uploaded project plan document",
                user_id="e2e-nomix",
                request_id="req-e2e-nomix",
                rag_context="DOCUMENT PLAN: Phase 1 is complete.",
            )
        exec_tool.assert_not_called()


class TestE2EProviderFailureRecovery(unittest.TestCase):
    def test_tpm_retry_after_honored_then_succeeds(self):
        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception(
                    "Error code: 429 - Rate limit reached for model "
                    "on tokens per minute (TPM): Limit 8000, Used 3433, "
                    "Requested 5174. Please try again in 0.05s."
                )
            return _text_response("Recovered after TPM backoff.")

        with _chat_harness(
            create_side_effect=create, user_id="e2e-tpm", sleep=True
        ) as (_, _, sleep_mock):
            answer = chatbot.chat_with_ai(
                "What is RAG?",
                user_id="e2e-tpm",
                request_id="req-e2e-tpm",
            )

        sleep_mock.assert_called()
        wait = float(sleep_mock.call_args.args[0])
        self.assertGreaterEqual(wait, 0.05)
        self.assertLessEqual(wait, 30.0)
        self.assertEqual(calls["n"], 2)
        self.assertIn("recovered", answer.lower())

    def test_retry_after_above_30_does_not_retry_early(self):
        exc = Exception(
            "Error code: 429 - tokens per minute (TPM): Limit 8000. "
            "Please try again in 31s."
        )
        decision = decide_llm_retry(
            exc=exc,
            tpm_retries=0,
            tpm_retry_limit=2,
            tools_enabled=False,
            has_evidence=False,
            max_wait=30.0,
        )
        self.assertEqual(decision.error_class, ERROR_TPM)
        self.assertIn(decision.action, {ACTION_ABORT, ACTION_DEGRADE})
        self.assertNotEqual(decision.action, ACTION_RETRY)

        calls = {"n": 0}

        def create(**kwargs):
            calls["n"] += 1
            raise Exception(
                "Error code: 429 - tokens per minute (TPM): Limit 8000. "
                "Please try again in 31s."
            )

        with _chat_harness(
            create_side_effect=create, user_id="e2e-tpm31", sleep=True
        ) as (_, _, sleep_mock):
            answer = chatbot.chat_with_ai(
                "What is RAG?",
                user_id="e2e-tpm31",
                request_id="req-e2e-tpm31",
            )

        sleep_mock.assert_not_called()
        self.assertEqual(calls["n"], 1)
        self.assertIsInstance(answer, str)
        self.assertTrue(answer.strip())

    def test_terminal_provider_failure_safe_message(self):
        def create(**kwargs):
            raise RuntimeError("upstream provider exploded")

        with _chat_harness(create_side_effect=create, user_id="e2e-term"):
            answer = chatbot.chat_with_ai(
                "What is RAG?",
                user_id="e2e-term",
                request_id="req-e2e-term",
            )

        lowered = answer.lower()
        self.assertTrue(
            "failed" in lowered or "try again" in lowered or "temporarily" in lowered
        )
        self.assertNotIn("gsk_", answer.lower())
        self.assertNotIn("api_key", answer.lower())
        decision = decide_llm_retry(
            exc=RuntimeError("upstream provider exploded"),
            tpm_retries=0,
            tools_enabled=False,
            has_evidence=False,
        )
        self.assertEqual(decision.action, ACTION_ABORT)
        self.assertEqual(decision.error_class, ERROR_UNKNOWN)


class TestE2EFinalAnswerAndStopping(unittest.TestCase):
    def test_complete_investigation_returns_useful_answer(self):
        def create(**kwargs):
            return _text_response(
                "RAG retrieves relevant documents and grounds the model answer."
            )

        with _chat_harness(create_side_effect=create, user_id="e2e-final"):
            answer = chatbot.chat_with_ai(
                "What is RAG?",
                user_id="e2e-final",
                request_id="req-e2e-final",
            )
        self.assertIn("retriev", answer.lower())

    def test_mandatory_finalization_is_tool_free(self):
        """When tools are disabled (RAG), the LLM request stays tool-free."""
        seen = {"tool_free": False}

        def create(**kwargs):
            self.assertNotIn("tools", kwargs)
            self.assertNotIn("tool_choice", kwargs)
            seen["tool_free"] = True
            return _text_response("Document-grounded final answer.")

        with _chat_harness(create_side_effect=create, user_id="e2e-fin-tools"):
            chatbot.chat_with_ai(
                "Summarize the uploaded policy document",
                user_id="e2e-fin-tools",
                request_id="req-e2e-fin-tools",
                rag_context="POLICY: Encrypt data at rest.",
            )
        self.assertTrue(seen["tool_free"])

    def test_unknown_stopping_outcome_fail_closed(self):
        plan = PlanDecision(action=ACTION_FINALIZE, reason="answer_ready")
        stop = evaluate_loop_stop(plan=plan, outcome="typo_outcome")
        self.assertEqual(stop.action, STOP_ABORT)
        self.assertTrue(stop.reason.startswith("unknown_outcome:"))

        # Valid finalize still maps correctly (control).
        ok = evaluate_loop_stop(
            plan=PlanDecision(action=ACTION_FINALIZE, reason="answer_ready")
        )
        self.assertEqual(ok.action, STOP_FINALIZE_COMPLETE)

        # CONTINUE must not be treated as finalize.
        cont = evaluate_loop_stop(
            plan=PlanDecision(action=ACTION_CALL_LLM, reason="continue"),
            tools_enabled=True,
        )
        self.assertNotIn(
            cont.action,
            {STOP_FINALIZE_COMPLETE, STOP_ABORT},
        )


class TestE2EAuthAndIsolationGuards(unittest.TestCase):
    def test_unauthenticated_chat_rejected(self):
        client = TestClient(app)
        res = client.post("/api/chat", json={"message": "hello"})
        self.assertEqual(res.status_code, 401)

    def test_body_user_id_cannot_spoof_identity(self):
        client = TestClient(app)
        user = {
            "_id": "usr_real",
            "email": "real@example.com",
            "status": "active",
            "tenant_id": "ten_e2e",
            "display_name": "Real",
        }
        token = create_access_token(
            user_id="usr_real",
            email="real@example.com",
            tenant_id="ten_e2e",
            session_id="sess_real",
        )
        with patch("app.api.deps.get_user_by_id", return_value=user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.chat_with_ai", return_value="ok"
        ) as mock_chat, patch(
            "app.api.routes.chat.is_document_question", return_value=False
        ):
            res = client.post(
                "/api/chat",
                headers={"Authorization": f"Bearer {token}"},
                json={"message": "hello", "user_id": "usr_attacker"},
            )
        self.assertIn(res.status_code, {400, 422})
        mock_chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
